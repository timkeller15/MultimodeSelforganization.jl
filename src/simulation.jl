function simulate!(sim::Simulation) 

    if sim.device == :GPU

        T = CuArray{Float32} 
        u = T(sim.u) 
        u_bar = fill!(T(undef,(2*sim.sys.N,sim.trajectories)),0f0) 
        du = fill!(T(undef,(2*sim.sys.N,sim.trajectories)),0f0) 
        op = Operators{T}(sim)
        data = GPUData{T}(sim)

        # main loop for the simulation
        t = Int32(1)
        for (ind,snapshot) in enumerate(sim.snapshots)
            # to minimize kernel launch times the system is evolved between data snapshots 
            @cuda threads=sim.sys.N blocks=sim.trajectories evolve_gpu!(u, u_bar, du, op, to_gpu(sim), t, Int32(snapshot))
            synchronize()
            @cuda threads=sim.sys.N blocks=sim.trajectories calculate_observables_gpu!(u, op, Int32(sim.sys.N), data, Int32(ind))
            t = Int32(snapshot)
        end

        to_cpu!(sim, u, data)
        save_data(sim)
    else
        T = Matrix{Float64} 
        u = copy(sim.u) 
        u_bar = fill!(T(undef,(2*sim.sys.N,sim.trajectories)),0.) 
        du = fill!(T(undef,(2*sim.sys.N,sim.trajectories)),0.) 
        op = Operators{T}(sim)

        # main loop for the simulation
        for i = sim.start[1]:sim.timesteps 

            evolve!(u,u_bar,du,op,sim,i)
            
            # calculating the observables at the logarithmically spaced time steps
            if i in sim.snapshots 
                x = @view u[1:sim.sys.N,:] # position variables
                p = @view u[sim.sys.N+1:end,:] # momentum variables

                ind = findfirst(isequal(i),sim.snapshots)
                for obs in sim.data
                    obs.data[ind,:] = obs.f(x,p)
                end

                sim.u .= u
                sim.start .= i
                save_data(sim)
            end
        end

        sim.u .= u
        sim.start .= sim.timesteps
        save_data(sim)
    end

    return nothing
end

function evolve!(u::Matrix{Float64}, u_bar::Matrix{Float64}, du::Matrix{Float64}, op::Operators, sim::Simulation, t::Int64) 
      
    set_diffusion!(op,sim,t)

    # intermediate time-step
    update_ops!(u,op,sim) # calculate trigonometric functions of particle positions and order parameters
    drift!(u,du,op,sim,t) # add drift term to du vector
    diff!(du,op,sim) # add diffusion term to du vector

    u_bar .= u .+ du # equivalent to u_bar = u .+ a*dt .+ b*dW
    
    # final time-step, same procedure for updated variables u_bar
    update_ops!(u_bar,op,sim)
    drift!(u_bar,du,op,sim,t)
    diff!(du,op,sim)

    u .+= 0.5*du # final update, equivalent to u .+= 0.5*(a + a_bar)*dt .+ 0.5*(b + b_bar)*dW

    # reset du
    du .= 0. 

    return nothing
end

function set_diffusion!(op::Operators{Matrix{Float64}}, sim::Simulation, t::Int64) 
    N = sim.sys.N
    dt = sim.dt

    # adjust parameters for ramp simulations
    if sim.τ > 0. && t*dt <= sim.τ
        α1 = sim.sys.α1*ramp(sim,t)
        α2 = sim.sys.α2*ramp(sim,t)
    else
        α1 = sim.sys.α1 
        α2 = sim.sys.α2 
    end

    for col in eachcol(op.D1)
        fill!(col,sqrt(α1*dt/N)*randn())
    end

    for col in eachcol(op.D2)
        fill!(col,2*sqrt(α2*dt/N)*randn())
    end

    return nothing
end

function update_ops!(u::Matrix{Float64}, op::Operators{Matrix{Float64}}, sim::Simulation) 
    N = sim.sys.N
    @inbounds x = @view u[1:N,:] # position variables
    @inbounds p = @view u[N+1:end,:] # momentum variables 

    # trigonometric functions for drift term 
    op.cs .= cos.(x)
    op.cs2 .= cos.(2.0*x) 
    op.sn .= sin.(x)
    op.sn2 .= sin.(2.0*x)
    op.snp .= op.sn.*p
    op.sn2p .= op.sn2.*p

    # order / drag parameters 
    op.θ1 .= mean(op.cs,dims=1)
    op.θ2 .= mean(op.cs2,dims=1)
    op.χ1 .= mean(op.snp,dims=1)
    op.χ2 .= mean(op.sn2p,dims=1)

    return nothing
end

function drift!(u::Matrix{Float64}, du::Matrix{Float64}, op::Operators, sim::Simulation, t::Int64)  

    N = sim.sys.N
    F0 = 2.0*sim.scaling*sim.dt 

    set_force!(op,sim,t)

    # position variables, following eq. of the form dx = 2*scaling*p*dt
    @inbounds du[1:N,:] .+= F0*u[N+1:end,:] 

    # momentum variables, following eq. of the form dp = -α*(θ + 2*scaling*χ)*sin(x)*dt 
    @inbounds du[N+1:end,:] .+= op.F1.*op.sn .+ op.F2.*op.sn2 

    return nothing
end

function set_force!(op::Operators, sim::Simulation, t::Int64) 

    # adjust parameters for ramp simulations
    if sim.τ > 0. && t*sim.dt <= sim.τ
        α1 = sim.sys.α1*ramp(sim,t)
        α2 = sim.sys.α2*ramp(sim,t)
    else
        α1 = sim.sys.α1 
        α2 = sim.sys.α2 
    end

    for (i,col) in enumerate(eachcol(op.F1)) 
        F1 = -α1*(op.θ1[i] + 2.0*sim.scaling*op.χ1[i])*sim.dt
        fill!(col,F1) 
    end

    for (i,col) in enumerate(eachcol(op.F2)) 
        F2 = -2.0*α2*(op.θ2[i] + 4.0*sim.scaling*op.χ2[i])*sim.dt
        fill!(col,F2)
    end

    return nothing
end

function diff!(du::Matrix{Float64}, op::Operators{Matrix{Float64}}, sim::Simulation) 

    N = sim.sys.N

    @inbounds du[N+1:end,:] .+= op.D1.*op.sn .+ op.D2.*op.sn2
    
    return nothing
end

function ramp(sim::Simulation, t::Int64)
    # linear ramp of duration τ
    if sim.τ > 0. && t*sim.dt <= sim.τ
        f = t*sim.dt/sim.τ
    else
        f = 1.
    end

    return f
end

function initial_state(N::Int64, trajectories::Int64, T::Float64) 
    # calculate thermal initial state 
    u0 = Matrix{Float64}(undef,(2*N,trajectories))

    # random positions x
    u0[1:N,:] = π*(2*rand(N,trajectories) .- 1.) 

    # momenta p according to normal distribution 
    u0[N+1:end,:] = sqrt(T)*randn(N,trajectories) 

    return u0
end