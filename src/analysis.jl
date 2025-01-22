function ensemble_average(sim::Simulation)

    data = Observable[]

    for obs in sim.data
        obs_out = typeof(obs)(obs.name,obs.f,typeof(obs.data)(undef,(length(sim.t),1)))
        obs_out.data .= mean(abs.(obs.data),dims=2)
        push!(data,obs_out)
    end

    return data
end

function plot_observables(sims::Union{Simulation,Vector{Simulation}}; labels::Union{Nothing,Vector{String},Vector{LaTeXString}} = nothing)

    p1 = plot(ylabel=L"\langle |\theta_1|\rangle",ylims=(0.,1.),legend=:topleft)
    p2 = plot(ylabel=L"\langle|\theta_2|\rangle",legend=:none,ylims=(0.,1.))
    p3 = plot(ylabel=L"\langle p^2 / 2m\rangle" * " " * L"[\hbar\omega_\mathrm{r}]",legend=:none)
    p4 = plot(ylabel=L"\mathcal{K}",legend=:none,ylims=(2.3,3.))

    for (i,sim) in enumerate(tuple(sims...))
        data = ensemble_average(sim)
        
        if typeof(labels) == Nothing
            legend_label = "N = $(sim.sys.N)"
        else
            legend_label = labels[i]
        end

        θ1_sim = data[findfirst(obs -> obs.name == :θ1,data)].data
        plot!(p1,sim.t,θ1_sim,xlims=(1,sim.tf),label = legend_label, color = i)

        θ2_sim = data[findfirst(obs -> obs.name == :θ2,data)].data
        plot!(p2,sim.t,θ2_sim,xlims=(1,sim.tf), color = i)
    
        kin = data[findfirst(obs -> obs.name == :kinetic_energy,data)].data
        plot!(p3,sim.t,kin,xlims=(1,sim.tf), color = i)
    
        kurtosis = data[findfirst(obs -> obs.name == :kurtosis,data)].data
        plot!(p4,sim.t,kurtosis,xlims=(1,sim.tf), color = i)

        if i == 1
            θ1, θ2 = magnetization(sim.sys)
            θ1 *= sign(θ1_sim[end])
            θ2 *= sign(θ2_sim[end])
            hline!(p1,[θ1],ls=:dash,color=:black,label="")
            hline!(p2,[θ2],ls=:dash,color=:black,label="")
            hline!(p3,[0.25/sim.scaling],ls=:dash,color=:black,label="")
            hline!(p4,[3],ls=:dash,color=:black,label="")
        end
    end

    plot(p1,p2,p3,p4,layout=(2,2),size=(1000,600),margin=5mm,xscale=:log10,xticks=10 .^(0:7),xlabel=L"\kappa t")
    
end

function join_simulations(sim1, sim2; fname = "simulation.jls")

    sys = sim2.sys
    
    u0 = sim1.u0
    u = sim2.u
    
    tf = sim1.tf + sim2.tf
    trajectories = minimum([sim1.trajectories, sim2.trajectories])
    dt = maximum([sim1.dt, sim2.dt])
        
    timesteps = sim1.timesteps + sim2.timesteps
    snapshots = vcat(sim1.snapshots,sim2.snapshots) 
    t = vcat(sim1.snapshots*sim1.dt, sim1.tf .+ sim2.snapshots*sim2.dt)
    
    τ = sim1.τ + sim2.τ
    start = 1
    scaling = sim2.scaling
    device = sim2.device

    data = Observable[]
    for obs1 in sim1.data
        obs2 = sim2.data[findfirst(obs -> obs.name == obs1.name,sim2.data)]
        push!(data,Observable{Matrix{Float64}}(obs1.name,obs1.f,vcat(obs1.data[:,1:trajectories],obs2.data[:,1:trajectories])))
    end

    sim = Simulation(sys,u0,u,tf,dt,t,τ,[start],timesteps,snapshots,scaling,trajectories,device,fname,data)
        
    return sim
end