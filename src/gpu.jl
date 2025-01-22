# adapting custom structs to GPU

Adapt.@adapt_structure Canonical
to_gpu(sys::Canonical{Int64,Float64}) = Canonical{Int32,Float32}(sys.N,sys.α1,sys.α2,sys.T)

Adapt.@adapt_structure Microcanonical
to_gpu(sys::Microcanonical{Int64,Float64}) = Microcanonical{Int32,Float32}(sys.N,sys.α1,sys.α2,sys.ϵ)

Adapt.@adapt_structure Operators

struct GPUSimulation{S,I,V,F}
    sys::S
    start::V
    timesteps::I
    snapshots::V
    scaling::F
    dt::F
    τ::F
end

Adapt.@adapt_structure GPUSimulation

# hardcoded observables for now as passing them to the GPU as bitstype is not straightforward
struct GPUData{T} 
    θ1::T
    θ2::T
    p2::T
    p4::T
end

Adapt.@adapt_structure GPUData

function GPUData{T}(sim::Simulation) where T
    dims = (length(sim.snapshots), sim.trajectories)
    mat = T(undef,dims)

    θ1 = copy(mat)
    θ2 = copy(mat)
    p2 = copy(mat)
    p4 = copy(mat)

    return GPUData(θ1,θ2,p2,p4)
end

function to_gpu(sim::Simulation)

    sys = to_gpu(sim.sys)
    start = CuArray(Int32.(sim.start))
    timesteps = Int32(sim.timesteps)
    snapshots = CuArray(Int32.(sim.snapshots))
    scaling = Float32(sim.scaling)
    dt = Float32(sim.dt)
    τ = Float32(sim.τ)
    
    return GPUSimulation(sys,start,timesteps,snapshots,scaling,dt,τ) 
end

function to_cpu!(sim::Simulation, u::CuArray{Float32}, dataGPU::GPUData) 

    sim.u .= Array(u)
    data = adapt(Array,dataGPU)

    for (i,(obs,field)) in enumerate(zip(sim.data,fieldnames(GPUData)))
        sim.data[i] = Observable{Matrix{Float64}}(obs.name, obs.f, getfield(data,field))
    end

    return nothing
end

## GPU kernels 

function calculate_observables_gpu!(u, op, N, data, ind)

    # hardcoded observables for now as passing them to the GPU as bitstype is not straightforward

    i = threadIdx().x
    j = blockIdx().x

    # using the sn arrays for temporarily storing p^2 and p^4 values 
    @inbounds op.sn[i,j] = u[N+i,j]^2
    @inbounds op.sn2[i,j] = u[N+i,j]^4
    sync_threads()

    if i == Int32(1)
        data.θ1[ind,j] = op.θ1[j]
    end
    if i == Int32(2)
        data.θ2[ind,j] = op.θ2[j]
    end
    if i == Int32(3)
        data.p2[ind,j] = CUDA.sum(@view op.sn[:,j])/N 
    end
    if i == Int32(4)
        data.p4[ind,j] = CUDA.sum(@view op.sn2[:,j])/N 
    end
    sync_threads() 

    if i == Int32(5)
        data.p4[ind,j] /=  data.p2[ind,j]^2
    end

    return nothing
end

function evolve_gpu!(u, u_bar, du, op, sim, t, snapshot) 

    i = threadIdx().x
    j = blockIdx().x

    N = sim.sys.N
    α1 = sim.sys.α1
    α2 = sim.sys.α2
    F0 = 2f0*sim.scaling*sim.dt

    D1 = Float32(sqrt(α1*sim.dt/N))
    D2 = Float32(2*sqrt(α2*sim.dt/N))

    while t <= snapshot 

        # adjust parameters for ramp simulations
        if sim.τ > 0f0 && t*sim.dt <= sim.τ
            α1 = sim.sys.α1*t*sim.dt/sim.τ
            α2 = sim.sys.α1*t*sim.dt/sim.τ
            D1 = Float32(sqrt(α1*sim.dt/N))
            D2 = Float32(2*sqrt(α2*sim.dt/N))
        end

        # set Wiener process noise
        if i == Int32(1)
            @inbounds op.D1[j] = D1*randn()
        end

        if i == Int32(2)
            @inbounds op.D2[j] = D2*randn()
        end

        ## 1 - intermediate Heun step
 
        # calculate cosines and sines for each particle position
        @inbounds op.cs[i,j] = @fastmath cos(u[i,j])
        @inbounds op.cs2[i,j] = @fastmath cos(2f0*u[i,j])

        @inbounds sn = @fastmath sin(u[i,j])
        @inbounds sn2 = @fastmath sin(2f0*u[i,j])
        @inbounds op.snp[i,j] = sn*u[N+i,j]
        @inbounds op.sn2p[i,j] = sn2*u[N+i,j]

        # calculate order parameters 
        sync_threads() 

        if i == Int32(1)
            @inbounds op.θ1[j] = CUDA.sum(@view op.cs[:,j])/N 
        end
        if i == Int32(2)
            @inbounds op.θ2[j] = CUDA.sum(@view op.cs2[:,j])/N 
        end
        if i == Int32(3)
            @inbounds op.χ1[j] = CUDA.sum(@view op.snp[:,j])/N 
        end
        if i == Int32(4)
            @inbounds op.χ2[j] = CUDA.sum(@view op.sn2p[:,j])/N 
        end

        # calculate forces
        sync_threads() 

        if i == Int32(1)
            @inbounds op.F1[j] = -α1*(op.θ1[j] + 2f0*sim.scaling*op.χ1[j])*sim.dt
        end
        if i == Int32(2)
            @inbounds op.F2[j] = -2f0*α2*(op.θ2[j] + 4f0*sim.scaling*op.χ2[j])*sim.dt
        end

        # update du
        sync_threads() 

        @inbounds du[i,j] += F0*u[N+i,j]
        @inbounds du[N+i,j] += op.F1[j]*sn + op.F2[j]*sn2

        @inbounds du[N+i,j] += op.D1[j]*sn + op.D2[j]*sn2

        # intermediate step
        @inbounds u_bar[i,j] = u[i,j] + du[i,j]
        @inbounds u_bar[N+i,j] = u[N+i,j] + du[N+i,j]

        
        ## 2 - final Heun step
 
        # calculate cosines and sines for each particle position
        @inbounds op.cs[i,j] = @fastmath cos(u_bar[i,j])
        @inbounds op.cs2[i,j] = @fastmath cos(2f0*u_bar[i,j])

        @inbounds sn = @fastmath sin(u_bar[i,j])
        @inbounds sn2 = @fastmath sin(2f0*u_bar[i,j])
        @inbounds op.snp[i,j] = sn*u_bar[N+i,j]
        @inbounds op.sn2p[i,j] = sn2*u_bar[N+i,j]

        # calculate order parameters
        sync_threads() 

        if i == Int32(1)
            @inbounds op.θ1[j] = CUDA.sum(@view op.cs[:,j])/N 
        end
        if i == Int32(2)
            @inbounds op.θ2[j] = CUDA.sum(@view op.cs2[:,j])/N 
        end
        if i == Int32(3)
            @inbounds op.χ1[j] = CUDA.sum(@view op.snp[:,j])/N 
        end
        if i == Int32(4)
            @inbounds op.χ2[j] = CUDA.sum(@view op.sn2p[:,j])/N 
        end

        # calculate forces
        sync_threads() 

        if i == Int32(1)
            @inbounds op.F1[j] = -α1*(op.θ1[j] + 2f0*sim.scaling*op.χ1[j])*sim.dt
        end
        if i == Int32(2)
            @inbounds op.F2[j] = -2f0*α2*(op.θ2[j] + 4f0*sim.scaling*op.χ2[j])*sim.dt
        end

        # update du
        sync_threads() 

        @inbounds du[i,j] += F0*u_bar[N+i,j]
        @inbounds du[N+i,j] += op.F1[j]*sn + op.F2[j]*sn2
    
        @inbounds du[N+i,j] += op.D1[j]*sn + op.D2[j]*sn2

        # final step
        @inbounds u[i,j] += 0.5f0*du[i,j] 
        @inbounds u[N+i,j] += 0.5f0*du[N+i,j] 

        # reset du
        @inbounds du[i,j] = 0f0 
        @inbounds du[N+i,j] = 0f0 

        # adjust time 
        t += Int32(1)
    end

    return nothing
end