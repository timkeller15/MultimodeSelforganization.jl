abstract type System{I,F} end

struct Canonical{I,F} <: System{I,F}
    N::I
    α1::F
    α2::F
    T::F # in units of T0
end

CanonicalSys(; N::Int64 = 50, α1::Float64 = 2.5, α2::Float64 = 0.5, T::Float64 = 1.) = Canonical{Int64,Float64}(N,α1,α2,T)

struct Microcanonical{I,F} <: System{I,F}
    N::I
    α1::F
    α2::F
    ϵ::F # in units of kb*T0 default: 0.5
end

MicrocanonicalSys(; N::Int64 = 50, α1::Float64 = 2.5, α2::Float64 = 0.5, ϵ::Float64 = 0.5) = Microcanonical{Int64,Float64}(N,α1,α2,ϵ)

struct Observable{T}
    name::Symbol
    f::Function
    data::T
end

function Observable{T}(name::Symbol, f::Function, dims::Tuple{Int64, Int64}) where T

    obs = Observable{T}(name,f,T(undef,dims)) 

   return obs
end

struct Simulation
    sys::System
    u0::Matrix{Float64}
    u::Matrix{Float64}
    tf::Float64
    dt::Float64
    t::Vector{Float64}
    τ::Float64 
    start::Vector{Int64} 
    timesteps::Int64
    snapshots::Vector{Int64}
    scaling::Float64
    trajectories::Int64
    device::Symbol
    fname::String
    data::Vector{Observable}
end

function Simulation(sys::System; # system parameters 
    u0::Union{Matrix{Float64},Nothing} = nothing, # initial state
    tf::Float64 = 1e4, # simulation time
    dt::Float64 = 0.1, # time-step size
    τ::Float64 = 0., # duration of pump strength ramp, zero for quench
    measurements::Int64 = 1000, # number of data outputs
    scaling::Float64 = 1/388.79659560406355, # scaling parameter from physical model, see below comment for calculation
    trajectories::Int64 = 100, # number of SDE trajectories to solve
    device::Symbol = :CPU, # :GPU requires CUDA-capable device
    start::Int64 = 1, # pick another data snapshot id for resuming simulation (only for CPU)
    fname::String = "simulation_data.jls" # file name for saving
    ) 
    
    ## Scaling Parameter
    # The scaling parameter ω_rec/κ is the ratio of recoil frequency to cavity decay from the physical model, where ω_rec = 0.5*ħ*k^2/mass
    # Assuming rubidium atoms with a mass = 85*amu, a cavity frequency close to the Rb-85 D2 line with k = 2*π/780e-9 and a cavity decay rate of κ = 2*π*1.5e6, 
    # yields the value of scaling = 388.79659560406355 used throughout the work. 

    # set initial state to thermal if no input is provided
    if typeof(u0) == Nothing
        T0 = 4*scaling
        u0 = initial_state(sys.N, trajectories, sys.T/T0) 
    end
    u = copy(u0)

    # calculate logarithmically spaced timesteps for data output 
    timesteps = Int(round(tf/dt))
    snapshots = Int.(unique(round.((10).^range(0,log10(timesteps),length=measurements))))
    t = snapshots*dt
    
    # set observables to be calculated 
    dims = (length(t),trajectories)
    T = Matrix{Float64}
    F = Float64
    
    data = [Observable{T}(:θ1,(x,p) -> mean(cos.(x),dims=1),dims),
            Observable{T}(:θ2,(x,p) -> mean(cos.(F(2)*x),dims=1),dims),
            Observable{T}(:kinetic_energy,(x,p) -> mean(p.^2,dims=1),dims),
            Observable{T}(:kurtosis,(x,p) -> mean(p.^4,dims=1)./mean(p.^2,dims=1).^2,dims)
            ] 

    sim = Simulation(sys,u0,u,tf,dt,t,τ,[start],timesteps,snapshots,scaling,trajectories,device,fname,data)
    
    return sim
end

struct Operators{T}  
    cs::T
    cs2::T
    sn::T
    sn2::T
    snp::T
    sn2p::T
    θ1::T
    θ2::T
    χ1::T
    χ2::T
    F1::T
    F2::T
    D1::T
    D2::T
end

function Operators{T}(sim::Simulation) where T
    mat = T(undef,(sim.sys.N,sim.trajectories))
    vec = T(undef,(1,sim.trajectories))

    # arrays for storing trigonometric values
    cs = copy(mat) # cos(x)
    cs2 = copy(mat) # cos(2*x)
    sn = copy(mat) # sin(x)
    sn2 = copy(mat) # sin(2*x)
    snp = copy(mat) # sin(x)*p
    sn2p = copy(mat) # sin(2*x)*p

    # arrays for storing force and diffusion values
    if sim.device == :GPU
        F1 = copy(vec) 
        F2 = copy(vec) 
        D1 = copy(vec) 
        D2 = copy(vec)
    else
        F1 = copy(mat)
        F2 = copy(mat)
        D1 = copy(mat)
        D2 = copy(mat)
    end

    # arrays for storing order and drag parameters
    θ1 = copy(vec)
    θ2 = copy(vec)
    χ1 = copy(vec)
    χ2 = copy(vec)

    return Operators(cs,cs2,sn,sn2,snp,sn2p,θ1,θ2,χ1,χ2,F1,F2,D1,D2)
end