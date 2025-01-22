# saving and loading of simulations via simple serialization
function save_data(sim::Simulation)

    serialize(sim.fname, sim)

    return nothing
end

load_data(fname::String) = deserialize(fname)

# Pretty-printing for custom structs

function Base.show(io::IO, sys::System{I,F}) where {I,F} 
    println(io, "Canonical System: ","N = ", getfield(sys,:N),", α1 = ", getfield(sys,:α1),", α2 = ", getfield(sys,:α2),", T = ", getfield(sys,:T))
end
        
function Base.show(io::IO, obs::Observable{T}) where {T} 
    println(io, "Observable")
    println(io, "  name = ", getfield(obs,:name))
    println(io, "  function = ", getfield(obs,:f))
    println(io, "  data : ", size(getfield(obs,:data)))
end
    
function Base.show(io::IO, sim::Simulation) 
    println(io, getfield(sim,:device), "-Simulation: ", getfield(sim,:fname))
    Base.show(io::IO, sim.sys)
    if getfield(sim,:τ) > 0 
        sim_type = "Ramp with τ = " * @sprintf("%.1e",getfield(sim,:τ))
    else
        sim_type = "Quench"
    end
    println(io, "Type: ", sim_type, " | Trajectories: ", getfield(sim,:trajectories)," | Start: ", getfield(sim,:start)[1])
    println(io, "Tf = ", @sprintf("%.1e",getfield(sim,:tf)), " | dt = ", getfield(sim,:dt), " | Timesteps: ", @sprintf("%.1e",getfield(sim,:timesteps)), " | Snapshots: ", length(getfield(sim,:snapshots)))
    println(io, "Observables: ", [getfield(obs,:name) for obs in getfield(sim,:data)], " | Scaling: ", @sprintf("%.2e",getfield(sim,:scaling)))
end