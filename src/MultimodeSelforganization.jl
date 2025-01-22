module MultimodeSelforganization

  # for outputting / analyzing data
  using Printf, Plots, Measures, LaTeXStrings, Statistics

  # for saving / loading simulations
  using Serialization

  # for calculating expected steady-state properties
  using NonlinearSolve, Integrals

  # for GPU capabilities
  using CUDA, Adapt

  include("structs.jl")
  export Simulation, Canonical, CanonicalSys, Microcanonical, MicrocanonicalSys, Observable

  include("gpu.jl")

  include("io.jl")
  export save_data, load_data

  include("analysis.jl")
  export plot_observables, ensemble_average, join_simulations
  
  include("equilibrium.jl")
  export magnetization, free_energy, entropy, canonical_phase_transition

  include("simulation.jl")
  export simulate!
end