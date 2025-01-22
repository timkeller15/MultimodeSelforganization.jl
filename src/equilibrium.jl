function magnetization_integral(x, p) 
    
    n, θ1, θ2, sys  = p

    if typeof(sys) <: Canonical
        y1 = θ1*sys.α1
        y2 = θ2*sys.α2
        arg = 2*(y1*cos(x) + y2*cos(2*x))/sys.T
    elseif typeof(sys) <: Microcanonical
        y1 = θ1*sys.α1
        y2 = θ2*sys.α2
        arg = (y1*cos(x) + y2*cos(2*x))/(sys.ϵ + y1*θ1 + y2*θ2)
    elseif typeof(sys) == Nothing
        arg = θ1*cos(x) + θ2*cos(2*x) # θ1 := a and θ2 := b in case of entropy calculation
    else
        arg = 0.
    end

    f = cos(n*x)*exp(arg) 

    return f 
end

function equilibrium(x, p)
    
    domain = (0, 2*π) 
    kwargs = (reltol = 1e-3, abstol = 1e-3)
    solver = HCubatureJL()

    if typeof(p) <: System
        θ1, θ2 = x
        params = [(n,x...,p) for n = 0:2]
    else
        θ1, θ2 = p
        params = [(n,x...,nothing) for n = 0:2]
    end

    integrals = [solve(IntegralProblem(magnetization_integral, domain, par), solver; kwargs...) for par in params]

    if reduce(&,[SciMLBase.successful_retcode(getfield(I,:retcode)) for I in integrals])
        y0, y1, y2 = [getfield(I,:u) for I in integrals]
    else
        y0, y1, y2 = 1., 0., 0.
    end
    
    cond1 = θ1 - y1/y0
    cond2 = θ2 - y2/y0

    return [cond1, cond2]
end


function magnetization(sys::System; u0::Vector{Float64} = [0.5, 0.5]) 
    
    prob = NonlinearProblem(equilibrium,u0,sys)
    sol = solve(prob)
    θ = [0., 0.]

    if SciMLBase.successful_retcode(sol)
        θ = sol.u
    end

    return θ 
end

function free_energy(x::Vector{Float64}, sys::System)

    y = sqrt.([sys.α1, sys.α2]).*x

    domain = (0, 2*π) 
    kwargs = (reltol = 1e-3, abstol = 1e-3)
    solver = HCubatureJL()

    sol = solve(IntegralProblem(magnetization_integral, domain, (0,x...,sys)), solver; kwargs...) 

    if SciMLBase.successful_retcode(sol)
        I = sol.u
    else
        I = 1.
    end

    F = y'*y - sys.T*log(I) # + const

    return F
end

function entropy(x::Vector{Float64}, sys::System) 

    θ1, θ2 = x
    α1 = sys.α1
    α2 = sys.α2

    # magnetizations limited by constant energy in microcanonical ensemble 
    bounds = (2*θ1^2 - 1) < θ2

    domain = (0, 2*π) 
    kwargs = (reltol = 1e-3, abstol = 1e-3)
    solver = HCubatureJL()

    S = NaN

    if bounds
        u0 = [1., 1.]
        sol = solve(NonlinearProblem(equilibrium,u0,x),TrustRegion()) 
        if SciMLBase.successful_retcode(sol)
            a,b = sol.u
            sol = solve(IntegralProblem(magnetization_integral, domain, (0,a,b,nothing)), solver; kwargs...) 
            if SciMLBase.successful_retcode(sol)
                I = sol.u
                S = 0.5*log(sys.ϵ + α1*θ1^2 + α2*θ2^2) - a*θ1 - b*θ2 + log(I) + 0.5 + 0.5*log(4*pi) 
            end
        end
    end

    return S
end

function canonical_phase_transition(α1::Float64, α2::Float64)
    
    paramagnetic = [0., 0.]
    ferromagnetic = [1., 1.]
    nematic = [0., 1.]
    
    if α2 > 1.
        sys = CanonicalSys(α1 = 0., α2 = α2)
        F1 = free_energy(magnetization(sys; u0 = nematic),sys)
    else
        # sys = CanonicalSys(α1 = 0., α2 = 0.)
        # F1 = free_energy(magnetization(sys; u0 = paramagnetic),sys)

        # setting F1 directly to its analytical value in the paramagnetic phase yields more consistent results
        F1 = -log(2*π)
    end

    # setting an initial value for F2 deep in the ferromagnetic phase
    sys = CanonicalSys(α1 = 3., α2 = 3.)
    F2 = free_energy(magnetization(sys; u0 = ferromagnetic),sys)

    dα = 1e-4
    diff = 1e-6
    repmax = Int(1/dα)
    reps = 0
    
    while abs(F1-F2) > diff && reps < repmax && α1 > 0 
        sys = CanonicalSys(α1 = α1, α2 = α2)
        F2 = free_energy(magnetization(sys; u0 = ferromagnetic),sys) 
        α1 -= dα
        reps +=1
    end

    return α1
end