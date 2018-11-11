using Distributed
@everywhere include("mouselab.jl")

import Random
using Dates: now

using PyCall
using LatinHypercubeSampling: LHCoptim

@pyimport skopt
# Optimizer methods
ask(opt)::Vector{Float64} = opt[:ask]()
tell(opt, x::Vector{Float64}, fx::Float64) = opt[:tell](Tuple(x), fx)

# Optimization.
function x2θ(x)
    cost_weight = x[1]
    voi_weights = diff([0; sort(collect(x[2:end])); 1])
    [cost_weight; voi_weights]
end

function max_cost(prm::Params)
    p = Problem(prm)
    θ = Float64[1, 0, 0, 1]
    computes() = bmps_policy(θ)(p, Belief(p)) != TERM

    while computes()
        θ[1] *= 2
    end

    while !computes()
        θ[1] /= 2
        if θ[1] < 2^-10
            error("Computation is too expensive")
        end
    end

    step_size = θ[1] / 100
    while computes()
        θ[1] += step_size
    end
    θ[1]
end

function avg_reward(prm, θ; n_roll=5)
    reward = @distributed (+) for i in 1:n_roll
        π = bmps_policy(θ)
        rollout(Problem(prm), π, max_steps=200).steps
    end
    reward / n_roll
end

function optimize(prm::Params; seed=0, n_iter=200, n_roll=1000, verbose=false)
    function loss(x; nr=n_roll)
        reward = avg_reward(prm, x2θ(x); n_roll=n_roll)
        verbose && println(">> ", reward)
        - reward
    end
    bounds = [ (0., max_cost(prm)), (0., 1.), (0., 1.) ]
    opt = skopt.Optimizer(bounds, random_state=seed)

    # Choose initial samples by Latin Hypersquare sampling.
    upper_bounds = [b[2] for b in bounds]
    n_latin = max(2, cld(n_iter, 4))
    latin_points = LHCoptim(n_latin, length(bounds), 1000)[1]
    for i in 1:n_latin
        x = latin_points[i, :] ./ n_latin .* upper_bounds
        tell(opt, x, loss(x))
    end

    # Bayesian optimization.
    for i in 1:(n_iter - n_latin)
        x = ask(opt)
        tell(opt, x, loss(x))
    end

    # Cross validation.
    best_x = opt[:Xi][sortperm(opt[:yi])][1:cld(n_iter, 20)]  # top 20%
    fx, i = findmin(loss.(best_x; nr=n_roll*10))

    return (theta=x2θ(best_x[i]), reward=-fx, X=opt[:Xi], y=opt[:yi])
end

function main(mouselab_prm; seed=0)
    Random.seed!(0)
    optimize(Params(mouselab_prm...); seed=seed,)
end

prm = Params(n_gamble=7, n_attr=4)
θ = [0.7, 0.8, 0, 0.2]
avg_reward(prm, θ)


# include("mouselab.jl")
# π = bmps_policy([0, 0.3, 0.3, 0.4])
# @time b, r, s = rollout(Problem(prm), π, max_steps=50); nothing
# X = features(p, b)
