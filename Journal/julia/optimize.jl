using Distributed
using JLD
@everywhere include("mouselab.jl")
import Random
using Dates: now
using Printf: @printf
using PyCall
using LatinHypercubeSampling: LHCoptim

@pyimport skopt
# Optimizer methods
ask(opt)::Vector{Float64} = opt[:ask]()
tell(opt, x::Vector{Float64}, fx::Float64) = opt[:tell](Tuple(x), fx)

function x2θ(x)
    cost_weight = x[1]
    voi_weights = diff([0; sort(collect(x[2:end])); 1])
    [cost_weight; voi_weights]
end

function max_cost(prm::Params)
    p = Problem(prm)
    b = Belief(p)
    θ = Float64[1, 0, 0, 1]
    computes() = Policy(θ)(b) != TERM

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

function avg_reward(prm, θ; n_roll=100)
    reward = @distributed (+) for i in 1:n_roll
        π = Policy(θ)
        rollout(Problem(prm), π, max_steps=200).reward
    end
    reward / n_roll
end

function optimize(prm::Params; seed=0, n_iter=100, n_roll=1000, verbose=false)
    function loss(x; nr=n_roll)
        reward, secs = @timed avg_reward(prm, x2θ(x); n_roll=n_roll)
        verbose && @printf "reward = %.3f   seconds = %.3f\n" reward secs
        flush(stdout)
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

function name(prm::Params)
    join(map(string, (
        prm.n_gamble,
        prm.n_attr,
        prm.reward_dist.μ,
        prm.reward_dist.σ,
        prm.compensatory,
        prm.cost
    )), "-")
end

import JSON
read_args(file) = Dict(Symbol(k)=>v for (k, v) in JSON.parsefile(file))

function main(prm; jobname="none", seed=0, opt_args...)
    println(prm)
    target = "runs/$(jobname)/results"
    mkpath(target)
    Random.seed!(seed)
    @time result = optimize(prm; seed=seed, opt_args...)
    println("THETA: ", result.theta)
    file = "$(target)/opt-$seed-$(name(prm)).jld"
    result = Dict(pairs(result))
    result[:prm] = prm
    save(file, "opt_result", result)
    println("Wrote $file")
    result
end

function main(file::String; opt_args...)
    args = read_args(file)
    seed = pop!(args, :seed, 0)
    jobname = pop!(args, :job_name, "none")
    stakes = pop!(args, :stakes)
    args[:reward_dist] = exp_dist(stakes)
    prm = Params(;args...)
    main(prm; jobname=jobname, seed=seed, verbose=true, opt_args...)
end


# main("runs/test/jobs/1.json"; n_roll=10, n_iter=4)



# include("mouselab.jl")
# π = Policy([0, 0.3, 0.3, 0.4])
# @time b, r, s = rollout(Problem(prm), π, max_steps=50); nothing
# X = features(p, b)
