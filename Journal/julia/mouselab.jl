using Parameters
using Distributions
import Base
using Memoize

const TERM = 0


# =================== Problem =================== #

@with_kw struct Params
    n_gamble::Int = 2
    n_attr::Int = 3
    reward_dist::Normal = Normal(0, 1)
    dispersion::Float64 = 1.
    cost::Float64 = 0.1
end

struct Problem
    prm::Params
    matrix::Matrix{Float64}
    weights::Vector{Float64}
end
Problem(prm::Params) = begin
    @unpack reward_dist, n_attr, n_gamble, dispersion = prm
    rs = rand(reward_dist, n_attr * n_gamble)
    Problem(
        prm,
        reshape(rs, n_attr, n_gamble),
        rand(Dirichlet(dispersion * ones(n_attr)))
    )
end
payoff(matrix, weights, choice) = sum((weights .* matrix)[:, choice])
payoff(prob::Problem, choice) = payoff(prob.matrix, prob.weights, choice)

struct Belief
    matrix::Matrix{Normal}
    weights::Vector{Float64}
    cost::Float64
end
Belief(p::Problem) = begin
    Belief(
        [p.prm.reward_dist for i in 1:p.prm.n_attr, j in 1:p.prm.n_gamble],
        p.weights,
        p.prm.cost
    )
end
Base.display(b::Belief) = begin
    X = map(b.matrix) do x
        x.σ < 1e-10 ? round(x.μ; digits=2) : 0
    end
    display(X)
end

U(b::Belief) = maximum(b.weights' * mean.(b.matrix))

function observe!(b::Belief, p::Problem, i::Int)
    b.matrix[i] = Normal(p.matrix[i], 1e-20)
end
observed(b::Belief, cell::Int) = b.matrix[cell].σ == 1e-20


# =================== Features =================== #

@memoize samples(d, g, a; n=10000) = rand(d, n)
function voi_map(b::Belief, obs::BitArray{2})
    n_attr, n_gamble = size(b.matrix)
    sample_cell(g, a) = begin
        d = b.matrix[a, g]
        obs[a, g] & (d.σ > 1e-10) ? samples(d, g, a) : [d.μ]
    end
    w = b.weights
    sample_gamble(g) = .+((w[a] * sample_cell(g, a) for a in 1:n_attr)...)
    v = mean(max.((sample_gamble(g) for g in 1:n_gamble)...))
    v - U(b)
end

function vpi(b)
    obs = trues(size(b.matrix))
    voi_map(b, obs)
end

function voi1(b, cell)
    obs = falses(size(b.matrix))
    obs[cell] = true
    voi_map(b, obs)
end

function voi_gamble(b, gamble)
    obs = falses(size(b.matrix))
    obs[:, gamble] .= true
    voi_map(b, obs)
end

function time_features()
    println("-"^70)
    @time voi1(Belief(p), 1); nothing
    @time voi_gamble(b, 1)
    @time vpi(Belief(p)); nothing
    @time features(p, b); nothing
end

const NULL_FEATURES = -1e10 * ones(4)
function features(b::Belief)
    n_attr, n_gamble = size(b.matrix)
    vpi_b = vpi(b)
    voi_gambles = [voi_gamble(b, g) for g in 1:n_gamble]
    phi(cell) = observed(b, cell) ? NULL_FEATURES : [
        -1,
        voi1(b, cell),
        voi_gambles[Int(ceil(cell / n_attr))],
        vpi_b
    ]
    hcat((phi(a) for a in eachindex(b.matrix))...)
end

function test_features()
    p = Problem(Params())
    b = Belief(p)
    for i in eachindex(b.matrix)
        observe!(b, p, i)
    end
    display(features(p, b))
    bmps_policy(Float64[1,1,1,1])(p, b)
end


# ========== Policy ========== #

function bmps_policy(θ::Array{Float64})
    function π(b)
        voc = (θ' * features(b))' .- b.cost
        v, c = findmax(voc)
        v <= 0 ? TERM : c
    end
end

function rollout(p::Problem, π; max_steps=100)
    b = Belief(p)
    reward = 0
    for step in 1:max_steps
        a = (step == max_steps) ? TERM : π(b)
        if a == TERM
            reward += U(b)
            return (belief=b, reward=reward, steps=step)
        else
            reward -= p.prm.cost
            observe!(b, p, a)
        end

    end
end
