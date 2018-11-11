module Mouselab

using Parameters
using Distributions
using Lazy: @>
import Base

const TERM = 0
Base.show(io::IO, d::Normal) = print(io, "N($(round(d.µ; digits=2)), $(d.σ))")
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
gambles(p::Problem) = 1:p.prm.n_gamble
attributes(p::Problem) = 1:p.prm.n_attr

struct Belief
    matrix::Matrix{Normal}
end
Belief(p::Problem) = begin
    Belief(
        [p.prm.reward_dist for i in 1:p.prm.n_attr, j in 1:p.prm.n_gamble]
    )
end
Base.display(b::Belief) = begin
    X = map(b.matrix) do x
        x.σ < 1e-10 ? round(x.μ; digits=2) : 0
    end
    display(X)
end
display(b)

Base.:*(x::Number, n::Normal) = Normal(x * n.µ, x * n.σ)
Base.:+(a::Normal, b::Normal) = Normal(a.μ + b.μ, a.σ + b.σ)
Base.:+(a::Normal, x::Number) = Normal(a.μ + x, a.σ)
Base.:+(x::Number, a::Normal) = Normal(a.μ + x, a.σ)
Base.zero(x::Normal) = Normal(0, 1e-20)

function observe!(b::Belief, p::Problem, i::Int)
    b.matrix[i] = Normal(p.matrix[i], 1e-20)
end
observed(b::Belief, cell::Int) = b.matrix[cell].σ == 1e-20

U(X::Matrix{Normal}) = maximum(sum(mean.(X), dims=1))
U(b::Belief) = U(b.matrix)

function voi_map(b::Belief, obs::BitArray{2})
    X = map(zip(obs, b.matrix)) do (o, d)
        o ? rand(d, 5000) : ones(5000) * mean(d)
    end
    gamble_samples = sum(X, dims=1)
    mean(max.(gamble_samples...))
end

obs = falses(size(b.matrix))
obs[1] = true
samp(i) = obs[i] ? rand(b.matrix[i], 5000) : [b.matrix[i].μ]




@time voi1(Belief(p), 1)
@time vpi(Belief(p))
@time features(p, b); nothing


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

# ========== Policy ==========

function features(p::Problem, b::Belief)
    vpi_b = vpi(b)
    voi_gambles = [voi_gamble(b, g) for g in gambles(p)]
    phi(cell) = [
        -1,
        voi1(b, cell),
        voi_gambles[Int(ceil(cell / p.prm.n_attr))],
        vpi_b
    ]
    hcat((phi(a) for a in eachindex(b.matrix))...)
end

function bmps_policy(θ::Array{Float64})
    function policy(p, b)
        voc = (θ' * features(p, b))' .- p.prm.cost
        v, a = findmax(voc)
        v <= 0 ? TERM : a
    end
end

function rollout(p::Problem, policy; max_steps=10)
    b = Belief(p)
    reward = 0
    for step in 1:max_steps
        a = (step == max_steps) ? TERM : policy(p, b)
        if a == TERM
            reward += U(b)
            return (belief=b, reward=reward, steps=step)
        else
            reward -= p.prm.cost
            observe!(b, p, a)
        end

    end
end

prm = Params(n_gamble=7, n_attr=4)

p = Problem(prm)
b = Belief(p)
features(p, b)
observe!(b, p, 1)
features(p, b)

pol = bmps_policy([0, 0.5, 0, 0.5])
@time b, r, s = rollout(Problem(prm), pol); nothing
display(b)
println(r)

# println(b)
# println(r)
# println(s)
# #
#
#
#
#
#

# maximum(sum((X), dims=1))

end
