using Parameters
using Distributions
import Base
using Memoize

const TERM = 0  # termination action
const NULL_FEATURES = -1e10 * ones(4)  # features for illegal computation
const N_SAMPLE = 10000
# =================== Problem =================== #

"Parameters defining a class of mouselab problems."
@with_kw struct Params
    n_gamble::Int = 2
    n_attr::Int = 3
    reward_dist::Normal = Normal(0, 1)
    dispersion::Float64 = 1.
    cost::Float64 = 0.1
end

"An individual mouselab problem (with sampled values)"
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

"Value of each gamble according to the true cell values OR a belief.
    If x is a Belief, a vector of distributions is returned.
    If x is a Problem, a vector a floats is returned.
"
gamble_values(x) = sum(x.weights .* x.matrix, dims=1)

"A belief about the values for a Problem.
    Currently, weights and cost are assumed to be known exactly.
"
struct Belief
    matrix::Matrix{Normal}
    weights::Vector{Float64}
    cost::Float64
end
"Initial belief for a given problem."
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

"Expected value of terminating computation with a given belief."
term_reward(b::Belief) = maximum(b.weights' * mean.(b.matrix))

"Update a belief by observing the true value of a cell."
function observe!(b::Belief, p::Problem, i::Int)
    b.matrix[i] = Normal(p.matrix[i], 1e-20)
end
"Update a belief by sampling the value of a cell."
function observe!(b::Belief, i::Int)
    val = rand(b.matrix[i])
    # Belief.matrix contains only Normals, so we represent certainty
    # with an extremeley low variance.
    b.matrix[i] = Normal(val, 1e-20)
end
observed(b::Belief, cell::Int) = b.matrix[cell].σ == 1e-20


# =================== Features =================== #

"Define basic arithmetic operations on Normal distributions."
Base.:*(x::Number, n::Normal) = Normal(x * n.µ, x * n.σ)
Base.:+(a::Normal, b::Normal) = Normal(a.μ + b.μ, √(a.σ^2 + b.σ^2))
Base.:+(a::Normal, x::Number) = Normal(a.μ + x, a.σ)
Base.:+(x::Number, a::Normal) = Normal(a.μ + x, a.σ)
Base.zero(x::Normal) = Normal(0, 1e-20)

"Highest value in x not including x[c]"
function competing_value(x::Vector{Float64}, c::Int)
    tmp = x[c]
    x[c] = -Inf
    val = maximum(x)
    x[c] = tmp
    val
end

"Expected maximum of a normal and a number."
function emax(d::Normal, c::Float64)
    p_improve = 1 - cdf(d, c)
    (1 - p_improve)  * c + p_improve * mean(Truncated(d, c, Inf))
end
emax(x::Float64, c::Float64) = max(x, c)

"Value of knowing the true value of a gamble."
function voi_gamble(b::Belief, gamble::Int)
    gamble_dists = gamble_values(b)
    μ = mean.(gamble_dists)[:]
    cv = competing_value(µ, gamble)
    emax(gamble_dists[gamble], cv) - maximum(μ)
end

"Value of knowing the value in a cell."
function voi1(b::Belief, cell::Int)
    n_attr, n_gamble = size(b.matrix)
    gamble = Int(ceil(cell / n_attr))
    attr = cell % n_attr
    col = b.matrix[:, gamble]
    new_dist = sum(b.weights[i] * (i == attr ? d : d.μ)
                   for (i, d) in enumerate(col))
    gamble_dists = gamble_values(b)
    μ = mean.(gamble_dists)[:]
    cv = competing_value(µ, gamble)
    emax(new_dist, cv) - maximum(μ)
end

"Value of knowing everything."
function vpi(b::Belief)
    gamble_dists = gamble_values(b)
    μ = mean.(gamble_dists)[:]
    mean(max.((rand(d, N_SAMPLE) for d in gamble_dists)...)) - maximum(μ)
end

"Features for every computation in a given belief."
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

# ========== Policy ========== #
"A metalevel policy that uses the BMPS features"
struct Policy
    θ::Vector{Float64}
end
"Selects a computation to perform in a given belief.
    e.g. Policy(θ)(b) -> c
"
(π::Policy)(b::Belief) = begin
    voc = (π.θ' * features(b))' .- b.cost
    v, c = findmax(voc)
    v <= 0 ? TERM : c
end

"Runs a Policy on a Problem."
function rollout(p::Problem, π::Policy; max_steps=100, belief_log=nothing)
    b = Belief(p)
    reward = 0
    for step in 1:max_steps
        if belief_log != nothing
            push!(belief_log, copy(b.matrix))
        end
        c = (step == max_steps) ? TERM : π(b)
        if c == TERM
            reward += term_reward(b)
            return (belief=b, reward=reward, n_steps=step)
        else
            reward -= p.prm.cost
            observe!(b, p, c)
        end
    end
end
