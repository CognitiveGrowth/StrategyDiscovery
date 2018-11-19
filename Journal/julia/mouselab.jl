using Parameters
using Distributions
import Base

const TERM = 0  # termination action
# const NULL_FEATURES = -1e10 * ones(4)  # features for illegal computation
const N_SAMPLE = 10000
# =================== Problem =================== #

"Parameters defining a class of mouselab problems."
@with_kw struct Params
    n_gamble::Int = 7
    n_attr::Int = 4
    reward_dist::Normal = Normal(0, 1)
    compensatory::Bool = true
    cost::Float64 = 0.01
end


function weights(prm::Params)
    function sample_p()
        x = [0; rand(prm.n_attr - 1); 1]
        sort!(x)
        p = diff(x)
    end
    cond = prm.compensatory ? (x -> all(0.1 .<= x .<= 0.4)) : (x -> maximum(x) >= 0.85)
    p = sample_p()
    while !cond(p)
        p = sample_p()
    end
    p
end

exp_dist(min, max) = begin
    Normal(
        (max + min) / 2,
        0.3 * (max - min),
        # min, max
    )
end
exp_dist(stakes::String) = begin
    Dict(
        "high" => exp_dist(0.01, 9.99),
        "low" => exp_dist(0.01, 0.25)
    )[stakes]
end

"An individual mouselab problem (with sampled values)"
struct Problem
    prm::Params
    matrix::Matrix{Float64}
    weights::Vector{Float64}
end
Problem(prm::Params) = begin
    @unpack reward_dist, n_attr, n_gamble = prm
    rs = rand(reward_dist, n_attr * n_gamble)
    Problem(
        prm,
        reshape(rs, n_attr, n_gamble),
        weights(prm)
        # rand(Dirichlet(dispersion * ones(n_attr
    )
end
computations(p::Problem) = 0:prod(size(p.matrix))

"A belief about the values for a Problem.
    Currently, weights and cost are assumed to be known exactly.
"
struct Belief
    matrix::Matrix{Distribution}
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
Base.show(io::IO, mime::MIME"text/plain", b::Belief) = begin
    X = map(b.matrix) do d
        d.σ < 1e-10 ? round(d.μ; digits=2) : 0
    end
    show(io, mime, X)
end
"Expected value of terminating computation with a given belief."
term_reward(b::Belief) = maximum(b.weights' * mean.(b.matrix))

"Update a belief by observing the true value of a cell."
function observe!(b::Belief, p::Problem, i::Int)
    @assert b.matrix[i].σ > 1e-10
    b.matrix[i] = Normal(p.matrix[i], 1e-20)
end

"Update a belief by sampling the value of a cell."
function observe!(b::Belief, i::Int)
    @assert b.matrix[i].σ > 1e-10
    val = rand(b.matrix[i])
    # Belief.matrix contains only Normals, so we represent certainty
    # with an extremeley low variance.
    b.matrix[i] = Normal(val, 1e-20)
end

"Returns a new Belief after sampling the value of a cell"
function observe(b::Belief, c::Int)::Belief
    b1 = deepcopy(b)
    observe!(b1, c)
    b1
end

observed(b::Belief, cell::Int) = b.matrix[cell].σ == 1e-20
unobserved(b::Belief) = filter(c -> !observed(b, c), 1:length(b.matrix))

"Value of each gamble according to a belief"
function gamble_values(b::Belief)::Vector{Normal{Float64}}
    sum(b.weights .* b.matrix, dims=1)[:]
end


# =================== Features =================== #

"Define basic arithmetic operations on Normal distributions."
Base.:*(x::Number, n::Normal)::Normal = Normal(x * n.µ, x * n.σ)
Base.:*(n::Normal, x::Number)::Normal = Normal(x * n.µ, x * n.σ)
Base.:+(a::Normal, b::Normal)::Normal = Normal(a.μ + b.μ, √(a.σ^2 + b.σ^2))
Base.:+(a::Normal, x::Number)::Normal = Normal(a.μ + x, a.σ)
Base.:+(x::Number, a::Normal)::Normal = Normal(a.μ + x, a.σ)
Base.zero(x::Normal)::Normal = Normal(0, 1e-20)

"Highest value in x not including x[c]"
function competing_value(x::Vector{Float64}, c::Int)
    tmp = x[c]
    x[c] = -Inf
    val = maximum(x)
    x[c] = tmp
    val
end

"Expected maximum of a normal and a number."
function emax(d::Distribution, c::Float64)
    p_improve = 1 - cdf(d, c)
    p_improve < 1e-10 && return c
    (1 - p_improve)  * c + p_improve * mean(Truncated(d, c, Inf))
end
emax(x::Float64, c::Float64) = max(x, c)

# Nested truncation is not supported by default.
# Truncated(d::Truncated, lower::Float64, upper::Float64) = begin
#     Truncated(d.untruncated, max(lower, d.lower), min(upper, d.upper))
# end

"Value of knowing the true value of a gamble."
function voi_gamble(b::Belief, gamble::Int)
    gamble_dists = gamble_values(b)
    μ = mean.(gamble_dists)[:]
    cv = competing_value(µ, gamble)
    emax(gamble_dists[gamble], cv) - maximum(μ)
end
function voi_gamble(b::Belief, gamble::Int, gamble_dists, μ)
    cv = competing_value(µ, gamble)
    emax(gamble_dists[gamble], cv) - maximum(μ)
end

"Value of knowing the value in a cell."
function voi1(b::Belief, cell::Int)
    gamble_dists = gamble_values(b)
    μ = mean.(gamble_dists)[:]
    return voi1(b, cell, μ)
end
function voi1(b::Belief, cell::Int, μ::Vector{Float64})::Float64
    n_attr, n_gamble = size(b.matrix)
    gamble = Int(ceil(cell / n_attr))
    attr = cell % n_attr
    col = @view b.matrix[:, gamble]
    new_dist = Normal(0, 1e-20)
    for i in 1:n_attr
        d = b.matrix[i, gamble]
        new_dist += b.weights[i] * (i == attr ? d : d.μ)
    end
    cv = competing_value(µ, gamble)
    emax(new_dist, cv) - maximum(μ)
end


"Value of knowing everything."
function vpi(b::Belief)
    gamble_dists = gamble_values(b)
    μ = mean.(gamble_dists)[:]
    mean(max.((rand(d, N_SAMPLE) for d in gamble_dists)...)) - maximum(μ)
end
function vpi(b::Belief, gamble_dists, μ)::Float64
    mean(max.((rand(d, N_SAMPLE) for d in gamble_dists)...)) - maximum(μ)
end

"Features for every computation in a given belief."
function features(b::Belief)
    n_attr, n_gamble = size(b.matrix)
    gamble_dists = gamble_values(b)
    μ = mean.(gamble_dists)[:]
    vpi_b = vpi(b, gamble_dists, μ)
    voi_gambles = [voi_gamble(b, g, gamble_dists, μ) for g in 1:n_gamble]
    phi(cell) = observed(b, cell) ? -1e10 * ones(4) : [
        -1,
        voi1(b, cell, μ),
        voi_gambles[Int(ceil(cell / n_attr))],
        vpi_b
    ]
    phis = [phi(a) for a in eachindex(b.matrix)]
    hcat(phis...)
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
    voc = (π.θ' * features(b))' .- b.cost .+ 1e-10 * rand(length(b.matrix))
    v, c = findmax(voc)
    @assert isfinite(v)
    v <= 0 ? TERM : c
end

"Runs a Policy on a Problem."
function rollout(p::Problem, π::Policy; initial_belief=nothing, max_steps=100, belief_log=nothing)
    b = initial_belief != nothing ? initial_belief : Belief(p)
    reward = 0
    computations = []
    for step in 1:max_steps
        if belief_log != nothing
            push!(belief_log, deepcopy(b))
        end
        c = (step == max_steps) ? TERM : π(b)
        push!(computations, c)
        if c == TERM
            reward += term_reward(b)
            return (belief=b, reward=reward, n_steps=step, computations=computations)
        else
            reward -= p.prm.cost
            observe!(b, p, c)
        end
    end
end

"Runs a Policy starting with a given belief."
function rollout(π::Policy, b::Belief, max_steps=100, belief_log=nothing)
    b = deepcopy(b)
    reward = 0
    computations = []
    for step in 1:max_steps
        if belief_log != nothing
            push!(belief_log, copy(b.matrix))
        end
        c = (step == max_steps) ? TERM : π(b)
        push!(computations, c)
        if c == TERM
            reward += term_reward(b)
            return (belief=b, reward=reward, n_steps=step, computations=computations,
                    belief_log=belief_log)
        else
            reward -= b.cost
            observe!(b, c)
        end
    end
end
