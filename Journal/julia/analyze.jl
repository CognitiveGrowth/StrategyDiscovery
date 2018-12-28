println("\n========== Setup ==========")
using Distributed
using Glob
using JLD
if length(workers()) == 1
    addprocs(min(30, Sys.CPU_THREADS - 1))
end
println("Running with $(length(workers())) worker processes.")

if !endswith(pwd(), "StrategyDiscovery/Journal/julia")
    @everywhere cd("StrategyDiscovery/Journal/julia")
end
println(pwd())

include("mouselab.jl")
@everywhere include("mouselab.jl")


#%%
println("\n========== Load policies ==========")

function name(prm::Params)
    dispersion = prm.compensatory ? "High" : "Low"
    stakes = prm.reward_dist.μ > 1 ? "High" : "Low"
    "$dispersion Dispersion - $stakes Stakes"
end


using DataStructures: DefaultDict
jobname = "ferret"
files = glob("runs/$jobname/results/opt*.jld")
policies = DefaultDict{Params, Vector{Policy}}(()->[])
scores = DefaultDict{Params, Vector{Float64}}(()->[])
for file in files
    result = load(file, "opt_result")
    prm = result[:prm]
    pol = Policy(result[:theta])
    push!(policies[prm], pol)
    push!(scores[prm], result[:reward])
end

for prm in keys(scores)
    println(name(prm))
    for (pol, s) in zip(policies[prm], scores[prm])
        println("  ", pol.θ, "   ", s)
    end
    println()
end

all_params = collect(keys(scores))
prm = all_params[3]
println(name(prm))
pol = policies[prm][1]

@everywhere begin
    pol = $pol
    prm = $prm
end

#%%
println("\n========== Monte carlo approximation of Q ==========")

function dmcQ(b::Belief, c::Int; n_roll=5000)
    c == TERM && return term_reward(b)
    observed(b, c) && return -Inf
    @everywhere b0, c = $b, $c
    reward = @distributed (+) for i in 1:n_roll
        b1 = deepcopy(b0)
        observe!(b1, c)
        rollout(pol, b1).reward
    end
    reward / n_roll
end

function mcQ(b::Belief, c::Int; n_roll=5000)
    c == TERM && return term_reward(b)
    observed(b, c) && return -Inf
    reward = 0
    for i in 1:n_roll
        b1 = deepcopy(b0)
        observe!(b1, c)
        reward += rollout(pol, b1).reward
    end
    reward / n_roll
end

# @time println(mcQ(b, 1, n_roll=100))
println("Timing dmcQ")
@everywhere include("mouselab.jl")
@time println(dmcQ(Belief(Problem(prm)), 1))

#%%
println("\n========== Regression ==========")

function regression_data(belief_clicks; n_roll=1000)
    φs = Vector{Float64}[]
    vocs = Float64[]
    for (i, (b, c)) in enumerate(belief_clicks)
        show(stdout, "text/plain", b)
        println("\n-> $c")
        if c == TERM
            continue
        end
        i % 25 == 0 && println("Processed $i beliefs")
        tr = term_reward(b)
        φ = features(b)
        for c1 in [c, rand(unobserved(b))]
            push!(φs, φ[:, c1])
            push!(vocs, dmcQ(b, c1; n_roll=n_roll) - tr)
        end
    end
    hcat(φs...)', vocs
end
regression_data(bc[1:1]);

function sample_belief_clicks(n)
    result = Tuple{Belief, Int}[]
    callback(b, c) = push!(result, (deepcopy(b), c))
    while length(result) < n
        p = Problem(prm)
        rollout(p, pol, callback=callback)
    end
    result
end

function write_regression()
    belief_clicks = sample_belief_clicks(5000)
    @time X, y = regression_data(belief_clicks; n_roll=1000)
    β = X \ y
    println("β:    ", β)

    function mae(bc)
        X, y = regression_data(bc; n_roll=50000)
        yhat = X * β
        abs_err = abs.(yhat - y)
        (mean(abs_err), std(abs_err) / √length(bc))
    end

    train_mae = mae(sample(belief_clicks, 1000; replace=false))
    test_mae = mae(sample_belief_clicks(1000))

    using Printf
    @printf("Train MAE: %.4f ± %.5f\n", train_mae...)
    @printf("Test MAE: %.4f ± %.5f\n", test_mae...)
    result = Dict(
        "β" => β,
        "train_mae" => train_mae,
        "test_mae" => test_mae,
        "prm" => prm,
    )
    JLD.save("regression.jld", result)
    result
end

# load_regression() = JLD.load("regression.jld")
# reg = load_regression()

#%% ========== Human data ==========

import CSV
using DataFrames
using DataFramesMeta
import JSON
df = CSV.read("../data/kogwis/trials.csv")

function parse_json!(df, col, T::Type)
    df[col] = map(x -> convert(T, JSON.parse(x)), df[col])
end
parse_json!(df, :clicks, Array{Int})
parse_json!(df, :ground_truth, Array{Float64})
parse_json!(df, :outcome_probs, Array{Float64})

Problem(payoffs, weights, μ, σ) = begin
    prm = Params(compensatory=maximum(weights) < 0.8, reward_dist=Normal(μ, σ))
    problem = Problem(prm, reshape(payoffs, 4, 7), weights)
end
Problem(row::DataFrameRow) = begin
    vars = [:ground_truth, :outcome_probs, :reward_mu, :reward_sigma]
    Problem(values(row[vars])...)
end

df[:problem] = map(Problem, eachrow(df))
df[:cond] = map(p -> name(p.prm), df.problem)
for cs in df.clicks
    cs .+= 1
end


# %% ==================== Regression ====================
function get_belief_clicks(df::DataFrame)
    result = Tuple{Belief, Int}[]
    for row in eachrow(df)
        b = Belief(row.problem)
        for c in row.clicks
            c += 1
            push!(result, (deepcopy(b), c))
            observe!(b, row.problem, c)
        end
    end
    result
end


sdf = @where df :cond .== name(prm)

bc = sdf |> get_belief_clicks
X, y = regression_data(bc[1:10])
b, c = bc[1]

b
regression_data(bc[1:1])

# # #%% ==========  ==========

# predict_voc(b) = features(b)' * β
# p = Problem(Params())
#
# b = Belief(p)
# round.(reshape(predict_voc(b), 4, 7); digits=3)
# v1 = dmcQ(b, 1; n_roll=100000)
# v3 = dmcQ(b, 3; n_roll=100000)
# v1 - v3
#
# pvoc = predict_voc(b)
# pvoc[1] - pvoc[3]
# p.weights
#
#

# println()
# β = [1., 2.]
# X = [1 2; 1 1.; 1 1]
# y = X * β
# X \ y
#
#%% ========== Likelihood ==========
load("regression.jld")

function Qc_list(problem::Problem, clicks)
    b = Belief(problem)
    map(clicks) do c
        display(b); println()
        Q = map(c -> dmcQ(b, c; n_roll=500), computations(problem))
        observe!(b, problem, c)
        (Q, c)
    end
end

function softmax(x)
    ex = exp.(x .- maximum(x))
    ex ./= sum(ex)
    ex
end

function logp((q, c), α)
    log(softmax(α .* q)[c+1])
end

#%% ========== Test likelihood on BMPS itself ==========

prob = Problem(prm)
clicks = rollout(prob, pol).computations
mean
pmap(1:1000) do i
    rollout(prob, pol).rewad
end



@time qc = Qc_list(prob, clicks[1:3])

exp.(logp.(qc, 0.0001))

reshape(qc[1][1][2:end], 4, 7)
qc[1][1][1]

findmax([sum(logp.(qc, α)) for α in 0.01:0.01:1])
dmcQ(b, 1)


# df.reward_mu |> unique
# # @where(df, :reward_mu .== 5) |> size
#
#
# Qc_list(row::DataFrameRow) = Qc_list(Problem(row), row.clicks)
#



# function logp(problem::Problem, clicks)
#     b = Belief(problem)
#     logp(b, c) = begin
#         Q = map(c -> mcQ(b, c), computations(problem))
#     end
#     Q
# end
# logp(row::DataFrameRow) = logp(Problem(row), row.clicks)
# logp(row)

Base.repeat(f::Function, n::Int) = [f() for i in 1:n]
include("mouselab.jl")
using StatsBase

reshape(logp(values(row)...)[2:end], 4, 7)


reward = 0
for step in 1:max_steps

    c = (step == max_steps) ? TERM : π(b)
    if c == TERM
        reward += term_reward(b)
        return (belief=b, reward=reward, n_steps=step)
    else
        reward -= p.prm.cost
        observe!(b, p, c)
    end
end

# convert(Array{Int}, JSON.parse(df.clicks[10]))

# function mcQ(b::Belief, c::Int; n_roll=1000)
#     sum_reward = @distributed (+) for i in 1:n_roll
#         rollout(pol, b).reward
#     end
#     sum_reward / n_roll
# end


b = Belief(Problem(prm))
rollout(pol, b).reward
mcQ(b, 0; n_roll=100)

# b = Belief(p)
# reward s= []
# beliefs = []
# for step in 1:100
#     push!(beliefs, copy(b.matrix))
#     c = (step == max_steps) ? TERM : π(b)
#     if c == TERM
#         push!(rewards, term_reward(b))
#         # return (belief=b, reward=reward, n_steps=step)
#     else
#         push!(rewards, p.prm.cost)
#         observe!(b, p, c)
#     end
#     return beliefs
# end

function main(n_roll=1000)
    println("-"^70)
    for file in glob("runs/onecost/results/opt-?-7-4-5.0-2.994-true-0.01.jld")
    # for file in glob("runs/onecost/results/opt-?-7-4-0.13-0.072-false-0.01.jld")
        @everywhere begin
            result = load($file, "opt_result")
            pol = Policy(result[:theta])
            prm = Params(compensatory=true, reward_dist=exp_dist("high"))
        end
        n_steps = @distributed (+) for i in 1:n_roll
            rollout(Problem(prm), pol, max_steps=200).n_steps
        end
        println(result[:theta], "  ", result[:reward], "  ", n_steps/n_roll)
    end
end

file = glob("runs/onecost/results/opt-?-7-4-0.13-0.072-false-0.01.jld")[1]
result = load(file, "opt_result")
pol = Policy(result[:theta])
prm = Params(compensatory=true, reward_dist=exp_dist("high"))
@everywhere begin
    result = load($file, "opt_result")
    pol = Policy(result[:theta])
    prm = Params(compensatory=true, reward_dist=exp_dist("high"))
end
n_roll = 1000
r = @distributed (+) for i in 1:n_roll
    rollout(Problem(prm), pol, max_steps=200).n_steps
end
println(pol.θ)

x = 1

@everywhere begin
    println($x)
end

pol = Policy(result[:theta])
n_step = rollout(Problem(prm), pol).n_steps for i in 1:100)






file = "$(target)/opt-$seed-$(name(prm)).jld"


save("$(target)/opt-$seed-$(name(prm)).jld", "opt_result", Dict(pairs(result)))

d = load(file, "opt_result")[:theta]

#%%
println("\n========== Variance of Monte Carlo estimate ==========")

function sampleQ(b::Belief, c::Int; n_roll=10)
    c == TERM && return term_reward(b)
    observed(b, c) && return -Inf
    @everywhere b, c = $b, $c
    pmap(1:n_roll) do i
        b = deepcopy(b)
        observe!(b, c)
        rollout(pol, b).reward
    end
end
p = Problem(prm)
qs = sampleQ(Belief(p), argmax(p.weights), n_roll=5000)
σ = std(qs)
sem = 0.01
n = Integer(ceil((σ/sem)^2))
println("σ: $σ")
println("# samples for SEM=$sem: $n")

#%%
println("\n========== Number of computations ==========")
x = @distributed (+) for i in 1:1000
    # rollout(Problem(prm), pol).n_steps - 1
    rollout(Problem(prm), pol).n_steps - 1
end
println(x / 1000)
