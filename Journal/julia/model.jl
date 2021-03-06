using Distributed
using Glob
using DataStructures: DefaultDict
using Printf
import CSV
import JSON
using DataFrames
using DataFramesMeta
using Dates
using Serialization

file = "../data/2conditions_costNoCost_210subjects_100uniqueTrials/experiment_results.csv"
raw_data = CSV.File(file; header=0)



@everywhere include("mouselab.jl")

const JOBNAME = "jaguar"

import Base
Base.map(f, d::AbstractDict) = [f(k,v) for (k,v) in d]
change_cost(b::Belief, cost) = Belief(b.matrix, b.weights, cost)

# %% ==================== Model ID ====================


struct MID
    compensatory::Bool
    high_stakes::Bool
    cost::Float64
end
MID(prm::Params) = MID(
    prm.compensatory,
    prm.reward_dist.μ > 1,
    prm.cost
)
MID(p::Problem) = MID(p.prm)
Params(id::MID) = Params(
    compensatory=id.compensatory,
    reward_dist=exp_dist(id.high_stakes ? "high" : "low"),
    cost=id.cost
)

function name(mid::MID)
    comp = mid.compensatory ? "" : "Non-"
    stakes = mid.high_stakes ? "High" : "Low"
    cmult = prm.cost / 0.01
    "$(comp)Compensatory - $stakes Stakes - $(cmult)x Cost"
end

Base.string(id::MID) = @sprintf "%d-%d-%.3f" id.compensatory id.high_stakes id.cost
Base.show(io::IO, id::MID) = print(io, string(id))

# %% ==================== Load policies ====================

function load_policies()
    policies = DefaultDict{MID, Vector{BMPSPolicy}}(()->[])
    for file in glob("runs/$JOBNAME/results/opt*.jld")
        result = open(deserialize, file)
        id = MID(result[:prm])
        println(id)
        push!(policies[id], BMPSPolicy(result[:theta]))
    end
    Dict(policies)
end

const policies = load_policies()



# %% ==================== Load human data ====================

const df = CSV.read("../data/kogwis/trials.csv")

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
df[:mid] = map(MID, df.problem)
for cs in df.clicks
    cs .+= 1
end

Cond = Tuple{Bool, Bool}
function name(cond::Cond)
    comp = cond[1] ? "" : "Non-"
    stakes = cond[2] ? "High" : "Low"
    "$(comp)Compensatory - $stakes Stakes"
end


condition(id::MID) = (id.compensatory, id.high_stakes)
condition(prm::Params) = (prm.compensatory, prm.reward_dist.μ > 1,)
df[:cond] = [(id.compensatory, id.high_stakes) for id in df.mid]

struct Datum
    b::Belief
    c::Int
    cond::Cond
end

Base.show(io::IO, mime::MIME"text/plain", d::Datum) = begin
    comp = d.cond[1] ? "" : "Non-"
    stakes = d.cond[2] ? "High" : "Low"
    println("     $(comp)Compensatory - $stakes Stakes")
    show_belief(d.b, d.c)
end

function simulate(π::Policy, problem::Problem)
    data = Datum[]
    cond = condition(problem.prm)
    rollout(π, problem, callback=(b,c)->push!(data, Datum(deepcopy(b), c, cond)))
    data
end

function parse_data(row::DataFrameRow)
    result = Datum[]
    b = Belief(row.problem)
    for c in row.clicks
        push!(result, Datum(deepcopy(b), c, row.cond))
        observe!(b, row.problem, c)
    end
    push!(result, Datum(b, 0, row.cond))
end
parse_data(df::AbstractDataFrame) = vcat(parse_data.(eachrow(df))...)

function softmax(x)
    ex = exp.(x .- maximum(x))
    ex ./= sum(ex)
    ex
end
