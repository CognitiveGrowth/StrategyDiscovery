# %% ==================== Setup ====================

if !endswith(pwd(), "StrategyDiscovery/Journal/julia")
    cd("StrategyDiscovery/Journal/julia")
end

const TEST = !isempty(ARGS) && ARGS[1] == "test"
if TEST
    println("SHORT TEST")
    flush(stdout)
end

const COST = 0.01
const N_ROLL_TEST = TEST ? 10 : 50000
const N_ROLL_TRAIN = TEST ? 10 : 1000

using Distributed
if length(workers()) == 1
    addprocs(min(30, Sys.CPU_THREADS))
end
println("Running with $(length(workers())) worker processes.")

include("model.jl")

const JOB = MID(true, true, 0.01)
const prm = Params(JOB)
println(JOB)
flush(stdout)
# if isempty(ARGS) || endswith(ARGS[1], "json")
#     const JOB = "Compensatory - High Stakes - 1.0x Cost"
# else
#     const JOB = ARGS[1]
# end


# %% ==================== Monte carlo approximation of Q ====================
my_policies = policies[JOB]

@everywhere begin
    my_policies = $my_policies
    prm = $prm
    get_pol(i)::Policy = my_policies[1 + i % length(my_policies)]
end

function mcQ(b::Belief, c::Int; n_roll=5000)
    c == TERM && return term_reward(b)
    observed(b, c) && return -Inf
    @everywhere b0, c = $b, $c
    reward = @distributed (+) for i in 1:n_roll
        b1 = deepcopy(b0)
        observe!(b1, c)
        rollout(get_pol(i), b1).reward - b0.cost
    end
    reward / n_roll
end

mcQ(Belief(Problem(prm)), 1; n_roll=10)
println("Timing mcQ with 5000 rollouts:")
@time mcQ(Belief(Problem(prm)), 1);

# %% ==================== Estimate Q for each belief in dataset ====================

function sample_at_most(xs, n)
    return sample(xs, min(length(xs), n); replace=false)
end

RegressionDatum = Tuple{Belief, Int, Vector{Float64}, Float64}
function regression_data(beliefs::Vector{Belief}; n_roll=10000, n_c=100)
    data = RegressionDatum[]
    for (i, b) in enumerate(beliefs)
        i % 25 == 0 && println("Processed $i beliefs")
        tr = term_reward(b)
        φ = features(b)
        for c in sample_at_most(unobserved(b), n_c)
            voc = mcQ(b, c; n_roll=n_roll) - tr
            push!(data, (b, c, φ[:, c], voc))
        end
    end
    data
end

function write_regression_data()
    println("Computing regression data...")
    sdf = df[df.cond .== [condition(JOB)], :]
    beliefs = [d.b for d in parse_data(sdf)]
    if TEST
        beliefs = beliefs[1:2]
    end
    test_beliefs = sample_at_most(beliefs, 1000)  # subset with higher precision VOC estimate
    println("Training on $(length(beliefs)) beliefs.")
    flush(stdout)
    @time train_data = regression_data(beliefs, n_roll=N_ROLL_TRAIN)
    @time test_data = regression_data(test_beliefs, n_roll=N_ROLL_TEST, n_c=1)
    result = Dict(
        "prm" => prm,
        "train" => train_data,
        "test" => test_data
    )
    mkpath("regression")
    file = "regression/rollouts-$JOB.jld"
    JLD.save(file, result)
    println("Wrote $file")
end
write_regression_data()
