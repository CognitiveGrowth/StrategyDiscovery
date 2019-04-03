# if !endswith(pwd(), "StrategyDiscovery/Journal/julia")
#     cd("StrategyDiscovery/Journal/julia")
# end
include("model.jl")
using Optim

using StatsBase
using Plots
using SplitApplyCombine
using TypedTables

# %% ====================  ====================
no_click_pol = BMPSPolicy([1.; zeros(4)])
meta_greedy_pol = BMPSPolicy([0., 1., 0., 0., 0.])
gdf = groupby(df, :workerid)

function load_policies()
    policies = DefaultDict{Float64, DefaultDict{Cond, BMPSPolicy
    }}(()->
        DefaultDict{Cond, BMPSPolicy
            }(()->no_click_pol)
    )
    for file in glob("runs/$JOBNAME/results/opt*.jld")
        result = load(file, "opt_result")
        cost = result[:prm].cost
        cond = condition(result[:prm])
        policies[cost][cond] = BMPSPolicy(result[:theta])
    end
    policies[Inf]  # this creates the Dict for infinite cost
    Dict(policies)
end

POLICIES = load_policies()
COSTS = collect(keys(POLICIES))
sort!(COSTS)


# %% ==================== BMPS likelihood ====================

change_cost(b::Belief, cost) = Belief(b.matrix, b.weights, cost)

function model_prob(d::Datum, cost::Float64, pol::Policy)
    b = change_cost(d.b, cost)
    v = [0; voc(pol, b)]
    is_opt = maximum(v) .- v .< 0.001
    is_opt[d.c+1] / sum(is_opt)
end

opt_prob(d, cost) = model_prob(d, cost, POLICIES[cost][d.cond])
greedy_prob(d, cost) = model_prob(d, cost, meta_greedy_pol)
rand_prob(d::Datum) = 1 / (length(unobserved(d.b)) + 1)
logp(ε, p_opt, p_rand) = sum(@. log((1-ε) * p_opt + ε * p_rand))



# %% ==================== Optimal and greedy ====================

function fit_model(df::AbstractDataFrame, model::Function)
    dd = parse_data(df)
    p_rand = rand_prob.(dd)
    lower, upper, init = [0.], [1.], [0.99]
    ε_fits = map(COSTS) do cost
        p_model = model.(dd, cost)
        f(x) = -logp(x[1], p_model, p_rand)
        res = optimize(f, lower, upper, init, Fminbox(BFGS()), autodiff=:forward)
        res
    end
    i = argmin([res.minimum for res in ε_fits])
    return (cost=COSTS[i], ε=ε_fits[i].minimizer[1],
            logp=-ε_fits[i].minimum, rand_logp=sum(log.(p_rand)))
end

@time opt_results = map(1:length(gdf)) do i
    fit_model(gdf[i], opt_prob)
end |> Table
@time greedy_results = map(1:length(gdf)) do i
    fit_model(gdf[i], greedy_prob)
end |> Table

# %% ==================== Directed Cognition ====================

include("directed_cognition.jl")
Base.startswith(x::Vector, prefix::Vector) = begin
    length(prefix) > length(x) && return false
    x[1:length(prefix)] == prefix
end

# We do some tricky stuff here to avoid recomputing dc_options
# repeatedly for different costs and ε values during fitting.
using Memoize
@memoize mem_dc_options(b::Belief) = dc_options(b, cost=0.)
function dc_option_values(b::Belief, cost)
    options = mem_dc_options(b)
    v, cs = invert(options)
    v[1:end-1] -= length.(cs[1:end-1]) * cost  # last one is alway TERM, no cost
    return v, cs
end

function dc_logp(trial::Vector{Datum}; cost=0.01, α=1e10, ε=0.0)
    trial_cs = [d.c for d in trial]
    function rec(i)
        i > length(trial) && return 1.
        d = trial[i]
        tcs = trial_cs[i:end]
        v, cs = dc_option_values(d.b, cost)
        probs = softmax(v * α)
        p_acc = 0.
        for (p, c) in zip(probs, cs)
            if p > 0 && startswith(tcs, c)
                p_acc += (1-ε) * p * rec(i + length(c))
            end
        end
        p_acc += ε * rand_prob(d) * rec(i + 1)
        return p_acc
    end
    log(rec(1))
end

function fit_dc(df::AbstractDataFrame)
    trials = [parse_data(t) for t in groupby(df, :trial_index)]
    lower, upper, init = [0.], [1.], [0.99]
    ε_fits = map(COSTS) do cost
        f(x) = -sum(dc_logp.(trials; ε=x[1], cost=cost))
        optimize(f, lower, upper, init, Fminbox(BFGS()), autodiff=:forward)
    end
    i = argmin([res.minimum for res in ε_fits])
    return (cost=COSTS[i], ε=ε_fits[i].minimizer[1], ε_fits=ε_fits,
            logp=-ε_fits[i].minimum)
end

@time dc_results = map(1:length(gdf)) do i
    @time fit_dc(gdf[i])
end

# %% ==================== Comparison ====================
bic(n, k, ℓ) = log(n)k - 2ℓ

function print_result(name, k, results)
    @printf "%10s %5.1f\n" name 2k - 2sum(invert(results).logp)
end
function print_result(name, aic)
    @printf "%10s %5.1f\n" name aic
end
println("="^80)
print_result("Greedy", k, greedy_results)
print_result("Optimal", k, opt_results)
print_result("DC", k, dc_results)
print_result("Random", -2sum(invert(opt_results).rand_logp))

# %% ====================  ====================
println("Optimal: ", 2k - 2sum(invert(opt_results).logp))
println("Greedy:  ", 2k - 2sum(invert(greedy_results).logp))
println("DC:      ", 2k - 2sum(invert(dc_results).logp))
println("Random:  ", -2sum(invert(opt_results).rand_logp))

# %% ====================  ====================
cost_counts = countmap(getfield.(results, :cost))
bar(COSTS, get.([cost_counts], COSTS, [0]))
for c in COSTS
    @printf "%1.4f %d\n" c get(cost_counts, c, 0)
end

# %% ====================  ====================

n_clicks = [sum(map(length, gdf[i].clicks)) for i in 1:100]
scatter(n_clicks, res.cost)
cor([n_clicks clamp.(res.cost, 0, 1)])[2]

# %% ====================  ====================

function soft_prob(dd, res)
    p_rand = rand_prob.(dd)
    V = [value(d, res.cost, meta_greedy_pol) for d in dd]
    cs = [d.c+1 for d in dd]
    sum(log(soft_p(V[i], cs[i], p_rand[i], α=res.α, ε=res.ε)) for i in 1:length(dd))
end

foo = map(1:100) do subj
    map(eachrow(gdf[subj])) do row
        (trial=row.trial_index+1,
         subj=subj,
         compensatory=row.cond[1],
         high_stakes=row.cond[2],
         logp=soft_prob(parse_data(row), soft_results[subj]))
    end
end |> flatten |> DataFrame

using GroupedErrors
@> foo begin
    @splitby _.compensatory
    @across _.subj
    @x _.trial
    @y _.logp
    @plot plot()
end


row.cond
gdf[subj]
plot(f.(1:20))

subj = 15
dd = parse_data(gdf[subj])
op = opt_prob.(dd, results[subj].cost)
opt_prob.(parse_data(gdf[subj][1, :]), results[subj].cost)

errors = dd[op .== 0]
soft_results[1]


function optimal_action(subj::Int, dd::Datum)
    pol = POLICIES[results[subj].cost][dd.cond]
    pol(dd.b)
end

function present(i)
    show(stdout, "text/plain", dd[i])
    a = optimal_action(subj, dd[i])
    op[i] == 0 && println("ERROR:  Optimal action is ", a)
    nothing
end
# %% ====================  ====================
itr = Iterators.Stateful(1:length(dd))
display(" "); i = popfirst!(itr); present(i)
present(i)
d = dd[i]
pol(d.b)
pol = policies[MID(d.cond..., 0.01)][1]

# %% ==================== Explore error predictions ====================
d = errors[2]
cost = results[subj].cost
pol = POLICIES[cost][d.cond]
b = Belief(d.b.matrix, d.b.weights, cost)
v = [0; voc(pol, b)]
is_opt = maximum(v) .- v .< 0.001

# %% ====================  ====================
subj = 1
function plot_clicks(subj)
    g = DataFrame(gdf[subj])
    g.condition = map(name, g.cond)
    plot(g.n_clicks, group=g.condition, label="")
end

rollout(meta_greedy_pol, dd[1].b).reward

plot_clicks(21)
plot_clicks(31)
plot_clicks(6)

p = Problem(Params(MID(false, true, 0.01)))
pol = POLICIES[0.01][(false, true)]

rollout(pol, Belief(p)).n_steps
# pol = deepcopy(meta_greedy_pol)
# pol = Policy([0., 0., 0., 0., 1.])
mean(rollout(pol, Belief(p)).reward for _ in 1:100)
mean(rollout(meta_greedy_pol, Belief(p)).reward for _ in 1:100)

mean(df[df.cond .== [(true, false)], :].reward)


plot([plot_clicks(i) for i in [6, 17, 21, 31]]...)



# %% ====================  ====================

get_errs = map(1:100) do subj
    dd = parse_data(gdf[subj])
    op = opt_prob.(dd, results[subj].cost)
    dd[op .== 0]
end
errs = flatten(get_errs)
countmap([name(e.cond) for e in errs])


errs = invert(get_errs)

n_c = [length(parse_data(gdf[i])) for i in 1:100]

errs = invert(describe_errors)
res = invert(results)
scatter(errs.n_err ./ n_c, errs.term_err)
scatter(res.ε, errs.term_err)
