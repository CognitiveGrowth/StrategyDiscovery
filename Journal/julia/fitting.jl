if !endswith(pwd(), "StrategyDiscovery/Journal/julia")
    cd("StrategyDiscovery/Journal/julia")
end
include("model.jl")
using Optim

using StatsBase
using Plots

# %% ====================  ====================
no_click_pol = Policy([1.; zeros(4)])
meta_greedy_pol = Policy([0., 1., 0., 0., 0.])
gdf = groupby(df, :workerid)

function load_policies()
    policies = DefaultDict{Float64, DefaultDict{Cond, Policy}}(()->
        DefaultDict{Cond, Policy}(()->no_click_pol)
    )
    for file in glob("runs/$JOBNAME/results/opt*.jld")
        result = load(file, "opt_result")
        cost = result[:prm].cost
        cond = condition(result[:prm])
        policies[cost][cond] = Policy(result[:theta])
    end
    policies[Inf]  # this creates the Dict for infinite cost
    Dict(policies)
end

POLICIES = load_policies()
COSTS = collect(keys(POLICIES))
sort!(COSTS)


# %% ====================  ====================

function model_prob(d::Datum, cost::Float64, pol::Policy)
    b = Belief(d.b.matrix, d.b.weights, cost)
    v = [0; voc(pol, b)]
    is_opt = maximum(v) .- v .< 0.001
    is_opt[d.c+1] / sum(is_opt)
end

opt_prob(d, cost) = model_prob(d, cost, POLICIES[cost][d.cond])
greedy_prob(d, cost) = model_prob(d, cost, meta_greedy_pol)
rand_prob(d::Datum) = 1 / (length(unobserved(d.b)) + 1)

logp(ε, p_opt, p_rand) = sum(@. log((1-ε) * p_opt + ε * p_rand))

# %% ==================== Simulate data with metagreedy ====================

function simulate(pol)
    data = Datum[]
    for row in eachrow(df)
        rollout(pol, row.problem, callback=(b,c)->push!(data, Datum(deepcopy(b), c, row.cond)))
        model_prob(data[2], 0.01, pol)
    end
    data
end

greedy_sim = simulate(meta_greedy_pol)
greedy_best_logp = sum(log(model_prob(d, 0.01, meta_greedy_pol)) for d in greedy_sim) / length(greedy_sim)


# %% ====================  ====================

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

# %% ====================  ====================
function softmax(x)
    ex = exp.(x .- maximum(x))
    ex ./= sum(ex)
    ex
end

function value(d::Datum, cost::Float64, pol::Policy)
    b = Belief(d.b.matrix, d.b.weights, cost)
    [0; voc(pol, b)]
end

function soft_p(v, c, p_rand; α, ε)
    p = softmax(α * v)[c]
    max(0, (1-ε) * p + ε * p_rand)
end

function fit_softmax(df::AbstractDataFrame)
    dd = parse_data(df)
    p_rand = rand_prob.(dd)
    lower, upper, init = [0., 1e-5], [10000., 1.], [0.01, 0.99]
    fits = map(COSTS) do cost
        V = [value(d, cost, meta_greedy_pol) for d in dd]
        cs = [d.c+1 for d in dd]
        f(x) = -sum(log(soft_p(V[i], cs[i], p_rand[i], α=x[1], ε=x[2]))
                   for i in 1:length(dd))
        res = optimize(f, lower, upper, init, Fminbox(BFGS()))
        res
    end
    i = argmin([res.minimum for res in fits])
    α, ε = fits[i].minimizer
    return (cost=COSTS[i], ε=ε, α=α,
            logp=-fits[i].minimum, rand_logp=sum(log.(p_rand)))
end

# %% ====================  ====================
@time results = map(1:length(gdf)) do i
    fit_model(gdf[i], opt_prob)
end
@time greedy_results = map(1:length(gdf)) do i
    fit_model(gdf[i], greedy_prob)
end
@time soft_results = map(1:length(gdf)) do i
    fit_softmax(gdf[i])
end

using SplitApplyCombine
k = 2 * 100
res = invert(results)
gres = invert(greedy_results)
sres = invert(soft_results)
println("Optimal: ", 2k - 2sum(res.logp))
println("Greedy:  ", 2k - 2sum(gres.logp))
println("Soft:  ", 600 - 2sum(sres.logp))
println("Random: ", -2sum(res.rand_logp))

all_data = parse_data(df)
round(exp(sum(gres.logp) / length(all_data)), digits=3)
round(exp(sum(sres.logp) / length(all_data)), digits=3)
round(exp(sum(sres.rand_logp) / length(all_data)), digits=3)

# %% ====================  ====================

function payoff(row)
    X = reshape(row.ground_truth, 4, 7)
    # X = reshape(row.ground_truth, 7, 4)
    payoffs = row.outcome_probs' * X
    payoffs[row.decision]
end

function reward(row)
    payoff(row) - 0.01 * row.n_clicks
end
df.payoff = map(payoff, eachrow(df))
df.reward = map(reward, eachrow(df))

using GroupedErrors
@> df begin
    @splitby _.cond
    @across _.trial_index
    @x _.trial_index
    @y _.reward
    @plot plot(legend=:bottomright)
end
png("meta-reward")


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

Base.show(io::IO, mime::MIME"text/plain", d::Datum) = begin
    comp = d.cond[1] ? "" : "Non-"
    stakes = d.cond[2] ? "High" : "Low"
    println("     $(comp)Compensatory - $stakes Stakes")
    show_belief(d.b, d.c)
end

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
