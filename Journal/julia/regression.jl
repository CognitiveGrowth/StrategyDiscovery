#%%
using JLD
using Printf
import Flux
if !endswith(pwd(), "StrategyDiscovery/Journal/julia")
   cd("StrategyDiscovery/Journal/julia")
end
println(pwd())

using Distributed
addprocs(24)
@everywhere include("mouselab.jl")
# %% ====================  ====================

# data = JLD.load("regression/data-Compensatory - High Stakes - 1.0x Cost.jld");
JOB = MID(true, true, 0.01)
data = JLD.load("regression/rollouts-$JOB.jld");

train = data["train"];
test = data["test"];
prm = data["prm"];

y_train = hcat((t[4] for t in train)...)[:]
y_test = hcat((t[4] for t in test)...)[:]
# %% ====================  ====================

@everywhere function get_x(b::Belief, c::Int, φ::Vector{Float64})
    outcome, gamble = get_index(b, c)
    B = b.weights .* b.matrix
    μ = getfield.(B, :μ)[:]
    σ = getfield.(B, :σ)[:]

    gv = sum(B, dims=1)
    μg = mean.(gv)[:]
    σg = std.(gv)[:]
    samples = [rand(d, N_SAMPLE) for d in gv]
    max_samples = max.(samples...)
    [
        φ;
        std(max_samples);
        mean(max_samples);
        μg[sortperm(-μg)[1:2]];
        σg[sortperm(-μg)[1:2]];
        # μ[c]; σ[c];
        # b.weights[outcome];
        μg[gamble]; σg[gamble];
        length(unobserved(b));
        28 - length(unobserved(b));
    ]
end

b, c, φ, q = rand(train)
c = 3
x = get_x(b, c, φ)

# %% ====================  ====================


function get_X(data)
    xs = pmap(data) do t
        get_x(t[1:3]...)
    end
    hcat(xs...)'[:, :]
end


@time X_train = get_X(train);
X_test = get_X(test)
β = X_train \ y_train
y_hat = X_test * β
err = y_hat - y_test
abs_err = abs.(err)
@printf("Test MAE: %.4f ± %.5f\n", mean(abs_err), std(abs_err) / √length(err))

using HDF5
h5open("regression/test.h5", "w") do f
    @write f X_train
    @write f X_test
    @write f y_train
    @write f y_test
end

h5open("regression/test.h5", "r") do f
    println(length(read(f, "y_test")))
end


h5write("regression/test.h5", "test", X_train)
# %% ====================  ====================


function test_X(X)
    β = X \ y_train
    y_hat = X_test * β
    err = y_hat - y_test
    abs_err = abs.(err)
    return mean(abs_err)
end

function without(i)
    X = deepcopy(X_train)
    X[:, i] .= 0
    test_X(X)
end

w = without.(1:length(x))
rank = sortperm(w .- test_X(X_train))
names[rank]

# %% ====================  ====================
# using GLM


names = [
    :intercept
    :voi1
    :voi_gamble
    :voi_outcome
    :vpi
    :std_max
    :mean_max
    :μg1
    :μg2
    :σg1
    :σg2
    # :μc
    # :σc
    # :w
    :μg_g
    :σg_g
    :unobs
    :obs
]

df = DataFrame(X_train); size(df)
m = lm(X_train, y_train)
stderror(m)
using Printf

# %% ====================  ====================

for (k, v) in zip(names, abs.(coef(m)) .- 2stderror(m))
    @printf "%20s    %s\n" k v
end


# %% ====================  ====================

using Flux: Chain, Dense, relu, train!, ADAM
m = Chain(
    Dense(length(x), 1)
    # Dense(length(x), 32, relu),
    # Dense(32, 1)
)

Flux.params(m)[1]

rows(X::Matrix) = [X[i, :] for i in 1:size(X, 1)]


loss(x, y) = abs(m(x)[1] - y)
train_data = collect(zip(rows(X_train), y_train))
for i in 1:10
    @time Flux.train!(loss, train_data, Flux.ADAM(Flux.params(m)))
    test_data = collect(zip(rows(X_test), y_test))
    yhat = [m(x)[1] for x in rows(X_test)]
    println(mean(abs.(yhat .- y_test)))
end
