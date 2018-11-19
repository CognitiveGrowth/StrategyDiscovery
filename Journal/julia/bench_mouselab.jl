include("mouselab.jl")
using Profile
using Test
prm = Params()
p = Problem(prm)
b = Belief(p)
pol = Policy([0, 0, 0, 1.])

#%%
function bench_voi1()
    for i in 1:100000
        voi1(b, 1)
    end
end
function bench_voi1_opt()
    gamble_dists = gamble_values(b)
    μ = mean.(gamble_dists)[:]
    for i in 1:100000
        voi1(b, 1, μ)
    end
end
@time bench_voi1()
@time bench_voi1_opt()
# @profiler bench_voi1_opt()

#%%
include("mouselab.jl")
function bench_voi_gamble()
    gamble_dists = gamble_values(b)
    μ = mean.(gamble_dists)[:]
    for i in 1:100000
        voi_gamble(b, 1, gamble_dists, μ)
    end
end
@time bench_voi_gamble()
# @profiler bench_voi_gamble()

#%%
include("mouselab.jl")
function bench_vpi(;n=1000)
    gamble_dists = gamble_values(b)
    μ = mean.(gamble_dists)[:]
    for i in 1:n
        vpi(b, gamble_dists, μ)
    end
end


@time bench_vpi(n=1000)
Profile.init(delay=0.01)
Profile.clear()
@profiler bench_vpi(n=1000)
Profile.print(format=:flat)
ProfileView.view()

#%%
function bench_pol(b)
    for i in 1:10
        pol(b)
    end
end
bench_pol(b)
@time bench_pol(b)
# @profiler bench_pol(b)

#%%
include("mouselab.jl")
function bench_roll(;n=10)
    r = 0.
    for i in 1:n
        r += rollout(pol, b).reward
    end
    r / n
end
bench_roll()
@time bench_roll();
# @profiler bench_pol(b)


#%%
gamble_dists = gamble_values(b)
μ = mean.(gamble_dists)[:]

function test()
    @time vpi(b, gamble_dists, μ)
    @time samples = [rand(d, N_SAMPLE) for d in gamble_dists]
    @time mean(max.(samples...)) - maximum(μ)
end
@time vpi(b)


#%%
gamble_dists = gamble_values(b)
μ = mean.(gamble_dists)[:]
const RANDN = randn(7, N_SAMPLE)
function vpi2(b, gamble_dists, μ)
    μ_g = [d.μ for d in gamble_dists]
    σ_g = [d.σ for d in gamble_dists]
    mean(maximum(μ_g .+ RANDN .* σ_g, dims=1)) - maximum(μ)
end
@time [vpi2(b, gamble_dists, μ) for i in 1:100];
@time [vpi(b, gamble_dists, μ) for i in 1:100];


μ_g = [d.μ for d in gamble_dists]
σ_g = [d.σ for d in gamble_dists]


x = randn(100)
y = randn(100)
d = gamble_dists[1]
x = randn(10)
y = randn(10)
max.(x, y)
max(x, y)

vpi(b)
