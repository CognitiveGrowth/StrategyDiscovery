include("mouselab.jl")


# prm = Params()
# p = Problem(prm)
# b = Belief(p)
# display(b.weights)
# pol = bmps_policy([0., 1, 0, 0])
# w = [0.75, 0.25, 0]
# b.weights[:] = w
# voi1_ = reshape(features(b)[2, :], size(b.matrix))
# @assert sum(abs.(voi1_[:, 1] .- voi1_[:, 2])) < 0.01
# @assert sum(abs.((voi1_[1, :] ./ voi1_[2, :]) .- w[1] / w[2])) < 0.1
# @assert sum(voi1_[3, :]) == 0


prm = Params(n_gamble=7, n_attr=4, cost=0.1)
pol = bmps_policy([0., 0, 0, 1])
p = Problem(prm)
b = Belief(p)
# @profiler rollout(p, pol).n_steps

function time_features()
    b = Belief(p)
    println("-"^30, " Timing Features ", "-"^30)
    print("voi1       "); @time voi1(b, 1); nothing
    print("voi_gamble "); @time voi_gamble(b, 1)
    print("vpi        "); @time vpi(b); nothing
    print("features   "); @time features(b); nothing
end

time_features()

function test_features()
    p = Problem(Params())
    b = Belief(p)
    for i in eachindex(b.matrix)
        observe!(b, p, i)
    end
    display(features(p, b))
    bmps_policy(Float64[1,1,1,1])(p, b)
end

pol = Policy([0., 0, 0, 1])
print("rollout time "); @time rollout(p, pol).n_steps; nothing
