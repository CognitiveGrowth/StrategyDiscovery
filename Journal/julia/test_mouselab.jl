include("mouselab.jl")

prm = Params()
p = Problem(prm)
b = Belief(p)
display(b.weights)

pol = bmps_policy([0., 1, 0, 0])
w = [0.75, 0.25, 0]
b.weights[:] = w
voi1_ = reshape(features(b)[2, :], size(b.matrix))
@assert sum(abs.(voi1_[:, 1] .- voi1_[:, 2])) < 0.01
@assert sum(abs.((voi1_[1, :] ./ voi1_[2, :]) .- w[1] / w[2])) < 0.1
@assert sum(voi1_[3, :]) == 0


prm = Params(n_gamble=7, n_attr=4, cost=0.1)
pol = bmps_policy([0., 0, 0, 1])
p = Problem(prm)
b = Belief(p)
@time rollout(p, pol).steps
