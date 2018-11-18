include("mouselab.jl")

prm = Params()
p = Problem(prm)
b = Belief(p)

pol = Policy([0, 0, 0, 1.])
pol(b)
@time pol(b)

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
@profiler bench_voi1_opt()
