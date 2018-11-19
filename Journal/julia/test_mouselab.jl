using Test
include("mouselab.jl")

const N_PROBLEM = 5
const N_GAMBLE = 7
const N_ATTR = 4
const N_CELL = N_GAMBLE * N_ATTR
const CELLS = reshape(1:N_CELL, N_ATTR, N_GAMBLE)
const N_SAMPLE = 10000
#%% ========== Helpers ==========
rand_problem() = Problem(Params(
    reward_dist=Normal(randn(), rand()+0.01),
    compensatory=rand([true, false])
))

function rand_belief()
    p = rand_problem()
    b = Belief(p)
    clicks = sample(1:N_CELL, rand(1:N_CELL), replace=false)
    for c in clicks
        observe!(b, c)
    end
    b
end

function observe_all(b)
    b = deepcopy(b)
    for c in unobserved(b)
        observe!(b, c)
    end
    b
end
function observe_all(b, p)
    b = deepcopy(b)
    for c in unobserved(b)
        observe!(b, p, c)
    end
    b
end

@testset "unobserved" begin
    for i in 1:N_PROBLEM
        b = rand_belief()
        for c in unobserved(b)
            @test !observed(b, c)
        end
        @test rand_belief() |> observe_all |> unobserved |> isempty
    end
end

function observe_gamble(b, gamble)
    b = deepcopy(b)
    for c in CELLS[:, gamble]
        if !observed(b, c)
            observe!(b, c)
        end
    end
    b
end

#%% ========== Tests ==========

@testset "term_reward" begin
    for i in 1:N_PROBLEM
        p = rand_problem()
        b = Belief(p)
        @test term_reward(b) ≈ p.prm.reward_dist.μ
    end
    for i in 1:N_PROBLEM
        p = rand_problem()
        b = observe_all(Belief(p), p)
        @test term_reward(b) ≈ maximum(p.weights' * p.matrix)
    end
end
include("mouselab.jl")
@testset "voi1" begin
    for i in 1:N_PROBLEM
        b = rand_belief()
        while isempty(unobserved(b))
            b = rand_belief()
        end
        c = rand(unobserved(b))
        @test !observed(b, c)
        base = term_reward(b)
        mcq = mean(term_reward(observe(b, c)) for i in 1:N_SAMPLE)
        v = voi1(b, c)
        @test v >= 0
        @test mcq - base ≈ v atol=0.01
    end
    # b = Belief(rand_problem())
    # observe!(b, 1)
    # voi1(b, 1) == 0
end


#%%

@testset "vpi" begin
    for i in 1:N_PROBLEM
        b = rand_belief()
        base = term_reward(b)
        mcq = mean(term_reward(observe_all(b)) for i in 1:N_SAMPLE)
        v = vpi(b)
        @test v >= 0
        @test mcq - base ≈ v atol=0.01
    end
end

@testset "voi_gamble" begin
    for i in 1:N_PROBLEM
        # b = rand_belief()
        b = Belief(Problem(Params()))
        base = term_reward(b)
        g = rand(1:N_GAMBLE)
        mcq = mean(term_reward(observe_gamble(b, g)) for i in 1:N_SAMPLE)
        v = voi_gamble(b, g)
        @test v >= 0
        @test mcq - base ≈ v atol=0.01
    end
end


#%% ========== Scratch ==========

# @testset "meta greedy" begin
#     meta_greedy = Policy([0., 1., 0, 0])
#     for i in 1:N_PROBLEM
#         b = rand_belief()
#         expected_return = mean(rollout(meta_greedy, b).reward for i in 1:1000)
#         @test expected_return >= (term_reward(b) - 0.005)
#     end
# end
