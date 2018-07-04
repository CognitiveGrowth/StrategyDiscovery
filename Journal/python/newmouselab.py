import numpy as np
from scipy.stats import norm, dirichlet
import scipy.integrate as integrate
import gym
import random
from functools import lru_cache
from agents import Agent
from evaluation import *
from distributions import *
import matplotlib.pyplot as plt
import pandas as pd

class DistRV(object):
    """An object that """
    def __init__(self, alpha, attributes, ground_truth = None):
        super().__init__()
        self.alpha = alpha
        self.attributes = attributes
        self.num_unobs = attributes
        self.init = np.ones(attributes)*-1
        self.state = self.init
        
        if ground_truth is False:
            self.ground_truth = False
        elif ground_truth is not None:
            self.ground_truth = np.array(ground_truth)
        else:
            self.ground_truth = self.sample_all()
        
    def __repr__(self):
        return 'DistRV(a=' + str(self.alpha) + '): [' + ", ".join(self.print_dist()) + "]" 

    def _reset(self):
        self.state = self.init
        return self.state
    
    def print_dist(self):
        return ['{:.3f}'.format(self.state[i]) if self.state[i] != -1 else 'p' +str(i) for i in range(self.attributes)]
    
    def observe_p(self, i):
        if self.state[i] == -1:
            self.num_unobs -= 1
            if self.ground_truth is not False:
                self.state[i] = self.ground_truth[i]
            else:
                self.state[i] = np.random.beta(self.alpha, self.alpha*(self.num_unobs - 1))
        return self.state
    
    def sample_p(self, i, n=1, expectation = False):
        p_vec = np.repeat(self.state[None,:],n,axis=0)
        if self.num_unobs == 0:
            return p_vec
        if self.state[i] == -1:
            if self.num_unobs == 1:
                p_vec[:,i] = (1-np.sum(p_vec[:,p_vec[0] != -1],1))
            else:
                p_vec[:,i] = np.random.beta(self.alpha, self.alpha*(self.num_unobs - 1),size=n)
        if expectation:
            filler = (1-np.sum(p_vec[:,p_vec[0] != -1],1)[:,None])
            if self.num_unobs > 1:
                filler /= (self.num_unobs - 1)
            p_vec[:,p_vec[0] == -1] = filler
        return np.squeeze(p_vec)
    
    def sample_all(self, n=1):
        p_vec = np.repeat(self.state[None,:],n,axis=0)
        if self.num_unobs == 0:
            return p_vec
        else:
            alpha_vec = np.ones(self.num_unobs)*self.alpha
            sampled_unobs = np.random.dirichlet(alpha_vec,size=n)
            p_vec[:,self.state == -1] = (1-np.sum(self.state[self.state != -1])) * sampled_unobs
        return np.squeeze(p_vec)
    
    def expectation(self):
        p_vec = np.copy(self.state)
        if self.num_unobs != 0:
            p_vec[p_vec == -1] = (1-np.sum(p_vec[p_vec != -1]))/self.num_unobs
        return p_vec

ZERO = PointMass(0)

class NewMouselabEnv(gym.Env):
    """MetaMDP for the Mouselab task."""

    term_state = '__term_state__'
    def __init__(self, gambles=4, attributes=5, reward=None, cost=0,
                 ground_truth=None, alpha=1, sample_term_reward=False, quantization=False):

        self.gambles = gambles # number of gambles
        self.quantization = quantization
        self.attributes = attributes
        self.outcomes = attributes

        self.distRV = DistRV(alpha, attributes, ground_truth = ground_truth)
        self.reward = reward if reward is not None else Normal(1., 1.)
        
        if quantization:
            self.discrete_reward = self.reward.to_discrete(quantization)

        if hasattr(reward, 'sample'):
            self.iid_rewards = True
        else:
            self.iid_rewards = False

        self.cost = - abs(cost)
        self.max = cmax
        self.init_rewards = tuple([self.reward,] * (self.gambles*self.outcomes))
        self.init = (self.distRV, self.init_rewards)

        # self.ground_truth only includes rewards
        # self.distRV.ground_truth has the distribution ground truth
        if ground_truth is False:
            self.ground_truth = False
        elif ground_truth is not None:
            self.ground_truth = np.array(ground_truth)
        else:
            if self.quantization:
                self.ground_truth = np.array([self.discrete_reward.sample() for _ in self.init])
            else:
                self.ground_truth = np.array(list(map(sample, self.init[1])))

        self.sample_term_reward = sample_term_reward
        self.term_action = (self.gambles+1)*self.outcomes
        self.reset()

    def _reset(self):
        self.init[0]._reset()
        self._state = self.init
        grid = np.array(self._state[1]).reshape(self.gambles,self.outcomes)
        self.dist = self.distRV.expectation()
        # tmp: Works only for Normal, possibly generalizable format:
        # self.mus = [expectation(np.sum(self.dist*grid[g])) for g in range(self.gambles)]
        # self.vars = [(np.sum(self.dist*grid[g])).sigma**2 for g in range(self.gambles)]
        self.mus = self.reward.mu*np.ones(self.gambles)
        self.vars = np.sum(self.dist**2*self.reward.sigma**2)*np.ones(self.gambles)
        return self._state

    def _step(self, action):
        self.vpi.cache_clear()
        self.vpi_action.cache_clear()
        if self._state is self.term_state:
            assert 0, 'state is terminal'
            # return None, 0, True, {}
        if action >= self.term_action:
            # self._state = self.term_state
            if self.sample_term_reward:
                if self.ground_truth is not False:
                    best_idx = np.argmax(self.mus)
                    gt_grid = self.ground_truth.reshape(self.gambles,self.outcomes)
                    reward = self.dist.dot(gt_grid[best_idx])
                else:
                    reward = sample(self.term_reward())
            else:
                reward = self.expected_term_reward()
            self.last_state = self._state
            self._state = self.term_state
            done = True
        elif self.term_action > action >= self.attributes:
            if not hasattr(self._state[1][action-self.attributes], 'sample'):  # already observed reward
    #             assert 0, self._state[action]
                reward = 0      
            else:  # observe a new node
                self._state = self._observe(action)
                reward = self.cost
            done = False
        else:
            if not self._state[0].state[action] == -1: # already observed attribute
                reward = 0
            else:  # observe a new attribute
                self._state = self._observe(action)
                reward = self.cost #todo: possibly have a separate cost for p observations
            done = False
        return self._state, reward, done, {}

    def _observe(self, action):
#         print('obs ' + str(action))
        if action >= self.attributes:
            action -= self.attributes
            if self.ground_truth is not False:
                result = self.ground_truth[action]
            elif self.quantization:
                assert hasattr(self._state[action], 'sample')
                result = self.discrete_reward.sample()
            else:
                result = self._state[action].sample()
            s = list(self._state[1])
            gamble = action // self.outcomes
            option = action % self.outcomes
            self.mus[gamble] += self.dist[option]*(result - self.reward.expectation())
            self.vars[gamble] = max(0,self.vars[gamble] - self.dist[option]**2*self.reward.sigma**2)
            # gambles = self.gamble_dists()
            # self.mus = [expectation(g) for g in gambles]
            # self.vars = [variance(g) for g in gambles]
            s[action] = result
            return (self._state[0],tuple(s))
        else:
            # edit so it is a temporary change unless assigned
            self._state[0].observe_p(action)
            self.dist = self._state[0].expectation()
            gambles = self.gamble_dists()
            self.mus = [expectation(g) for g in gambles]
            self.vars = [variance(g) for g in gambles]
            return self._state

    def actions(self, state=None):
        """Yields actions that can be taken in the given state.

        Actions include observing the value of each unobserved node and terminating.
        """
        probs = state[0] if state is not None else self._state[0]
        rewards = state[1] if state is not None else self._state[1]
        if state is self.term_state:
            return
        for i in range(self.attributes):
            if probs.state[i] == -1:
                yield i
        for i, v in enumerate(rewards):
            if hasattr(v, 'sample'):
                yield i + self.attributes
        yield self.term_action

    #todo: update
    def results(self, state, action):
        """Returns a list of possible results of taking action in state.

        Each outcome is (probability, next_state, reward).
        """
        # May not work with p random variables (at least without quantization)
        if action == self.term_action:
            # R = self.term_reward()
            # S1 = Categorical([self.term_state])
            # return cross(S1, R)
            yield (1, self.term_state, self.expected_term_reward(state))
        else:
            for r, p in state[action].to_discrete(self.quantization):
                s1 = list(state[1])
                s1[action] = r
                yield (p, tuple(s1), self.cost)


    def action_features(self, action, state=None):
        state = state if state is not None else self._state
        assert state is not None


        if action == self.term_action:
            return np.array([
                0,
                0,
                0,
                0,
                self.expected_term_reward(state)
            ])
        else:
            gamble = action // self.outcomes
            gamble = -action if gamble == 0 else gamble
            return np.array([
                self.cost,
                self.myopic_voi(action),
                self.vpi_action(gamble),
                self.vpi(),
                self.expected_term_reward(state)
            ])

    def gamble_dists(self, state = None, sample_all = False):
        state = state if state is not None else self._state
        sdist = state[0].sample_all() if sample_all else self.dist
        grid = np.array(state[1]).reshape(self.gambles, self.outcomes)
        return np.dot(grid, sdist)
    
    def print_state(self,state=None):
        state = state if state is not None else self._state
        if state is self.term_state:
            return self.print_state(state = self.last_state)
        return pd.DataFrame(self.grid(state),columns=state[0].print_dist())
    
    def grid(self,state=None):
        state = state if state is not None else self._state
        return np.array(state[1]).reshape(self.gambles,self.outcomes)

    @lru_cache(None)
    def vpi(self):
        sdist = self._state[0].sample_all(2500)
        grid = np.array(self._state[1]).reshape(self.gambles,self.outcomes)
        sampled_gambles = np.vectorize(lambda g: sample(g))(sdist.dot(grid.T))
        samples_max = np.amax(sampled_gambles,1)
        return np.mean(samples_max) - np.max(self.mus)

    @lru_cache(None)
    def vpi_action(self, gamble):
        #E[value if gamble corresponding to action is fully known]
        if gamble > 0:
            gamble -= 1
            mus_wo_g = np.delete(self.mus,gamble)
            k = np.max(mus_wo_g)
            m = self.mus[gamble]
            s = np.sqrt(self.vars[gamble])
            e_higher = integrate.quad(lambda x: x*norm.pdf(x,m,s), k, np.inf)[0]
            e_val = k*norm.cdf(k,m,s) + e_higher
        else:
            action = -1*gamble
            sdist = self._state[0].sample_all(2500)
            grid = np.array(self._state[1]).reshape(self.gambles,self.outcomes)
            rgrid = np.repeat(grid[None,:,:],2500,axis=0)
            rgrid[:,:,action] = np.vectorize(lambda g: sample(g))(rgrid[:,:,action])
            rgrid = np.vectorize(lambda g: expectation(g))(rgrid)
            sampled_gambles = np.einsum('ijk,ik->ij',rgrid,sdist)
            e_val = np.mean(np.amax(sampled_gambles,1))
        return e_val - np.max(self.mus)

    def myopic_voi(self, action):
        #E[value if gamble corresponding to action is fully known]
        if action >= self.attributes:
            action -= self.attributes
            gamble = action // self.outcomes
            outcome = action % self.outcomes
            mus_wo_g = np.delete(self.mus,gamble)
            k = np.max(mus_wo_g)
            m = self.mus[gamble]
            s = self.reward.sigma*self.dist[outcome]
            e_higher = integrate.quad(lambda x: x*norm.pdf(x,m,s), k, np.inf)[0]
            e_val = k*norm.cdf(k,m,s) + e_higher
        else:
            egrid = (np.vectorize(lambda g: expectation(g)) #may need to add otypes=[float]
                (np.array(self._state[1]))).reshape(self.gambles,self.outcomes)
            sdist = self._state[0].sample_p(action, n = 2500, expectation = True)
            smus = sdist.dot(egrid.T)
            e_val = np.mean(np.amax(smus,1))
        return e_val - np.max(self.mus)

    def term_reward(self, state=None):
        state = state if state is not None else self._state
        grid = np.array(state[1]).reshape(self.gambles,self.outcomes)
        best_idx = np.argmax(self.mus)
        return self.dist.dot(grid[best_idx])

    def expected_term_reward(self, state=None):
        state = state if state is not None else self._state
        return max(map(expectation, self.gamble_dists(state)))
