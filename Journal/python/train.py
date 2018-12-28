
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings("ignore", 
                        message="The objective has been evaluated at this point before.")
warnings.filterwarnings("ignore", 
                        message="numpy.core.umath_tests is an internal")

from newmouselab import NewMouselabEnv
from evaluation import *
from distributions import Normal


N_JOBS = 24
N_CALLS = 40
N_TRAIN = 500

def hd_dist(attributes):
    dist = [1,]*attributes
    dist[0] = np.random.randint(85,97)
    for i in range(1,attributes-1):
        dist[i] += np.random.randint(0,100-np.sum(dist))
    dist[-1] += 100-np.sum(dist)
    dist = np.around(np.array(dist)/100,decimals=2)
    np.random.shuffle(dist)
    return dist


def ld_dist(attributes):
    constrain = True
    while constrain:
        dist = [np.random.randint(10,50) for _ in range(attributes)]
        dist = np.around(np.array(dist)/sum(dist),decimals=2)
        constrain = np.min(dist) <= 0.10 or np.max(dist) >= 0.40
    np.random.shuffle(dist)
    return dist


gambles = 7
attributes = 4
high_stakes = Normal((9.99+0.01)/2, 0.3*(9.99-0.01))
low_stakes = Normal((0.25+0.01)/2, 0.3*(0.25-0.01))
reward = high_stakes
cost=.03

envs = [NewMouselabEnv(gambles, attributes, reward, cost, alpha=0.15) 
                 for _ in range(N_TRAIN)]


print('Start', datetime.now())
from contexttimer import Timer
with Timer() as t:
    bo_pol, result = bo_policy(envs, max_cost=30, n_jobs=N_JOBS, n_calls=N_CALLS, 
                               n_random_starts=min(10, N_CALLS),
                               verbose=1, normalize_voi=True, return_result=True)
    print('Total time:', t.elapsed)
    print('theta', bo_pol.theta)

import joblib
from toolz import get
import sys
job_id = get(1, sys.argv, 'X')
del result.specs['args']['func']
joblib.dump(result, f'result_{job_id}.pkl')

