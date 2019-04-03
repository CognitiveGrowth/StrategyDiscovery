import pandas as pd
import json
from toolz import *
from glob import glob

def parse(row):
    d = json.loads(row.data)
    decisions = map(int, concat(d['decisions']))
    outcomes = d['outcomes']
    dps = concat(d['decision_problems'])

    for decision, outcome, dp in zip(decisions, outcomes, dps):
        yield {
            'participant': row.Index + 1,
            'decision': decision,
            'outcome': outcome,
            'matrix': [[float(x) for x in xs] for xs in dp['payoffs']],
            'mu': dp['mu'][0],
            'sigma': dp['sigma'][0],
            'probabilities': dp['probabilities'],
            'reveal_order': dp['reveal_order'],
            'click_cost': 0 if d['basic_info']['isNoCost'] else 0.01
        }

files = glob("raw/*")
raw = pd.concat([pd.read_csv(file, sep='\t') for file in files])
raw['data'] = raw.pop('Answer.data')
clean = pd.DataFrame(list(concat(map(parse, raw.itertuples()))))
clean.to_csv('clean.csv')
