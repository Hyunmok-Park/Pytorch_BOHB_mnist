from bohb import BOHB
import bohb.configspace as cs

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from runner import *

def evaluate(params, budget):
    loss = NeuralNetworkRunner().train(**params, epoch=budget)
    return loss


if __name__ == '__main__':
    lr = cs.CategoricalHyperparameter('lr', [0.001, 0.01, 0.1])
    batch_size = cs.CategoricalHyperparameter('batch_size', [1, 2, 4])

    configspace = cs.ConfigurationSpace([lr, batch_size])
    opt = BOHB(configspace, evaluate, max_budget=100, min_budget=1)
    logs = opt.optimize()
    print(logs)