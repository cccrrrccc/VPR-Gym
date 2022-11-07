import numpy as np
import torch

def gen_softmax_exploration_policy(tau, weights):
    def policy_function(Q):
        probs = torch.exp(Q / tau)
        probs = probs / torch.sum(probs)
        probs = torch.mul(probs, weights)
        action = torch.multinomial(probs, 1).item()
        return action
    return policy_function