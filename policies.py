import numpy as np
import torch


def epsilon_greedy_policy(action_probs, epsilon=0.1):
    if np.random.rand() < epsilon:
        return np.random.randint(action_probs.size(1))
    else:
        return torch.argmax(action_probs).item()


def pure_stochastic(action_probs, rand_decay):
    # Convert the action probabilities to a multinomial distribution and then sample
    action = torch.distributions.Categorical(action_probs).sample()
    return (
        action.item()
    )  # Assuming you need the action as a Python integer for the environment
