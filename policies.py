import numpy as np
import torch


def epsilon_greedy_policy(action_probs, epsilon=0.1):
    if np.random.rand() < epsilon:
        return np.random.randint(action_probs.size(1))
    else:
        return torch.argmax(action_probs).item()


# Probability Distribution: action_probs should be a tensor containing the
# probability of each action. These probabilities should sum up to 1 across the action space for a given state.
# Sampling: When you call .sample() on a Categorical distribution object, it randomly selects an action based on the provided probabilities.
# This means that an action with a higher probability is more likely to be chosen, but it is not guaranteed. The sampling is stochastic, reflecting the probabilities of the actions.
# Use in Reinforcement Learning: This approach allows for exploration of the action space in reinforcement learning.
# Early in training, a good policy should explore a variety of actions,
# even those with lower immediate expected rewards, to discover potentially better strategies.
# Over time, as the policy improves, the action probabilities should adjust to favor more rewarding actions, but there will always be some level of exploration if the probabilities permit.
# For instance, if your action_probs for a 3-action space is [0.1, 0.8, 0.1],
# the action with the middle index has an 80% chance of being selected, but there's still a 20% chance collectively that one of the other two actions will be sampled.
# To always select the action with the highest probability, you would use something like torch.argmax(action_probs), which deterministically selects the action with the highest probability, effectively removing the stochastic nature of the action selection
# However, this approach is typically used during evaluation of the policy rather than during training, where exploration is necessary.
def pure_stochastic(action_probs, rand_decay):
    # Convert the action probabilities to a multinomial distribution and then sample
    action = torch.distributions.Categorical(action_probs).sample()
    return (
        action.item()
    )  # Assuming you need the action as a Python integer for the environment
