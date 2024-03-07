# %% Imports
import torch
import torch.optim as optim
import numpy as np
from maze_env import MazeEnv
from policy_network import PolicyNetwork
from sum_utils import train_agent_REINFORCE, train_agent_PPO

from policies import epsilon_greedy_policy, pure_stochastic
import matplotlib.pyplot as plt
# %% Markdowncell

# %% Deivce setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# %% Init Environment and policy

env = MazeEnv()

num_episodes = 1000
maze_size = (9, 9)

action_dim = env.action_space.n
print(action_dim)
# %% Policy Setup
maze_input_dim = np.prod(maze_size)
pos_input_dim = np.prod(maze_size)
policy_net = PolicyNetwork(maze_input_dim, pos_input_dim, action_dim).to(device)
policy = pure_stochastic
optimizer = optim.Adam(policy_net.parameters(), lr=1e-2)
# %% Training of REINFORCE
# reward_logs = train_agent_REINFORCE(
#     env, policy_net, policy, optimizer, num_episodes, maze_size, device
# )
# %% Ttraining of PPO
reward_logs = train_agent_PPO(
    env, policy_net, policy, optimizer, num_episodes, maze_size, device
)
# %% visu training
xs = [x for x in range(len(reward_logs))]

plt.plot(xs, reward_logs)
plt.show()
# %% Debug returns
# %% [markdown]
