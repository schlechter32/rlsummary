# %% Imports
import torch
import torch.optim as optim
import numpy as np
from maze_env import MazeEnv
from policy_network import PolicyNetwork
from sum_utils import train_agent
from policies import epsilon_greedy_policy, pure_stochastic

# %% Deivce setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# %% Init Environment and policy

env = MazeEnv()

num_episodes = 1000
maze_size = (9, 9)

action_dim = env.action_space.n
maze_input_dim = np.prod(maze_size)
pos_input_dim = np.prod(maze_size)
policy_net = PolicyNetwork(maze_input_dim, pos_input_dim, action_dim).to(device)
policy = pure_stochastic
optimizer = optim.Adam(policy_net.parameters(), lr=1e-2)
# %% Training
train_agent(env, policy_net, policy, optimizer, num_episodes, maze_size, device)
# %%
