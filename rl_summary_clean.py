# %% Imports
from trainer import Trainer
from model import RlModel, PolicyNetwork
from policies import pure_stochastic
from maze_env import MazeEnv
import torch.optim as optim
import numpy as np

# %% Env init
env = MazeEnv()
# %% Env introspection
input_dim = env.observation_space.size
print(f"Input size flat: {input_dim}")
output_dim = env.action_space.n
print(f"Actionspace size flat: {output_dim}")
# env.action_space
# %% Main Loop
neural_net = PolicyNetwork(input_dim, output_dim)
optimizer = optim.Adam(neural_net.parameters(), lr=1e-2)
policy = pure_stochastic
model = RlModel(neural_net, optimizer)
trainer = Trainer(model, policy, env, 1000, 1, 0.99)
trainer.reinforce()
# %%
