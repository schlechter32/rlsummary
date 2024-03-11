
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")

# Environment Setup


class GridEnvironment:
    def __init__(self, size=9, start=(0, 0), goal=(8, 8)):
        self.size = size
        self.start = start
        self.goal = goal
        self.state = start

    def reset(self):
        self.state = self.start
        return self.state

    def step(self, action):
        y, x = self.state
        if action == 0:  # up
            y = max(y - 1, 0)
        elif action == 1:  # down
            y = min(y + 1, self.size - 1)
        elif action == 2:  # left
            x = max(x - 1, 0)
        elif action == 3:  # right
            x = min(x + 1, self.size - 1)

        self.state = (y, x)
        reward = -1
        done = self.state == self.goal
        return self.state, reward, done

    def state_to_tensor(self, state):
        y, x = state
        state_tensor = torch.zeros(self.size, self.size)
        state_tensor[y, x] = 1
        return state_tensor.view(-1).to(device)

# Policy Network (Actor)


class PolicyNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, output_size),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        return self.fc(x)

# Value Network (Critic)


class ValueNetwork(nn.Module):
    def __init__(self, input_size, output_size=1):
        super(ValueNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, output_size),
        )

    def forward(self, x):
        return self.fc(x)

# Common Functions


def visualize_path(env, path):
    RED = "\033[91m"
    RESET = "\033[0m"
    grid = [["." for _ in range(env.size)] for _ in range(env.size)]
    for y, x in getattr(env, "walls", []):
        grid[y][x] = "#"
    start_y, start_x = env.start
    goal_y, goal_x = env.goal
    grid[start_y][start_x] = "S"
    grid[goal_y][goal_x] = "G"
    for y, x in path:
        if (y, x) not in [env.start, env.goal]:
            grid[y][x] = f"{RED}.{RESET}"
    for row in grid:
        print(" ".join(row))


def simulate_episode_for_visualization(policy_network, env, max_steps=100):
    state = env.reset()
    done = False
    path = [state]
    steps = 0
    while not done and steps < max_steps:
        state_tensor = env.state_to_tensor(state).float()
        with torch.no_grad():
            action_probs = policy_network(state_tensor)
        action = torch.argmax(action_probs).item()
        state, _, done = env.step(action)
        if not done:
            path.append(state)
        steps += 1
    return path

# REINFORCE Method


def reinforce(policy_network, optimizer, env, episodes, gamma=0.99):
    # Implementation of REINFORCE
    ...

# A2C Method


def a2c(policy_network, value_network, optimizer_policy, optimizer_value, env, episodes, gamma=0.99):
    # Implementation of A2C


def a2c(
    policy_network,
    value_network,
    optimizer_policy,
    optimizer_value,
    env,
    episodes,
    gamma=0.99,
):
    total_rewards = []
    policy_network.to(device)
    value_network.to(device)
    last_total_reward = None
    for episode in range(episodes):
        saved_log_probs = []
        saved_values = []
        rewards = []
        state = env.reset()
        done = False

        policy_network.train()
        value_network.train()

        while not done:
            state_tensor = env.state_to_tensor(state).float().to(device)
            action_probs = policy_network(state_tensor)
            value = value_network(state_tensor)

            distribution = Categorical(action_probs)
            action = distribution.sample()
            saved_log_probs.append(distribution.log_prob(action))
            saved_values.append(value)

            state, reward, done = env.step(action.item())
            rewards.append(reward)
        total_reward = sum(rewards)
        total_rewards.append(total_reward)

        if last_total_reward is not None and total_reward <= last_total_reward:
            no_improvement_count += 1
        else:
            no_improvement_count = 0
        last_total_reward = total_reward

        if no_improvement_count >= early_stopping_rounds:
            print(f"Early stopping triggered after {episode+1} episodes.")
            break
        # Compute returns and advantages
        returns = []
        advantages = []
        R = 0
        for i in reversed(range(len(rewards))):
            R = rewards[i] + gamma * R
            advantage = R - saved_values[i]
            returns.append(R)
            advantages.append(advantage)

        # Update policy network (actor)
        policy_loss = [
            -log_prob * advantage.detach()
            for log_prob, advantage in zip(saved_log_probs, advantages)
        ]
        policy_loss = torch.cat(policy_loss).sum()
        optimizer_policy.zero_grad()
        policy_loss.backward()
        optimizer_policy.step()

        # Update value network (critic)
        value_loss = [advantage.pow(2) for advantage in advantages]
        value_loss = torch.cat(value_loss).sum()
        optimizer_value.zero_grad()
        value_loss.backward()
        optimizer_value.step()

        if episode % 10 == 0:
            visualize_path(
                env, simulate_episode_for_visualization(policy_network, env))
            print(f"Episode {episode}: Total Reward = {sum(rewards)}")
    return total_rewards


def a2c(
    policy_network,
    value_network,
    optimizer_policy,
    optimizer_value,
    env,
    episodes,
    gamma=0.99,
    entropy_beta=0.01,
):
    policy_network.to(device)
    value_network.to(device)
    rewards_log = []

    for episode in range(episodes):
        saved_log_probs = []
        saved_values = []
        rewards = []
        state = env.reset()
        done = False

        while not done:
            state_tensor = env.state_to_tensor(state).float().to(device)
            action_probs = policy_network(state_tensor)
            value = value_network(state_tensor).squeeze()

            distribution = Categorical(action_probs)
            action = distribution.sample()
            log_prob = distribution.log_prob(action)
            saved_log_probs.append(log_prob)
            saved_values.append(value)

            state, reward, done = env.step(action.item())
            rewards.append(reward)

        # Convert lists to tensors
        saved_log_probs = torch.stack(saved_log_probs)
        saved_values = torch.stack(saved_values)
        returns = compute_returns(rewards, gamma, device)

        advantages = returns - saved_values
        advantages = (advantages - advantages.mean()) / \
            (advantages.std() + 1e-8)

        policy_loss = -(saved_log_probs * advantages.detach()).mean()
        value_loss = 0.5 * advantages.pow(2).mean()

        # Entropy regularization
        entropy = torch.sum(
            action_probs * torch.log(action_probs + 1e-8), dim=1)
        policy_loss -= entropy_beta * entropy.mean()

        # Update networks
        optimizer_policy.zero_grad()
        policy_loss.backward()
        optimizer_policy.step()

        optimizer_value.zero_grad()
        value_loss.backward()
        optimizer_value.step()

        # Optional: Adjust learning rate via scheduler
        # scheduler_policy.step()
        # scheduler_value.step()

        rewards_log.append(sum(rewards))

        if episode % 10 == 0:
            visualize_path(
                env, simulate_episode_for_visualization(policy_network, env))
            print(f"Episode {episode}: Total Reward = {sum(rewards)}")

            print(f"Episode {episode}: Total Reward = {sum(rewards)}")

    return rewards_log


# Initialization and Training
input_size = 9 * 9
output_size = 4  # Up, Down, Left, Right
policy_net = PolicyNetwork(input_size, output_size).to(device)
value_net = ValueNetwork(input_size).to(device)
optimizer_policy = optim.Adam(policy_net.parameters(), lr=0.01)
optimizer_value = optim.Adam(value_net.parameters(), lr=0.01)

env = GridEnvironment(size=9)  # Initialize your environment

# Uncomment the method you wish to use for training
# reinforce(policy_net, optimizer_policy, env, episodes=1000)
# a2c(policy_net, value_net, optimizer_policy, optimizer_value, env, episodes=1000)
