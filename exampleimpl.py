import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical

# import matplotlib.pyplot as plt

import plotext as plt


import plotext as plt


def plot_rewards_terminal(rewards):
    plt.clc()  # Clear previous plots to ensure a fresh start (use clp() instead of clear_plot())

    # Plot the rewards
    plt.plot(rewards, color="cyan", marker="dot")

    # Customize the plot
    plt.title("Training Rewards Over Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")

    # Attempting to set grid lines (if available in your version of plotext)
    # Else, you might need to remove this if it causes errors
    try:
        plt.grid(True, axis="both", color="green", linestyle="--", linewidth=0.5)
    except TypeError:
        plt.grid(True)  # Use without the unsupported keyword arguments

    # Unfortunately, plotext does not support setting the number of ticks directly as matplotlib does
    # But you can set the limits and labels manually if needed

    # Show plot
    plt.show()


# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")


def visualize_rewards(rewards):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label="Rewards per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Rewards Over Episodes")
    plt.legend()
    plt.show()


# Environment setup


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
        """Converts state into a tensor for the neural network."""
        y, x = state
        state_tensor = torch.zeros(self.size, self.size)
        state_tensor[y, x] = 1
        # Ensure tensor is sent to the device
        return state_tensor.view(-1).to(device)


class GridEnvironmentWithWalls(GridEnvironment):
    def __init__(self, size=9, start=(0, 0), goal=(8, 8), walls=None):
        super().__init__(size, start, goal)
        if walls is None:
            walls = []
        self.walls = walls

    def step(self, action):
        y, x = self.state
        new_y, new_x = y, x

        if action == 0:  # up
            new_y = max(y - 1, 0)
        elif action == 1:  # down
            new_y = min(y + 1, self.size - 1)
        elif action == 2:  # left
            new_x = max(x - 1, 0)
        elif action == 3:  # right
            new_x = min(x + 1, self.size - 1)

        if (new_y, new_x) not in self.walls:
            self.state = (new_y, new_x)
        reward = -1
        done = self.state == self.goal
        return self.state, reward, done


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


def visualize_path_no(env, path):
    grid = np.zeros((env.size, env.size), dtype=int)
    # Use getattr to handle environments without walls
    for y, x in getattr(env, "walls", []):
        grid[y, x] = 1  # Mark walls with 1
    for y, x in path:
        if grid[y, x] != 1:  # Ensure not overwriting walls
            grid[y, x] = 2  # Mark path with 2
    grid[env.goal] = 3  # Mark goal with 3
    print(grid)


def visualize_path(env, path):
    # Define ANSI escape code for red color and reset
    RED = "\033[91m"
    RESET = "\033[0m"

    grid = [["." for _ in range(env.size)] for _ in range(env.size)]
    # Use getattr to handle environments without walls
    for y, x in getattr(env, "walls", []):
        grid[y][x] = "#"

    # Mark the start and goal
    start_y, start_x = env.start
    goal_y, goal_x = env.goal
    grid[start_y][start_x] = "S"
    grid[goal_y][goal_x] = "G"

    for y, x in path:
        if (y, x) not in [env.start, env.goal]:  # Don't overwrite start or goal
            grid[y][x] = f"{RED}.{RESET}"  # Mark path with red dots

    # Print the grid
    for row in grid:
        print(" ".join(row))


def simulate_episode_for_visualization(policy_network, env, max_steps=100):
    state = env.reset()
    done = False
    path = [state]
    steps = 0
    policy_network.eval()
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


def compute_returns(rewards, gamma, device):
    R = 0
    returns = []
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return torch.tensor(returns, dtype=torch.float32, device=device)


def normalize(advantages):
    if advantages.std() > 0:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return advantages


def evaluate_actions(policy_network, value_network, states, actions):
    action_probs = policy_network(states)
    dist = Categorical(action_probs)

    # Evaluate actions
    log_probs = dist.log_prob(actions)
    entropy = dist.entropy()

    # Evaluate state values
    state_values = value_network(states)

    return log_probs, state_values, entropy


def collect_trajectories(env, policy_network, episodes_per_update):
    memories = []
    for _ in range(episodes_per_update):
        state = env.reset()
        done = False
        print("Started new run")
        while not done:
            state_tensor = env.state_to_tensor(state).float().to(device)
            with torch.no_grad():
                action_probs = policy_network(state_tensor)
            distribution = Categorical(action_probs)
            action = distribution.sample()

            next_state, reward, done = env.step(action.item())
            # print(f"Action is {action}")

            # Store the experience
            memories.append(
                {
                    "state": state_tensor,
                    "action": action,
                    "reward": reward,
                    "log_prob": distribution.log_prob(action),
                }
            )
            state = next_state

    return memories


def compute_advantages(memories, value_network, gamma, tau=0.95):
    running_add = 0
    advantages = []
    returns = []
    for i in reversed(range(len(memories))):
        if i == len(memories) - 1:
            next_value = 0
        else:
            next_value = value_network(memories[i + 1]["state"]).item()
        td_error = (
            memories[i]["reward"]
            + gamma * next_value
            - value_network(memories[i]["state"]).item()
        )
        running_add = td_error + gamma * tau * running_add
        advantages.insert(0, running_add)
        returns.insert(0, running_add + value_network(memories[i]["state"]).item())

    return torch.tensor(advantages, dtype=torch.float, device=device), torch.tensor(
        returns, dtype=torch.float, device=device
    )


def ppo(
    policy_network,
    value_network,
    optimizer_policy,
    optimizer_value,
    env,
    episodes,
    gamma=0.99,
    epsilon=0.1,
    ppo_steps=4,
    episodes_per_update=10,
    visualize_every=20,
    early_stopping_rounds=10,
):
    policy_network.to(device)
    value_network.to(device)
    rewards_log = []

    best_avg_reward = float("-inf")
    no_improvement_count = 0

    for update in range(episodes // episodes_per_update):
        print("Started making memories")
        memories = collect_trajectories(env, policy_network, episodes_per_update)
        print("Computing advantages from memories")
        advantages, returns = compute_advantages(memories, value_network, gamma)

        # Prepare data for PPO update
        # Assuming `memories` is a list of dictionaries containing 'state', 'action', 'log_prob', and 'reward'
        states = torch.stack([m["state"] for m in memories]).to(device)
        actions = torch.tensor([m["action"] for m in memories], dtype=torch.long).to(
            device
        )
        old_log_probs = torch.tensor(
            [m["log_prob"] for m in memories], dtype=torch.float
        ).to(device)

        print("Updating Actor in Critic")
        for _ in range(ppo_steps):
            # Calculate current log probabilities and values
            log_probs, state_values, entropy = evaluate_actions(
                policy_network, value_network, states, actions
            )
            ratios = torch.exp(log_probs - old_log_probs.detach())

            # Clipped surrogate function
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - epsilon, 1 + epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy.mean()
            value_loss = F.mse_loss(state_values.squeeze(-1), returns)

            optimizer_policy.zero_grad()
            optimizer_value.zero_grad()
            (policy_loss + value_loss).backward()
            optimizer_policy.step()
            optimizer_value.step()

        # Calculate average reward for this update's episodes
        avg_reward = np.sum([m["reward"] for m in memories])
        rewards_log.append(avg_reward)

        # Early stopping
        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        if no_improvement_count >= early_stopping_rounds:
            print(
                f"Early stopping triggered after {
                    update + 1} updates with no improvement."
            )
            break

        if (update + 1) % (visualize_every // episodes_per_update) == 0:
            print(f"Update {update + 1}: Average Total Reward = {avg_reward}")
            path = simulate_episode_for_visualization(policy_network, env)
            visualize_path(env, path)

    return rewards_log


def a2c(
    policy_network,
    value_network,
    optimizer_policy,
    optimizer_value,
    scheduler_policy,
    scheduler_value,
    env,
    episodes,
    gamma=0.99,
    entropy_beta=0.01,
    early_stopping_rounds=50,
):
    policy_network.to(device)
    value_network.to(device)
    rewards_log = []

    best_reward = float("-inf")
    episodes_since_improvement = 0

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

        total_reward = sum(rewards)
        rewards_log.append(total_reward)

        if total_reward > best_reward:
            best_reward = total_reward
            episodes_since_improvement = 0
        else:
            episodes_since_improvement += 1

        if episodes_since_improvement >= early_stopping_rounds:
            print(f"Early stopping triggered after {episode + 1} episodes.")
            break

        # Compute returns and advantages
        returns = compute_returns(rewards, gamma, device)

        saved_log_probs = torch.stack(saved_log_probs)
        saved_values = torch.stack(saved_values).squeeze(-1)
        advantages = returns - saved_values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Compute losses
        policy_loss = -(saved_log_probs * advantages.detach()).mean()
        value_loss = F.mse_loss(saved_values, returns)

        # Entropy regularization
        entropy_loss = entropy_beta * distribution.entropy().mean()

        # Total loss
        total_loss = policy_loss + value_loss - entropy_loss

        # Perform backpropagation
        optimizer_policy.zero_grad()
        optimizer_value.zero_grad()
        total_loss.backward()
        optimizer_policy.step()
        optimizer_value.step()

        # Update the learning rate
        scheduler_policy.step()
        scheduler_value.step()

        if episode % 10 == 0:
            visualize_path(env, simulate_episode_for_visualization(policy_network, env))
            print(f"Episode {episode}: Total Reward = {sum(rewards)}")

    return rewards_log


# Compute returns helper function


def compute_returns(rewards, gamma, device):
    R = 0
    returns = []
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return torch.tensor(returns, dtype=torch.float32, device=device)


def reinforce(
    policy_network, optimizer, env, episodes, gamma=0.99, early_stopping_rounds=10
):
    policy_network.to(device)  # Ensure the network is on the correct device
    no_improvement_count = 0
    last_total_reward = None
    total_rewards = []

    for episode in range(episodes):
        policy_network.train()
        saved_log_probs = []
        rewards = []
        state = env.reset()
        done = False

        while not done:
            state_tensor = env.state_to_tensor(state).float()
            action_probs = policy_network(state_tensor)
            distribution = Categorical(action_probs)
            action = distribution.sample()
            saved_log_probs.append(distribution.log_prob(action))

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

        returns = []
        R = 0
        for r in rewards[::-1]:
            R = r + gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns).to(device)
        epsilon = 1e-9  # To avoid division by zero in normalization
        returns = (returns - returns.mean()) / (returns.std() + epsilon)

        policy_loss = sum(
            [-log_prob * R for log_prob, R in zip(saved_log_probs, returns)]
        )
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        if episode % 10 == 0:
            visualize_path(env, simulate_episode_for_visualization(policy_network, env))
            print(f"Episode {episode}: Total Reward = {sum(rewards)}")
    return total_rewards


# Policy network and optimizer setup
input_size = 9 * 9
output_size = 4  # 4 actions
policy_net = PolicyNetwork(input_size, output_size).to(device)
optimizer = optim.Adam(policy_net.parameters(), lr=0.01)

# Choose the environment
# No walls env
# env = GridEnvironment(size=9)  # For environment without walls
# walls env
easy_walls = [(1, i) for i in range(1, 8)] + [(7, i) for i in range(1, 8)]
# Define a more complicated set of walls for the maze
complicated_walls = [
    (1, 2),
    (1, 3),
    (1, 4),
    (1, 5),
    (1, 6),
    (2, 6),
    (3, 0),
    (3, 1),
    (3, 2),
    (3, 4),
    (3, 6),
    (4, 4),
    (5, 2),
    (5, 4),
    (5, 6),
    (5, 7),
    (5, 8),
    (6, 2),
    (7, 4),
    (7, 5),
    (7, 6),
]
# For environment with walls
env = GridEnvironmentWithWalls(size=9, walls=complicated_walls)

# Training with visualization

value_net = ValueNetwork(input_size).to(device)
optimizer_policy = optim.Adam(policy_net.parameters(), lr=0.01)
optimizer_value = optim.Adam(value_net.parameters(), lr=0.01)

# Initialize learning rate schedulers
scheduler_policy = torch.optim.lr_scheduler.StepLR(
    optimizer_policy, step_size=100, gamma=0.99
)
scheduler_value = torch.optim.lr_scheduler.StepLR(
    optimizer_value, step_size=100, gamma=0.99
)
# Training with reinforce
# total_rewards = reinforce(policy_net, optimizer, env, episodes=1000)

# Training with A2C
# total_rewards = a2c(
#     policy_net,
#     value_net,
#     optimizer_policy,
#     optimizer_value,
#     scheduler_policy,
#     scheduler_value,
#     env,
#     episodes=1000,
# )
# Training with ppo

total_rewards = ppo(policy_net, value_net, optimizer_policy, optimizer_value, env, 1000)
# ppo(policy_net, value_net, optimizer, env, 1000)

# Plotting of rewards
plot_rewards_terminal(total_rewards)
