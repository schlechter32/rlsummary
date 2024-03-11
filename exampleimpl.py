import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")

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


def reinforce(
    policy_network, optimizer, env, episodes, gamma=0.99, early_stopping_rounds=10
):
    policy_network.to(device)  # Ensure the network is on the correct device
    no_improvement_count = 0
    last_total_reward = None

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
reinforce(policy_net, optimizer, env, episodes=1000)
