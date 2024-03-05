import torch
import torch.nn as nn
import torch.nn.functional as F


# Define the policy network
class PolicyNetwork(nn.Module):
    def __init__(self, maze_input_dim, pos_input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        # Maze configuration input branch
        self.fc1_maze = nn.Linear(maze_input_dim, 32)
        # Agent position input branch
        self.fc1_pos = nn.Linear(pos_input_dim, 32)
        # Combined layer
        self.fc2 = nn.Linear(64, 64)  # Combining maze and position inputs
        self.action_head = nn.Linear(64, output_dim)  # Output layer for actions
        self.value_head = nn.Linear(64, 1)  # Output layer for state value estimate

    def forward(self, maze, pos):
        maze = F.relu(self.fc1_maze(maze))
        pos = F.relu(self.fc1_pos(pos))
        combined = torch.cat(
            (maze, pos), dim=1
        )  # Combine the features from both inputs
        x = F.relu(self.fc2(combined))
        action_probs = F.softmax(
            self.action_head(x), dim=-1
        )  # Probability distribution over actions
        state_values = self.value_head(x)  # Estimated value of the current state
        return action_probs, state_values


# Define the value network
class ValueNetwork(nn.Module):
    def __init__(self, input_dim):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)  # Hidden layer
        self.fc2 = nn.Linear(64, 1)  # Output layer

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def ppo_update(
    policy_net,
    value_net,
    optimizer_policy,
    optimizer_value,
    states,
    actions,
    log_probs_old,
    returns,
    advantages,
    epsilon=0.2,
    c1=0.5,
    c2=0.01,
):
    # Convert lists to tensors
    states = torch.stack(states)
    actions = torch.stack(actions)
    log_probs_old = torch.stack(log_probs_old)
    returns = torch.stack(returns).detach()
    advantages = torch.stack(advantages).detach()

    # Policy loss
    log_probs = torch.log(
        policy_net(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
    )
    ratio = torch.exp(log_probs - log_probs_old)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    # Value loss
    values = value_net(states).squeeze()
    value_loss = (returns - values).pow(2).mean()

    # Total loss
    loss = (
        policy_loss
        + c1 * value_loss
        - c2 * (log_probs * torch.log(log_probs + 1e-10)).mean()
    )

    # Update policy network
    optimizer_policy.zero_grad()
    policy_loss.backward()
    optimizer_policy.step()

    # Update value network
    optimizer_value.zero_grad()
    value_loss.backward()
    optimizer_value.step()
