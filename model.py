import torch
import torch.nn as nn
import torch.nn.functional as F
# import optimizer


class RlModel:
    def __init__(self, neural_net_architechture: nn.Module, optimizer):
        self.neuralnet = neural_net_architechture
        self.observationspace_dim = self.neuralnet.input_dim
        self.outputs = dict()
        self.curr_action_distribution = None
        self.current_value = None

        self.optimizer = optimizer

    def update(self, states, actions, returns):
        """
        Update the policy network using a loop over all collected states, actions, and returns.

        Parameters:
        - policy_net (torch.nn.Module): The policy network.
        - optimizer (torch.optim.Optimizer): Optimizer for the policy network.
        - states (torch.Tensor): Tensor of states from the episode(s).
        - actions (torch.Tensor): Tensor of actions taken in the episode(s).
        - returns (torch.Tensor): Tensor of discounted returns for each timestep.
        """
        self.optimizer.zero_grad()  # Reset gradients

        # Initialize the total loss
        total_loss = 0

        # Loop over all states, actions, and returns
        for state, action, Gt in zip(states, actions, returns):
            # Forward pass to get action probabilities
            action_probs = self.neuralnet(state)
            # print(f"actionprobs in update {action_probs}")
            log_probs = torch.log(action_probs)

            # Select the log probability for the taken action
            selected_log_prob = log_probs[action]

            # Calculate the loss (negative log probability weighted by the return)
            # Note: We negate the loss here because we're doing gradient ascent
            # print(f"Return is {Gt}")
            loss = -selected_log_prob * Gt

            # Accumulate the loss
            total_loss += loss

        # After accumulating losses, perform a backward pass and an optimization step
        # print(f"Total loss:{total_loss}")
        total_loss.backward()
        self.optimizer.step()
        # for param in self.neuralnet.parameters():
        #     print(param.grad)


class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        # Maze configuration input branch
        self.fc1_maze = nn.Linear(input_dim, 64)
        # Agent position input branch
        # self.fc1_pos = nn.Linear(, 32)
        # Combined layer
        self.fc2 = nn.Linear(64, 64)

        self.fc3 = nn.Linear(64, 64)
        # Output layer for actions
        self.action_head = nn.Linear(64, output_dim)
        # self.value_head = nn.Linear(64, 1)  # Output layer for state value estimate

    def forward(self, observationspace):
        # print(f"Observationspace in forward:\n{observationspace}")
        maze = F.relu(self.fc1_maze(observationspace))
        # pos = F.relu(self.fc1_pos(pos))
        # combined = torch.cat(
        #     (maze, pos), dim=1
        # )  # Combine the features from kkboth inputs
        x = F.relu(self.fc2(maze))
        x = F.relu(self.fc3(x))
        action_probs = F.softmax(
            self.action_head(x), dim=-1
        )  # Probability distribution over actions
        # state_values = self.value_head(x)  # Estimated value of the current state
        return action_probs


class PolicyValueNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyValueNetwork, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        # Maze configuration input branch
        self.fc1_maze = nn.Linear(input_dim, 64)
        # Agent position input branch
        # self.fc1_pos = nn.Linear(pos_input_dim, 32)
        # Combined layer
        self.fc2 = nn.Linear(64, 64)  # Combining maze and position inputs
        # Output layer for actions
        self.action_head = nn.Linear(64, output_dim)
        # Output layer for state value estimate
        self.value_head = nn.Linear(64, 1)

    def forward(self, maze):
        maze = F.relu(self.fc1_maze(maze))
        x = F.relu(self.fc2(maze))
        action_probs = F.softmax(
            self.action_head(x), dim=-1
        )  # Probability distribution over actions
        # Estimated value of the current state
        state_values = self.value_head(x)
        return action_probs, state_values


# class NeuralNetArchitecture(nn.Module):
#     def __init__(self):
#         super(NeuralNetArchitecture, self).__init__()
#
#     def forward(self, observationspace):
#         input = observationspace
#         for layer in self:
#             output = F.layer.activation(layer(input))
#             input = output
#         return output
