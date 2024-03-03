import torch
import torch.optim as optim
import numpy as np
from policy_network import PolicyNetwork
from policies import epsilon_greedy_policy


def encode_position(position, maze_size, device):
    one_hot = np.zeros(np.prod(maze_size))
    index = position[0] * maze_size[1] + position[1]
    one_hot[index] = 1
    return torch.tensor(one_hot, device=device, dtype=torch.float32).unsqueeze(0)


def compute_returns(next_value, rewards, masks, device, gamma=0.99):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return torch.tensor(returns, device=device).unsqueeze(1)


def train_agent(env, net, policy, optimizer, num_episodes, maze_size, device):
    # maze_input_dim = np.prod(maze_size)
    # pos_input_dim = np.prod(maze_size)
    # action_dim = env.action_space.n
    policy_net = net

    # policy_net = PolicyNetwork(maze_input_dim, pos_input_dim, action_dim).to(device)
    # optimizer = optim.Adam(policy_net.parameters(), lr=1e-2)

    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0

        log_probs = []
        values = []
        rewards = []
        masks = []
        # action_index_values = [(0, "down"), (1, "right"), (2, "up"), (3, "left")]
        # action_dict = dict(action_index_values)

        while not done:
            maze_layout = env.maze.flatten()
            maze_input = (
                torch.tensor(maze_layout, dtype=torch.float32).unsqueeze(0).to(device)
            )
            # print("maze input is", maze_input)
            # print("state: ", state)

            pos_input = encode_position(state, maze_size, device)
            # print("pos input is", pos_input)

            action_probs, value = policy_net(maze_input, pos_input)
            action = policy(action_probs, max(0.05, 0.1 - 0.01 * (episode // 100)))
            # print("Action:", action_dict[action])

            log_prob = torch.log(action_probs.squeeze(0)[action])
            next_state, reward, done, _, _ = env.step(action)
            # print("done:", done)

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)
            masks.append(1.0 - done)

            state = next_state
            total_reward += reward

        next_value = (
            torch.tensor([0], device=device)
            if done
            else policy_net(maze_input, pos_input)[1]
        )
        returns = compute_returns(next_value, rewards, masks, device)

        log_probs = torch.stack(log_probs)
        values = torch.cat(values).squeeze(-1)

        advantage = returns - values
        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = 0.5 * advantage.pow(2).mean()

        optimizer.zero_grad()
        total_loss = actor_loss + critic_loss
        total_loss.backward()
        optimizer.step()

        print(f"Episode {episode + 1}: Total Reward: {total_reward}")


# def old_main():
#     env = MazeEnv()
#     # .to(device)
#     num_episodes = 1000
#     maze_size = (9, 9)
#
#     maze_input_dim = np.prod(maze_size)
#     pos_input_dim = np.prod(maze_size)
#     action_dim = env.action_space.n
#
#     policy_net = PolicyNetwork(maze_input_dim, pos_input_dim, action_dim).to(device)
#     optimizer = optim.Adam(policy_net.parameters(), lr=1e-2)
#
#     for episode in range(num_episodes):
#         state, _ = env.reset()
#         done = False
#         total_reward = 0
#
#         log_probs = []
#         values = []
#         rewards = []
#         masks = []
#         action_index_values = [(0, "down"), (1, "right"), (2, "up"), (3, "left")]
#         action_dict = dict(action_index_values)
#
#         while not done:
#             maze_layout = env.maze.flatten()
#             maze_input = (
#                 torch.tensor(maze_layout, dtype=torch.float32).unsqueeze(0).to(device)
#             )
#             # print("maze input is", maze_input)
#             # print("state: ", state)
#
#             pos_input = encode_position(state, maze_size)
#             # print("pos input is", pos_input)
#
#             action_probs, value = policy_net(maze_input, pos_input)
#             action = epsilon_greedy_policy(
#                 action_probs, epsilon=max(0.05, 0.1 - 0.01 * (episode // 100))
#             )
#             # print("Action:", action_dict[action])
#
#             log_prob = torch.log(action_probs.squeeze(0)[action])
#             next_state, reward, done, _, _ = env.step(action)
#             # print("done:", done)
#
#             log_probs.append(log_prob)
#             values.append(value)
#             rewards.append(reward)
#             masks.append(1.0 - done)
#
#             state = next_state
#             total_reward += reward
#
#         next_value = (
#             torch.tensor([0], device=device)
#             if done
#             else policy_net(maze_input, pos_input)[1]
#         )
#         returns = compute_returns(next_value, rewards, masks)
#
#         log_probs = torch.stack(log_probs)
#         values = torch.cat(values).squeeze(-1)
#
#         advantage = returns - values
#         actor_loss = -(log_probs * advantage.detach()).mean()
#         critic_loss = 0.5 * advantage.pow(2).mean()
#
#         optimizer.zero_grad()
#         total_loss = actor_loss + critic_loss
#         total_loss.backward()
#         optimizer.step()
#
#         print(f"Episode {episode + 1}: Total Reward: {total_reward}")
