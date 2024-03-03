import torch
import torch.optim as optim
import numpy as np
from maze_env import MazeEnv  # Make sure your MazeEnv is properly defined
from policy_network import PolicyNetwork  # Adjust PolicyNetwork as previously discussed


def encode_position(position, maze_size):
    one_hot = np.zeros(np.prod(maze_size))
    index = position[0] * maze_size[1] + position[1]
    one_hot[index] = 1
    return one_hot


def epsilon_greedy_policy(action_probs, epsilon=0.1):
    if np.random.rand() < epsilon:
        return np.random.randint(action_probs.size(1))
    else:
        return torch.argmax(action_probs).item()


def compute_returns(next_value, rewards, masks, gamma=0.99):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return tuple([torch.tensor(returns)])


def main():
    env = MazeEnv()
    num_episodes = 1000
    gamma = 0.99

    maze_size = (9, 9)
    maze_input_dim = np.prod(maze_size)
    pos_input_dim = np.prod(maze_size)
    action_dim = env.action_space.n

    policy_net = PolicyNetwork(maze_input_dim, pos_input_dim, action_dim)
    optimizer = optim.Adam(policy_net.parameters(), lr=1e-2)

    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0

        log_probs = []
        values = []
        rewards = []
        masks = []

        while not done:
            print("Start of loop")
            maze_layout = env.maze.flatten()
            maze_input = torch.FloatTensor(maze_layout).unsqueeze(0)

            pos_input = encode_position(state, maze_size)
            pos_input = torch.FloatTensor(pos_input).unsqueeze(0)

            action_probs, value = policy_net(maze_input, pos_input)
            action = epsilon_greedy_policy(
                action_probs, epsilon=max(0.05, 0.1 - 0.01 * (episode // 100))
            )

            log_prob = torch.log(action_probs.squeeze(0)[action])

            # return np.array(self.agent_pos), reward, terminated, truncated, info
            next_state, reward, done, truncatec, info = env.step(action)

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)
            masks.append(1.0 - done)

            state = next_state
            total_reward += reward

        next_value = 0 if done else policy_net(maze_input, pos_input)[1]
        print("Going to compute returns")
        returns = compute_returns(next_value, rewards, masks, gamma)

        log_probs = torch.stack(log_probs)
        returns = torch.stack(returns).detach()
        values = torch.cat(values)

        advantage = returns - values
        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = 0.5 * advantage.pow(2).mean()

        optimizer.zero_grad()
        (actor_loss + critic_loss).backward()
        print("Doing optimizer step")
        optimizer.step()

        print(f"Episode {episode + 1}: Total Reward: {total_reward}")


if __name__ == "__main__":
    main()
