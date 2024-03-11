# Imports
import torch
import numpy as np


class Trainer:
    def __init__(self, model, policy, env, train_episodes, batch_size, gamma) -> None:
        self.model = model
        self.train_episodes = train_episodes
        self.batch_size = batch_size
        self.policy = policy
        self.env = env
        self.next_state = None
        self.nex_value = None
        self.rewards = []
        self.values = []
        self.masks = []
        self.total_reward = 0
        self.log_probs = []
        self.gamma = gamma
        self.device = "cuda"
        self.actions = []
        self.states = []
        self.no_episode = 0

    def a2c(self):
        # maze_input_dinv.action_space.n
        # Logging returns
        reward_logs = []
        episode_logs = []
        # policy_net = net

        # policy_net = PolicyNetwork(maze_input_dim, pos_input_dim, action_dim).to(device)
        # optimizer = optim.Adam(policy_net.parameters(), lr=1e-2)

        for episode in range(self.train_episodes):
            state = self.env.reset()
            done = False
            total_reward = 0

            log_probs = []
            values = []
            rewards = []
            masks = []
            # action_index_values = [(0, "down"), (1, "right"), (2, "up"), (3, "left")]
            # action_dict = dict(action_index_values)

            while not done:
                # maze_layout = self.env.maze.flatten()
                # maze_input = (
                #     torch.tensor(maze_layout, dtype=torch.float32)
                #     .unsqueeze(0)
                #     .to(device)
                # )
                # print("maze input is", maze_input)
                # print("state: ", state)

                # pos_input = encode_position(state, maze_size, device)
                # print("pos input is", pos_input)

                action_probs, value = self.model.neuralnet(
                    self.env.observation_space_tensor.to(self.device)
                )
                action = self.policy(
                    action_probs, max(0.05, 0.1 - 0.01 * (episode // 100))
                )
                # print("Action:", action_dict[action])

                log_prob = torch.log(action_probs.squeeze(0)[action])
                next_state, reward, done, _, _ = self.env.step(action)
                # print("done:", done)

                log_probs.append(log_prob)
                values.append(value)
                rewards.append(reward)
                masks.append(1.0 - done)

                state = next_state
                total_reward += reward

            next_value = (
                torch.tensor([0], device=self.device)
                if done
                else self.model.neuralnet(self.env.observation_space_tensor)[1]
            )
            returns = compute_returns_orig(next_value, rewards, masks, self.device)

            log_probs = torch.stack(log_probs).to(self.device)
            values = torch.cat(values).squeeze(-1).to(self.device)

            advantage = returns - values
            # print(f"advantage {values}")
            actor_loss = -(log_probs * advantage.detach()).mean()
            critic_loss = 0.5 * advantage.pow(2).mean()

            self.model.optimizer.zero_grad()
            total_loss = actor_loss + critic_loss
            total_loss.backward()
            self.model.optimizer.step()
            # print(
            #     f"Last agent pos at step {
            #       self.env.step_count} was {self.env.agent_pos}"
            # )

            print(f"Episode {episode + 1}: Total Reward: {total_reward}")
            reward_logs.append(total_reward)

    def reinforce(self):
        for episode in range(self.train_episodes):
            self.no_eisode = episode
            self.generate_episode()
            self.compute_returns()
            print(
                f"Episode reward for {episode} was {
                    self.total_reward}",
                end="\n",
            )
            # print(f"States buffer is {self.states}")
            # self.states = torch.tensor(self.states)
            # self.actions = torch.tensor(self.actions)
            # self.returns = torch.tensor(self.returns)
            self.model.update(
                self.states,
                self.actions,
                self.returns,
            )
            self.clear_episode_buffers()

    def compute_returns(self):
        """
        Compute the total discounted returns for each timestep in an episode.

        Parameters:
        - rewards (list of float): The rewards obtained from the episode.
        - gamma (float): The discount factor.

        Returns:
        - numpy.ndarray: The discounted returns for each timestep.
        """
        # Initialize an array to store the returns
        self.returns = np.zeros_like(self.rewards, dtype=float)
        # The return at the final timestep is simply the last reward
        G = 0

        # Iterate backwards through the rewards list to accumulate returns
        # print(f"Rewards in compute returns {self.rewards}")
        for t in reversed(range(len(self.rewards))):
            # Update the total return according to the discount factor
            G = self.rewards[t] + self.gamma * G
            self.returns[t] = G

        # return returns

    def clear_episode_buffers(self):
        self.rewards = []
        self.values = []
        self.masks = []
        self.log_probs = []
        self.total_reward = 0
        self.returns = []

        self.actions = []
        self.states = []

    def generate_episode(self):
        actions = []
        rewards = []
        # values = []
        # masks = []
        log_probs_buffer = []
        total_reward = 0
        done = False
        state = self.env.reset()
        rewards = []
        states = []
        next_states = []
        # states.append(state)
        while not done:
            action_probs = self.model.neuralnet(
                self.env.observation_space_tensor.to(self.device)
            )
            action = self.policy(action_probs, self.no_episode)
            log_probs = torch.log(action_probs)
            # print(f"State after append is {state}")
            next_state, reward, done, _, _ = self.env.step(action)
            states.append(next_state)
            next_states.append(next_state)
            state = next_state
            actions.append(action)
            log_probs_buffer.append(log_probs)
            rewards.append(reward)
            total_reward += reward

        self.actions = actions
        self.states = torch.stack(states).to(self.device)
        # print(f"Stacked states are {self.states}")
        self.log_probs.append(log_probs)
        self.rewards = rewards
        self.total_reward = total_reward

    #
    # def train(self):
    #     for episode in range(self.train_episodes):
    #         state, _ = self.env.reset()
    #         done = False
    #
    #         self.rewards = []
    #         self.values = []
    #         self.masks = []
    #         self.log_probs = []
    #         self.total_reward = 0
    #         while not done:
    #             flat_env = self.env.flatten()
    #             flat_env_tensor = (
    #                 torch.tensor(flat_env, dtype=torch.float32)
    #                 .unsqueeze(0)
    #                 .to(self.device)
    #             )
    #             # TODO implement in model
    #             # self.model.outputs = self.model(self.env.observation_spave)
    #             self.model.net(self.env.observation_space)
    #             action = self.policy(self.model.curr_action_distribution)
    #             log_prob = torch.log(self.model.curr_action_distribution)
    #             next_state, reward, done, _, _ = self.env.step(action)
    #             self.log_probs.append(log_prob)
    #             self.values.append(self.model.current_value)
    #             self.rewards.append(reward)
    #             self.masks.append(1.0 - done)
    #             # env.
    #             self.total_reward += reward
    #         # This needs to be improved model output can be quite diverse
    #         # ! Where to put this block is unclear
    #         self.model.net(self.env.observation_space)
    #         next_value = 0 if done else self.model.current_value
    #         returns = self.compute_returns(
    #             next_value, self.rewards, self.masks, self.device
    #         )
    #         advantage = returns - self.values
    #         actor_loss = -(self.log_probs * advantage.mean())
    #         critic_loss = 0.5 * advantage.pow(2).meand()
    #
    #         self.optimizer.zero_grad()
    #         total_loss = actor_loss + critic_loss
    #         total_loss.backward()
    #         self.optimizer.step()
    #         print(f"Episode {episode + 1}: Total Reward: {total_reward}")
    #         self.reward_logs.append(total_reward)

    def determine_device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # def compute_returns(self, next_value, rewards, masks, device, gamma=0.99):
    #     R = self.next_value
    #     returns = []
    #     for step in reversed(range(len(rewards))):
    #         R = rewards[step] + gamma * R * masks[step]
    #         returns.insert(0, R)
    #     return torch.tensor(returns, device=device).unsqueeze(1)


def compute_returns_orig(next_value, rewards, masks, device, gamma=0.99):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return torch.tensor(returns, device=device).unsqueeze(1)
