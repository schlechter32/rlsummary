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
        self.actions = []
        self.states = []

    def reinforce(self):
        for episode in range(self.train_episodes):
            self.generate_episode()
            self.compute_returns()
            print(f"Episode reward for {episode} was {self.total_reward}")
            self.model.update(self.states, self.actions, self.returns)
            self.clear_episode_buffers()

            # state, _ = self.env.reset()
            # done = False

            # while not done:
            #     flat_env = self.env.flatten()
            #     flat_env_tensor = (
            #         torch.tensor(flat_env, dtype=torch.float32)
            #         .unsqueeze(0)
            #         .to(self.device)
            #     )
            #     # TODO implement in model
            #     # self.model.outputs = self.model(self.env.observation_spave)
            #     self.model.net(self.env.observation_space)
            #     action = self.policy(self.model.curr_action_distribution)
            #     log_prob = torch.log(self.model.curr_action_distribution)
            #     next_state, reward, done, _, _ = self.env.step(action)
            #     self.log_probs.append(log_prob)
            #     # self.values.append(self.model.current_value)
            #     self.rewards.append(reward)
            #     self.masks.append(1.0 - done)
            #     # env.
            #     self.total_reward += reward
            # # This needs to be improved model output can be quite diverse
            # # self.model.net(self.env.observation_space)
            #
            # # Do the policy network update here
            #
            # next_value = 0 if done else self.model.current_value
            # returns = self.compute_returns(
            #     next_value, self.rewards, self.masks, self.device
            # )
            # advantage = returns - self.values
            # actor_loss = -(self.log_probs * advantage.mean())
            # critic_loss = 0.5 * advantage.pow(2).meand()
            #
            # self.optimizer.zero_grad()
            # total_loss = actor_loss + critic_loss
            # total_loss.backward()
            # self.optimizer.step()
            # print(f"Episode {episode + 1}: Total Reward: {total_reward}")
            # self.reward_logs.append(total_reward)

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
        log_probs = []
        total_reward = 0
        done = False
        state = self.env.reset()
        rewards = []
        states = []
        # states.append(state)
        while not done:
            action_probs = self.model.neuralnet(self.env.observation_space_tensor)
            action = self.policy(action_probs)
            log_prob = torch.log(action_probs)
            states.append(state)
            next_state, reward, done, _, _ = self.env.step(action)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            total_reward += reward

        self.actions = actions
        self.states = states
        self.log_probs.append(log_probs)
        self.rewards = rewards
        self.total_reward = total_reward

    def train(self):
        for episode in range(self.train_episodes):
            state, _ = self.env.reset()
            done = False

            self.rewards = []
            self.values = []
            self.masks = []
            self.log_probs = []
            self.total_reward = 0
            while not done:
                flat_env = self.env.flatten()
                flat_env_tensor = (
                    torch.tensor(flat_env, dtype=torch.float32)
                    .unsqueeze(0)
                    .to(self.device)
                )
                # TODO implement in model
                # self.model.outputs = self.model(self.env.observation_spave)
                self.model.net(self.env.observation_space)
                action = self.policy(self.model.curr_action_distribution)
                log_prob = torch.log(self.model.curr_action_distribution)
                next_state, reward, done, _, _ = self.env.step(action)
                self.log_probs.append(log_prob)
                self.values.append(self.model.current_value)
                self.rewards.append(reward)
                self.masks.append(1.0 - done)
                # env.
                self.total_reward += reward
            # This needs to be improved model output can be quite diverse
            # ! Where to put this block is unclear
            self.model.net(self.env.observation_space)
            next_value = 0 if done else self.model.current_value
            returns = self.compute_returns(
                next_value, self.rewards, self.masks, self.device
            )
            advantage = returns - self.values
            actor_loss = -(self.log_probs * advantage.mean())
            critic_loss = 0.5 * advantage.pow(2).meand()

            self.optimizer.zero_grad()
            total_loss = actor_loss + critic_loss
            total_loss.backward()
            self.optimizer.step()
            print(f"Episode {episode + 1}: Total Reward: {total_reward}")
            self.reward_logs.append(total_reward)

    def determine_device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # def compute_returns(self, next_value, rewards, masks, device, gamma=0.99):
    #     R = self.next_value
    #     returns = []
    #     for step in reversed(range(len(rewards))):
    #         R = rewards[step] + gamma * R * masks[step]
    #         returns.insert(0, R)
    #     return torch.tensor(returns, device=device).unsqueeze(1)
