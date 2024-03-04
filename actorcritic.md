# Machine learning

## DPR

## Bayes Model

## Random processes

### Distributions

# Deep Neural Networks

## Layers

### Deep

### Convolution

### Recurrent

### LSTM

### Graph

# Reinforcement Learning

## Bellmann

## Bellman Equation for Value Functions

The **Bellman equation** for value functions captures the relationship between the value of a state and the values of subsequent states in a reinforcement learning context. It provides a recursive decomposition of the value function, which is central to solving reinforcement learning problems.

Given a policy $\pi$, the value function $V^\pi(s)$ of a state $s$ under policy $\pi$ is defined as the expected return starting from state $s$, and then following policy $\pi$. The Bellman equation for the value function under policy $\pi$ is:

$$V^\pi(s) = \sum_{a \in A} \pi(a|s) \sum_{s', r} p(s', r | s, a) \left[ r + \gamma V^\pi(s') \right]$$

Where:

- $V^\pi(s)$ is the value of state $s$ under policy $\pi$,
- $\pi(a|s)$ denotes the probability of taking action $a$ in state $s$ under policy $\pi$,
- $p(s', r | s, a)$ is the transition probability to state $s'$ with reward $r$ after taking action $a$ in state $s$,
- $\gamma$ is the discount factor, which quantifies the difference in importance between future rewards and present rewards.

## Bellman Optimality Equation

The **Bellman Optimality Equation** pertains to the optimal policy, defining the best possible value function, known as the optimal value function. It describes the value of the best action to take in a given state.

The Bellman Optimality Equation for the optimal value function $V^*(s)$ is:

$$V^*(s) = \max_{a \in A} \sum_{s', r} p(s', r | s, a) \left[ r + \gamma V^*(s') \right]$$

For the action-value function $Q^*(s, a)$, which represents the value of taking action $a$ in state $s$ and then following the optimal policy, the equation is:

$$Q^*(s, a) = \sum_{s', r} p(s', r | s, a) \left[ r + \gamma \max_{a'} Q^*(s', a') \right]$$

Where:

- $V^*(s)$ and $Q^*(s, a)$ are the optimal value and action-value functions, respectively,
- The $\max$ operator is used to select the action that maximizes the expected return.

## Summary

The Bellman equations offer a recursive solution to determining value functions in reinforcement learning. They form the theoretical foundation for dynamic programming techniques such as Value Iteration and Policy Iteration, which seek to find optimal policies by improving value function estimates iteratively.

## Methodologies

## Training Methods

## Value Estimation and Policy Estimation

In reinforcement learning (RL), learning optimal behaviors involves two fundamental concepts: **value estimation** and **policy estimation**. They are crucial for navigating the environment but approach the problem differently.

### Value Estimation

Value estimation determines the worth of states (or state-action pairs). The "value" indicates the expected cumulative rewards from that state (or after taking an action) following a policy.

- **State Value Function (V)**: $V^\pi(s)$ is the expected return from state $s$ following policy π, showing how good it is to be in that state.

- **Action Value Function (Q)**: $Q^\pi(s, a)$ is the expected return from taking action $a$ in state $s$ following policy π, indicating the worth of an action in a state.

Value estimation is key in methods like Dynamic Programming and Temporal Difference Learning, focusing on improving value function estimations.

### Policy Estimation

Policy estimation directly determines the best action in a given state. It seeks to learn or improve the policy based on received rewards to maximize cumulative rewards.

- **Deterministic Policies**: Specify a single action for each state. Policy $\pi(s)$ maps to action $a$.

- **Stochastic Policies**: Provide action probabilities in a state. Policy $\pi(a|s)$ gives the probability of taking action $a$ in state $s$.

Policy estimation methods, like Policy Gradient and Actor-Critic, focus on learning or improving the policy directly.

### Key Differences

- **Objective Focus**: Value estimation evaluates the expected returns of states or actions. Policy estimation identifies the best action in each state.

- **Approach**: Value estimation methods evaluate without specifying action selection, while policy estimation explicitly defines action selection strategies.

- **Dependency**: Policy estimation can benefit from value estimation but can also be independent.

- **Applications**: Value estimation is foundational for many RL algorithms, understanding the environment's dynamics. Policy estimation determines the optimal interaction strategy.

Both value and policy estimation are crucial for effective reinforcement learning algorithms, encompassing action evaluation and decision-making aspects.

# Policies

## On-Policy vs. Off-Policy Learning

In reinforcement learning (RL), on-policy and off-policy learning distinguish how the learning algorithm utilizes data generated by the agent's policy.

### On-Policy Learning

On-policy methods learn the value of the policy being used for decisions. The exploration and the evaluated policy are the same, with the agent learning from its own actions.

- **Example**: SARSA is an on-policy algorithm that updates the action-value function based on the current policy's actions.

### Off-Policy Learning

Off-policy methods learn the optimal policy's value independently of the agent's actions, allowing learning from observed or hypothetical actions not taken by the behavior policy.

- **Example**: Q-Learning is an off-policy method that updates the action-value function using the maximum reward from the next state, regardless of the behavior policy's action.

### Key Differences

- **Policy Alignment**: On-policy aligns exploration with the target policy. Off-policy separates the exploration and target policies, learning an optimal policy from any actions.

- **Data Utilization**: On-policy methods explore and learn with the same policy, possibly leading to slower convergence. Off-policy methods can learn from any policy's data, offering more data efficiency.

- **Flexibility and Complexity**: Off-policy methods are more flexible but complex, learning from any data source. On-policy methods are straightforward but may require efficient exploration strategies.

- **Application Scenarios**: On-policy is suited for environments where exploration with the improving policy is feasible. Off-policy is beneficial where exploration is costly or when learning from a dataset of experiences generated by different policies.

Understanding on-policy versus off-policy learning helps in selecting the appropriate algorithm for reinforcement learning problems, considering exploration strategies and learning objectives.
## Deterministic and Stochastic Policies in Reinforcement Learning

In reinforcement learning (RL), policies guide agent decisions, crucial for maximizing cumulative rewards. Policies are either deterministic or stochastic, each with unique applications.

### Deterministic Policies

Deterministic policies prescribe a single action for each state, offering clear, unambiguous direction. Represented as \( \pi: S \rightarrow A \), for state \( s \), \( \pi(s) \) outputs action \( a \).

- **Example Usage**: Used in predictable environments where the best action per state is clearly defined.

### Stochastic Policies

Stochastic policies assign probabilities over actions for each state, allowing for exploration and handling uncertainties. Represented as \( \pi(a|s) \), indicating the likelihood of action \( a \) in state \( s \).

- **Example Usage**: Useful in uncertain environments or for exploration, modeling randomness in action outcomes.

### Key Differences

- **Action Selection**: Deterministic policies select one action; stochastic policies provide action probabilities.
  
- **Exploration vs. Exploitation**: Deterministic policies exploit known strategies; stochastic policies explore various actions.
  
- **Adaptability**: Stochastic policies adapt better to changing or uncertain environments.
  
- **Complexity**: Stochastic policies are complex due to managing and adjusting probabilities.

### Applications

- **Deterministic Policies**: Suited for static, predictable environments.
  
- **Stochastic Policies**: Ideal for uncertain environments or when exploration improves learning.

The choice between deterministic and stochastic policies affects agent learning, exploration, and performance in various environments.
## Value Iteration Algorithm

Value iteration is a dynamic programming algorithm used in reinforcement learning to find the optimal policy by iteratively improving the value function of each state. It utilizes the Bellman optimality equation during the update process.

#### Steps of the Value Iteration Algorithm:

1. **Initialization**: Start with an arbitrary value function \(V_0\) for all states, usually initialized to zero.

2. **Iteration**:
   For each state \(s\) in the state space \(S\), update the value function using the Bellman optimality equation:
   \[
   V_{k+1}(s) = \max_{a \in A} \sum_{s', r} P(s', r | s, a) [r + \gamma V_k(s')]
   \]
   - \(V_{k+1}(s)\) is the updated value at iteration \(k+1\).
   - \(A\) is the set of actions.
   - \(P(s', r | s, a)\) is the transition probability.
   - \(\gamma\) is the discount factor.
   - \(V_k(s')\) is the value at iteration \(k\).

3. **Convergence Check**: Repeat until the value function changes less than a small threshold \(\theta\) for all states.

4. **Policy Extraction**: Derive the optimal policy \(\pi^*\) by choosing the action that maximizes the expected utility for each state, based on the converged value function:
   \[
   \pi^*(s) = \arg\max_{a \in A} \sum_{s', r} P(s', r | s, a) [r + \gamma V(s')]
   \]
```python

def value_iteration(states, actions, transition_probabilities, rewards, gamma, theta):
    V = {s: 0 for s in states}  # Initialize value function
    policy = {s: None for s in states}  # Initialize policy
    
    while True:
        delta = 0
        for s in states:
            v = V[s]
            V[s] = max([sum([transition_probabilities[s][a][s_prime] * (rewards[s][a][s_prime] + gamma * V[s_prime]) for s_prime in states]) for a in actions])
            delta = max(delta, abs(v - V[s]))
        if delta < theta:  # Convergence check
            break
    
    for s in states:
        policy[s] = max(actions, key=lambda a: sum([transition_probabilities[s][a][s_prime] * (rewards[s][a][s_prime] + gamma * V[s_prime]) for s_prime in states]))
    
    return policy
```
### Flowchart Description:

- **Start** -> **Initialize \(V(s)\)**
- **For each state \(s\) in \(S\)**:
  - **Update \(V(s)\) using Bellman optimality**
- **Convergence Check**: If changes in \(V(s)\) are small, proceed to **Policy Extraction**; else, repeat.
- **Policy Extraction**: Compute \(\pi^*(s)\) for all \(s\).
- **End**

## Q-Learning Algorithm

Q-learning is a model-free reinforcement learning algorithm that informs an agent on optimal actions by learning the value of actions directly.

### Steps of the Q-Learning Algorithm:

1. **Initialization**: Initialize Q-values \(Q(s, a)\) for all state-action pairs.

2. **Episode Iteration**: For each episode:

   - **State Initialization**: Start from an initial state \(s\).
   
   - **Action Selection**: Choose action \(a\) based on a policy derived from Q-values (e.g., epsilon-greedy).

   - **Environment Interaction**: Take action \(a\), observe reward \(r\), and next state \(s'\).

   - **Q-value Update**: Update Q-value for \((s, a)\) using:
     \[
     Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma \max_{a'} Q(s', a') - Q(s, a)]
     \]
     where \(\alpha\) is the learning rate, and \(\gamma\) is the discount factor.

   - **State Update**: Set \(s = s'\).

3. **Repeat** until Q-values converge or satisfactory performance is reached.

4. **Policy Extraction**: Derive optimal policy \(\pi^*\) by selecting the action with the highest Q-value in each state:
   \[
   \pi^*(s) = \arg\max_{a} Q(s, a)
   \]

## Advantage Actor-Critic (A2C) Algorithm

Advantage Actor-Critic (A2C) combines value-based and policy-based methods, using two networks: the actor for action selection and the critic for value estimation.

### Steps of the A2C Algorithm:

1. **Initialization**: Initialize actor and critic networks with random weights.

2. **Rollout**:
   - Interact with the environment to generate states, actions, rewards, and next states using the current policy.
   - Calculate the advantage estimate (TD error) for each step:
     \[
     \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
     \]
   
3. **Update**:
   - **Critic Update**: Minimize the squared TD error using the critic network.
   - **Actor Update**: Maximize expected rewards adjusted by the advantage, updating the actor network with policy gradients:
     \[
     \nabla_{\theta} \log \pi_{\theta}(a_t|s_t) \delta_t
     \]
   
4. **Repeat**: Continue until policy convergence or a set number of iterations.

5. **Policy Extraction**: The final policy is given by the actor network's action proposals.

## Basics

### Advantage

loss or gain

### Actor

policy

### Critic

value or q value estimation

## DDPG (Deep Deterministic Policy Gradient)

## ACER

## TRPO

## PPO

# A3C Asynchronous Actor Critic
