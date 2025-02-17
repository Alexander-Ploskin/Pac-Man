# Controllers for Pac-Man Reinforcement Learning

This directory contains various controller implementations for the Pac-Man reinforcement learning environment. Each controller defines how Pac-Man makes decisions within the game.

## Available Controllers

### 1. Random Controller (`basic.py`)

#### Description

The `RandomController` is a simple controller that selects actions randomly from the available legal moves. It serves as a baseline for comparing more sophisticated control strategies and is useful for testing the environment.

#### Methods
-   `choose_action(state)`: Selects a random valid action from the current game state.
    -   **Input**:
        -   `state`: The current game state.
    -   **Output**: A randomly chosen valid action.

#### Usage

This controller is primarily used for testing the environment setup and providing a basic level of interaction. It does not involve any learning or intelligent decision-making.

### 2. Q-Learning Controller (`qlearn.py`)

#### Description

The `QLearningController` implements the Q-learning algorithm, a model-free reinforcement learning method, to train Pac-Man to navigate the environment effectively. It learns an optimal policy by estimating the Q-values for each state-action pair.

#### Core Concepts of Q-Learning

Q-learning aims to learn a Q-function, $Q(s, a)$, which represents the expected cumulative reward of taking action *a* in state *s* and following the optimal policy thereafter. The Q-function is updated iteratively using the Bellman equation:

$$
Q(s, a) = Q(s, a) + α * (R(s, a) + γ * \max_{a'}(Q(s', a')) - Q(s, a))
$$

Where:
-   $Q(s, a)$ is the current Q-value for state $s$ and action $a$.
-   $α$ is the learning rate.
-   $R(s, a)$ is the reward received after taking action $a$ in state $s$.
-   $γ$ is the discount factor.
-   $s'$ is the next state.

Q-learning also uses **exploration rate $\epsilon$**. In every action during training algorithm with probability $\epsilon$ sample random action. That gives algorithm an opportunity to explore the Map.

To make algorithm lay more on learned Q-value, exploration rate decreases with **gamma_eps** $\gamma_{\epsilon}$ (Factor to decay epsilon):

$$
\epsilon_{t+1} = \epsilon_{t} * \gamma_{\epsilon}
$$

### 3. Value Iteration Controller (`value_iteration.py`)

#### Description

The `ValueIterationAgent` implements the Value Iteration algorithm, a model-based dynamic programming approach to find the optimal policy. It computes the optimal value function for each state and uses it to determine the best actions.

#### Core Concepts of Value Iteration

Value Iteration computes the optimal value function V*(s) for each state by iteratively applying the Bellman optimality equation:

$$
V_{k+1}(s) = \max_a(R(s,a) + \gamma \cdot V_k(s'))
$$

Where:
- $V_k(s)$ is the value of state $s$ at iteration $k$
- $R(s,a)$ is the immediate reward for taking action $a$ in state $s$
- $\gamma$ is the discount factor
- $P(s'|s,a)$ is the transition probability (in our deterministic environment, this is always 1 for the resulting state)

The algorithm converges when the maximum change in values between iterations is less than a small threshold θ:

$$
\max_s |V_{k+1}(s) - V_k(s)| < \theta
$$

#### Parameters
Common parameters for Value Iteration:
- gamma = 0.98 (discount factor)
- theta = 1e-6 (convergence threshold)
- max_iterations = 1000

#### Limitations and Solutions

Due to the exponential growth of the state space with grid size (considering all possible pellet configurations), Value Iteration becomes computationally intractable for larger grids. For a 10×10 grid:
- Number of valid positions ≈ $ 61$
- Each valid position (except Pac-Man's) can have/not have a pellet
- Total states ≈ $61 * 2^{60}$ ≈ $10^{25}$

To address this limitation, two potential approaches are proposed:

1. **Stochastic State Sampling**:
   - Instead of considering all possible states, randomly sample a manageable subset
   - Train on this reduced state space to approximate the optimal policy
   - Suitable for getting approximate solutions on larger maps

2. **Hierarchical Reinforcement Learning**:
   - Train the agent on small subgrids (e.g., 4×4)
   - Use the learned policy recursively on larger maps
   - Decompose the large state space into manageable chunks

Due to these computational constraints, Value Iteration is currently implemented and tested only on 6×6 grids, and comparative experiments are not included in the results table.

### 4. Cross Entropy

#### Description

In Cross Entropy method we:

- Generate N sessions
- Select M elite sessions with the highest reward
- Train policy model on pairs (state, action) from these elite actions
- Repeat until convergence

### parameters for experiment
Common parameters:

- alpha = 0.3
- train_epsilon = 0.9
- test_epsilon = 0.2
- gamma = 0.98
- gamma_eps = 0.99997

#### For full-hash
- numTraining = 100000

#### For pacman position hash
- numTraining = 10000

## Experiments

Summary of results on `BasicPacmanEnvironment(grid_size=10, max_steps=200)` over 1000 games:

| | Random | Q-learning | Q-learning(full-hash) | Cross Entropy|
|:- | :----------- | :----------- | :--- | :--- |
| Mean score| 403.01     | 489.11      | 553.72 | 599.98
