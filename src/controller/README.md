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

| | Random | Q-learning | Q-learning(full-hash) |
|:- | :----------- | :----------- | :--- |
| Mean score| 403.01     | 489.11      | 553.72 |
