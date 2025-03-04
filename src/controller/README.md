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


### 5. REINFORCE

#### Description

**REINFORCE** (Williams, 1992) is a classic **policy gradient** algorithm in Reinforcement Learning. Instead of learning a value function, it **directly** optimizes the parameters $\theta$ of a stochastic policy $\pi_\theta(a \mid s)$ to maximize expected returns.

**Key Steps**:

1. **Sample Trajectory**  
   Interact with the environment under the current policy to gather a sequence of experiences:

   $$\tau = \{ (s_0, a_0, R_0), \ldots, (s_T, a_T, R_T) \}.$$

2. **Compute Returns**  
   For each time step $t$ in the trajectory, determine the discounted total reward:

   $$G_t = \sum_{k=t}^{T} \gamma^{(k - t)} \ R_k,$$

   where $\gamma \in [0,1]$ is a discount factor.

3. **Gradient Update**  
   After collecting a full trajectory, update parameters $\theta$ through **gradient ascent**:

   $$\theta_{t+1} = \theta_t + \alpha \nabla_{\theta} \log \pi_{\theta}(a_t | s_t) G_t$$

   where:
   - $\alpha$ is the learning rate,
   - $\nabla_\theta \log \pi_\theta(a_t \mid s_t)$ measures how the probability of choosing $a_t$ in $s_t$ depends on $\theta$,
   - $G_t$ is the return following time step $t$.

By **increasing** the log-probability of action–state pairs that led to higher rewards, REINFORCE effectively shifts the policy towards more **rewarding** behaviors. Over repeated episodes, this procedure refines $\theta$ to enhance performance.

#### Neural Network Architecture

**NeuralNetworkPolicy** uses a **Convolutional Neural Network (CNN)** to process the **Pac-Man grid**:

1. **Input**  
   - A 2D grid (Pac-Man’s map) fed as a single-channel tensor of shape  
     $(1 \times \text{map-size} \times \text{map-size})$.

2. **Convolution + Pooling Layers**  
   - Three blocks of convolution (`Conv2d`) interspersed with pooling (`MaxPool2d`, where applicable), activation (`GELU`), and normalization (`BatchNorm2d`).
   - Extracts **spatial features** essential for Pac-Man gameplay (pellets, walls, ghosts).

3. **Flatten + Dense Layer**  
   - The final convolution output is **flattened** and passed to a `LazyLinear` layer that produces $ \text{num\_actions} $ **logits**.

4. **Action Selection (Softmax)**  
   - The logits are divided by a **temperature** $t$ before applying softmax (or log-softmax), yielding action probabilities for **Up, Right, Down, Left**.
   - Higher $t$ ⇒ more **exploration**; lower $t$ ⇒ more **greedy**.

#### Temperature $t$

During training, $t$ is **decreased** from larger values, which gradually transitions the policy from exploratory to more deterministic behavior.


## REINFORCE Algorithm

```pseudo
Algorithm: REINFORCE (Monte Carlo Policy Gradient)

Input:
    - Number of training iterations: N
    - Number of episodes per iteration: M
    - Discount factor: γ ∈ [0, 1]
    - Learning rate: α

Initialize policy parameters θ

for i in 1 to N do:
    Initialize total_loss ← 0
    Initialize total_return ← 0

    for episode in 1 to M do:
        # 1. Generate an episode (trajectory) using π_θ
        s_0 ← initial_state
        trajectory ← empty list

        while episode not finished do:
            a_t ~ π_θ(a_t | s_t)            # Sample action a_t from policy
            s_{t+1}, r_{t+1} ← step_env(a_t) # Take action, observe next state & reward
            trajectory.append((s_t, a_t, r_{t+1}))
            s_t ← s_{t+1}

        # 2. Compute returns for each step in trajectory
        G ← 0
        for t in reverse(trajectory indices):
            G ← r_{t+1} + γ * G
            # Store G_t in the trajectory for updating θ
            trajectory[t].return ← G

        # 3. Accumulate policy gradient loss
        episode_loss ← 0
        episode_return ← sum of all rewards in trajectory
        for each (s_t, a_t, G_t) in trajectory do:
            episode_loss ← episode_loss - (G_t * log π_θ(a_t | s_t))

        total_loss ← total_loss + episode_loss
        total_return ← total_return + episode_return

    # 4. Update θ via gradient descent on total_loss
    θ ← θ - α * ∇_θ [ total_loss / M ]
```


## Experiments

### REINFORCE

#### Reward on basic env
![reward basic](https://raw.githubusercontent.com/Alexander-Ploskin/Pac-Man/master/assets/reward_basic.jpg)

#### Score on basic env
![score basic](https://raw.githubusercontent.com/Alexander-Ploskin/Pac-Man/master/assets/score_basic.jpg)

#### Reward on ghosts env
![reward ghosts](https://raw.githubusercontent.com/Alexander-Ploskin/Pac-Man/master/assets/reward_ghosts.jpg)

#### Score on ghosts env
![score ghosts](https://raw.githubusercontent.com/Alexander-Ploskin/Pac-Man/master/assets/score_ghosts.jpg)

Summary of results on `BasicPacmanEnvironment(grid_size=10, max_steps=200)` over 1000 games:

| | Random | Q-learning(cut state) | Q-learning | Cross Entropy| REINFORCE |
|:- | :----------- | :----------- | :--- | :--- | :--- |
| Mean score| 403.01     | 489.11      | 583.33 | 599.98| 600 |

(**cut state** means that state space is only *pacman position* without *pellets positions*)

Summary of results on `GhostsPacmanEnvironment(grid_size=10, max_steps=200, num_ghosts=2)` over 1000 games:

| | Random | Q-learning | REINFORCE |
|:-  | :----------- | :--- | :--- |
| Mean score|  113.73  | 128.47      | 143.65 |
| Mean steps| 35.077 | 38.256 | 64.482|


