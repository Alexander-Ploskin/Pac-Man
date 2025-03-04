# Pac-Man Reinforcement Learning Project

A modular framework for experimenting with reinforcement learning algorithms by training an agent to control Pac-Man in a simplified grid-based environment.

---

## Description
This project implements a simplified version of the classic Pac-Man game, designed as an environment for experimenting with reinforcement learning (RL) algorithms. The goal is to train an RL agent to control Pac-Man, navigating a grid-based maze, collecting pellets, avoiding ghosts, and maximizing the score.

The project demonstrates the power of RL techniques by providing a modular framework where various RL algorithms can be applied.

---

## MDP Problem Definition

We model the Pac-Man game as a Markov Decision Process (MDP) defined by the tuple $(\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R})$, where:

### State Space $\mathcal{S}$

The state space is represented by a matrix $S \in \mathbb{R}^{n \times n}$, where $n$ is the grid size. Each element $s_{ij}$ of the matrix can take the following values:

$$
s_{ij} = \begin{cases}
0 & \text{empty cell} \\
1 & \text{pellet} \\
-1 & \text{wall} \\
2 & \text{Pac-Man position} \\
-2 & \text{ghost position}
\end{cases}
$$

### Action Space $\mathcal{A}$

The action space is defined as:

$$
\mathcal{A} = \{\text{Up}, \text{Down}, \text{Left}, \text{Right}\}
$$

Mathematically, we can represent these actions as unit vectors:

$$
\text{Up} = (-1, 0), \text{Down} = (1, 0), \text{Left} = (0, -1), \text{Right} = (0, 1)
$$

### Reward Function $\mathcal{R}$

The reward function $$\mathcal{R}: \mathcal{S} \times \mathcal{A} \times \mathcal{S} \rightarrow \mathbb{R}$$ is defined as:

$$
\mathcal{R}(s, a, s') = \begin{cases}
10 & \text{if Pac-Man eats a pellet} \\
-1 & \text{for each move} \\
-50 & \text{if Pac-Man crashes into a wall} \\
-100 & \text{if Pac-Man collides with a ghost} \\
0 & \text{otherwise}
\end{cases}
$$

### Transition Model $\mathcal{P}$

The transition probability $\mathcal{P}(s'|s,a)$ is defined as follows:

For Pac-Man:
$$\mathcal{P}(s'|s,a) = \begin{cases}
1 & \text{if } s' \text{ is the deterministic result of action } a \text{ in state } s \\
0 & \text{otherwise}
\end{cases}$$

For ghosts:
Let $d$ be the current direction of a ghost, and $D$ be the set of possible directions excluding walls.

$$
\mathcal{P}(d'|d) = \begin{cases}
0.8125 & \text{if } d' = d \text{ and } d \in D \\
\frac{0.1875}{|D|-1} & \text{if } d' \neq d \text{ and } d' \in D \\
0 & \text{if } d' \notin D
\end{cases}
$$

### Finish Conditions

The game terminates when:
1. Pac-Man collects all pellets: $$\sum_{i,j} \mathbb{1}_{s_{ij}=1} = 0$$
2. Pac-Man collides with a ghost: $$\exists (i,j) : s_{ij} = 2 \wedge (s_{i+1,j} = -2 \vee s_{i-1,j} = -2 \vee s_{i,j+1} = -2 \vee s_{i,j-1} = -2)$$

---

## Build and Run

1. Clone this repository:
   ```bash
   git clone https://github.com/Alexander-Ploskin/Pac-Man.git
   cd Pac-Man
   ```
2. Install the dependencies:
   ```bash
   pip install .
   ```
3. To run the game with different configurations:
   ```bash
   python cli.py controller=random environment=basic
   python cli.py controller=qlearn environment=ghosts
   python cli.py controller=value_iteration environment=basic
   ```

4. To load and run a previously trained model:
   ```bash
   python cli.py controller=qlearn controller.qlearn.model_path=checkpoints/qlearn//checkpoint.pkl environment=ghosts
   ```

Note: When using model_path:
- If the file exists, the model will be loaded from it
- If the path doesn't exist or is omitted, the model will be trained from scratch

---

## Demonstration
Below is a demonstration of the project in action:

### Random walk policy:
![Pac-Man Gameplay random](https://raw.githubusercontent.com/Alexander-Ploskin/Pac-Man/master/assets/random.gif)

### Q-learning (SARSA update):
![Pac-Man Gameplay Q-learning](https://raw.githubusercontent.com/Alexander-Ploskin/Pac-Man/master/assets/q-learning.gif)

### Value Iteration
![Pac-Man Gameplay Value Iteration](https://raw.githubusercontent.com/Alexander-Ploskin/Pac-Man/master/assets/value_iteration.gif)

### Cross Entropy
![Pac-Man Gameplay Cross Entropy](https://raw.githubusercontent.com/Alexander-Ploskin/Pac-Man/master/assets/cross_entropy.gif)

---

## Application Architecture

The project is structured into modular components, making it easy to extend or replace individual parts.

### Core Components
1. **Environment (`environment`)**:
   - Defines the game world, including walls, pellets, ghosts, and Pac-Man's position.
   - Handles game logic such as movement, collision detection, and reward calculation.
   - Implementations: `BasicPacmanEnvironment` and `GhostsPacmanEnvironment`

2. **Controller (`controller`)**:
   - Decides Pac-Man's actions based on observations from the environment.
   - Supports different controllers:
     - `BasicController`: Selects random actions for testing.
     - `QLearnAgent`: Implements Q-learning algorithm.
     - `ValueIterationAgent`: Implements Value Iteration algorithm.

3. **Drawer (`drawer`)**:
   - Visualizes the game state using Pygame.
   - `PygameDrawer` renders walls, pellets, ghosts, and Pac-Man on a grid.

4. **State (`state`)**:
   - Encapsulates the observation data passed from the environment to the controller.
   - Includes positions of walls, pellets, ghosts, and Pac-Man.

### Workflow
1. The environment generates an observation of the current game state.
2. The controller receives this observation and selects an action.
3. The environment processes this action, updates the game state, and calculates rewards.
4. The drawer visualizes the updated state.
