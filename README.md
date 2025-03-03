# Pac-Man Reinforcement Learning Project

A modular framework for experimenting with reinforcement learning algorithms by training an agent to control Pac-Man in a simplified grid-based environment.

---

## Description
This project implements a simplified version of the classic Pac-Man game, designed as an environment for experimenting with reinforcement learning (RL) algorithms. The goal is to train an RL agent to control Pac-Man, navigating a grid-based maze, collecting pellets, avoiding ghosts, and maximizing the score.

The project demonstrates the power of RL techniques by providing a modular framework where various RL algorithms can be applied.

---

## MDP Problem Definition

To apply reinforcement learning, we model the Pac-Man game as a Markov Decision Process (MDP). The key elements of our MDP are defined as follows:

*   **States:** A state `s` is defined by Pac-Man's current position, the remaining pellets' positions, and ghost positions on the grid.
*   **Actions:** Pac-Man can take one of four actions: `Up`, `Down`, `Left`, or `Right`, corresponding to movements in those directions.
*   **Rewards:**
    *   **+10 Reward:** Eating a pellet grants a reward of +10.
    *   **-1 Reward:** Each move costs a small penalty of -1 to encourage Pac-Man to find the shortest path to pellets and finish the game faster.
    *   **-50 Reward:** Crashing into a wall yields a reward of -50.
    *   **-100 Reward:** Colliding with a ghost results in a reward of -100.
*   **Transition Model:** The transition model `P(s'|s, a)` defines the probability of transitioning to state `s'` after taking action `a` in state `s`. In our environment, transitions are deterministic for Pac-Man's movements, while ghost movements introduce stochasticity.

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
