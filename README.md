# Pac-Man Reinforcement Learning Project

A modular framework for experimenting with reinforcement learning algorithms by training an agent to control Pac-Man in a simplified grid-based environment.

---

## Description
This project implements a simplified version of the classic Pac-Man game, designed as an environment for experimenting with reinforcement learning (RL) algorithms. The goal is to train an RL agent to control Pac-Man, navigating a grid-based maze, collecting pellets, and avoiding obstacles while maximizing the score.

The project demonstrates the power of RL techniques by providing a modular framework where various RL algorithms can be applied.

---

## 1. Build and Run

1. Clone this repository:
   ```bash
   git clone https://github.com/Alexander-Ploskin/Pac-Man.git
   cd Pac-Man
   ```
2. Install the dependencies:
   ```bash
   pip install .
   ```
3. To run random walk policy:
   ```bash
   python cli.py random
   ```
   To run Q-learning algorithm (full_hash means that states == full map's objects positions, states == position of packman otherwise):
   ```bash
   python cli.py qlearn --full_hash
   ```

---

## 2. Demonstration
Below is a demonstration of the project in action:

### Random walk policy:
![Pac-Man Gameplay random](https://raw.githubusercontent.com/Alexander-Ploskin/Pac-Man/master/assets/random.gif)

### Q-learning (SARSA update):
![Pac-Man Gameplay Q-learning](https://raw.githubusercontent.com/Alexander-Ploskin/Pac-Man/master/assets/q-learning.gif)
---

## 3. Application Architecture

The project is structured into modular components, making it easy to extend or replace individual parts.

### **Core Components**
1. **Environment (`environment`)**:
   - Defines the grid-based game world, including walls, pellets, and Pac-Man's position.
   - Handles game logic such as movement, collision detection, and reward calculation.
   - Example: `BasicPacmanEnvironment` provides a simple implementation of the environment.

2. **Controller (`controller`)**:
   - Decides Pac-Man's actions based on observations from the environment.
   - Supports different controllers, such as:
     - `BasicController`: Selects random actions for testing.
     - Future controllers can implement RL algorithms.

3. **Drawer (`drawer`)**:
   - Visualizes the game state using Pygame.
   - Example: `PygameDrawer` renders walls, pellets, and Pac-Man on a grid.

4. **State (`state`)**:
   - Encapsulates the observation data passed from the environment to the controller.
   - Includes positions of walls, pellets, and Pac-Man.

### **Workflow**
1. The environment generates an observation of the current game state.
2. The controller receives this observation and selects an action.
3. The environment processes this action, updates the game state, and calculates rewards.
4. The drawer visualizes the updated state.
