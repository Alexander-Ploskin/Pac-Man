import random
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

from src.controller import Controller
from src.state import Map, Observation, ActionSpaceEnum, Position
from src.environment import PacmanEnvironment


def map_to_state_vector(map_object: Map) -> torch.Tensor:
    """
    Converts a Map object to a state tensor representing the grid.

    Each position in the grid is encoded as follows:
    - 0: Empty
    - 1: Pac-Man
    - 2: Pellet

    Args:
        map_object (Map): The Map object.

    Returns:
        torch.Tensor: A tensor representing the grid state.
    """

    x_coords = [pos.x for pos in map_object.walls]
    y_coords = [pos.y for pos in map_object.walls]

    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)

    state = []

    for x in range(x_min, x_max + 1):
        for y in range(y_min, y_max + 1):
            position = Position(x, y)
            
            if position == map_object.pacman_position:
                state.append(1.0)
            elif position in map_object.pellets:
                state.append(2.0)
            elif position in map_object.walls:
                continue
            else:
                state.append(0.0)

    return torch.tensor(state, dtype=torch.float32)


class QNetwork(nn.Module):
    """
    A neural network for estimating Q-values of state-action pairs.
    """

    def __init__(self, state_size, action_size, hidden_size=128):
        """
        Initializes the Q-Network.

        Args:
            state_size (int): The size of the state space.
            action_size (int): The size of the action space.
            hidden_size (int): The number of units in the hidden layer.
        """
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DQN(Controller):
    """
    Deep Q-Network agent.
    """

    def __init__(self, state_size, action_size, alpha=0.001, gamma=0.98,
                 train_epsilon=0.9, test_epsilon=0.2,
                 epsilon_decay=0.99997, replay_buffer_size=10000,
                 batch_size=64, target_update_interval=1000,
                 numTraining=100000, verbose=False, device="cpu"):
        """
        Initializes the DQN agent.

        Args:
            state_size (int): The size of the state space.
            action_size (int): The size of the action space.
            alpha (float): Learning rate.
            gamma (float): Discount factor.
            train_epsilon (float): Exploration rate during training.
            test_epsilon (float): Exploration rate during testing.
            epsilon_decay (float): Decay rate for epsilon.
            replay_buffer_size (int): Size of the replay buffer.
            batch_size (int): Batch size for training.
            target_update_interval (int): Interval to update the target network.
            numTraining (int): Number of training episodes.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.alpha = alpha
        self.gamma = gamma
        self.train_epsilon = train_epsilon
        self.test_epsilon = test_epsilon
        self.epsilon = train_epsilon
        self.epsilon_decay = epsilon_decay
        self.replay_buffer_size = replay_buffer_size
        self.batch_size = batch_size
        self.target_update_interval = target_update_interval
        self.numTraining = numTraining
        self.verbose = verbose
        self.device = device

        # Q-Network and Target Network
        self.q_network = QNetwork(state_size, action_size).to(self.device)
        self.target_network = QNetwork(state_size, action_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())  # Initialize target network with Q-network weights
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.alpha)

        # Replay Buffer
        self.replay_buffer = deque(maxlen=self.replay_buffer_size)

        # Training variables
        self.episodesPassed = 0
        self.lastAction: ActionSpaceEnum | None = None
        self.lastState: Map | None = None
        self.train_ = True
        self.step_count = 0

    def state_to_tensor(self, state: Map) -> torch.Tensor:
        """
        Converts the state (Map) to a tensor that can be used as input to the neural network.

        Args:
            state (Map): The current state of Pac-Man's environment.

        Returns:
            torch.Tensor: The tensor representation of the state.
        """
        # Extract relevant features from the state
        pacman_position = state.pacman_position
        ghost_positions = state.ghost_positions
        food_positions = state.food_positions
        walls = state.walls

        # Convert features to numpy arrays
        pacman_position = np.array(pacman_position)
        ghost_positions = np.array(ghost_positions)
        food_positions = np.array(food_positions)
        walls = np.array(walls)

        # Flatten the arrays
        pacman_position = pacman_position.flatten()
        ghost_positions = ghost_positions.flatten()
        food_positions = food_positions.flatten()
        walls = walls.flatten()

        # Concatenate the features into a single array
        state_array = np.concatenate([pacman_position, ghost_positions, food_positions, walls])

        # Convert the numpy array to a PyTorch tensor
        state_tensor = torch.tensor(state_array, dtype=torch.float).to(self.device)
        return state_tensor

    def getQValue(self, state: torch.Tensor, action: int) -> float:
        """
        Retrieves the Q-value for a given state-action pair.

        Args:
            state (torch.Tensor): The current state of Pac-Man.
            action (int): The action taken.

        Returns:
            float: The Q-value associated with the given state-action pair.
        """
        with torch.no_grad():
            q_values = self.q_network(state)
            return q_values[action].item()

    def best_action(self, state: Map) -> ActionSpaceEnum | None:
        """
        Determines the best action to take in a given state based on current Q-values.

        Args:
            state (Map): The current state of Pac-Man's environment.

        Returns:
            ActionSpaceEnum | None: The best action to take; returns None if no legal actions are available.
        """
        legal_actions = state.get_legal_actions()
        if not legal_actions:
            return None

        if ((self.train_ and random.random() < self.epsilon) or
                (not self.train_ and random.random() < self.test_epsilon)):
            return random.choice(legal_actions)

        state_tensor = map_to_state_vector(state)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
            # Convert ActionSpaceEnum to integer indices
            legal_action_indices = [action.value for action in legal_actions]  # Assuming ActionSpaceEnum has integer values
            # Filter Q-values to only include legal actions
            legal_q_values = q_values[legal_action_indices]
            best_action_index = torch.argmax(legal_q_values).item()
            best_action = legal_actions[best_action_index]
        return best_action

    def remember(self, state: Map, action: ActionSpaceEnum, reward: float, next_state: Map, done: bool):
        """
        Adds a transition to the replay buffer.

        Args:
            state (Map): The current state.
            action (ActionSpaceEnum): The action taken.
            reward (float): The reward received.
            next_state (Map): The next state.
            done (bool): Whether the episode is done.
        """
        self.replay_buffer.append((state, action, reward, next_state, done))

    def learn(self):
        """
        Samples a minibatch from the replay buffer and performs a learning step.
        """
        if len(self.replay_buffer) < self.batch_size:
            return

        # Sample a minibatch from the replay buffer
        minibatch = random.sample(self.replay_buffer, self.batch_size)

        # Convert the minibatch to tensors
        states, actions, rewards, next_states, dones = zip(*minibatch)

        state_tensors = torch.stack([map_to_state_vector(state) for state in states]).to(self.device)
        action_tensors = torch.tensor([action.value for action in actions], dtype=torch.int64).to(self.device)  # Assuming ActionSpaceEnum has integer values
        reward_tensors = torch.tensor(rewards, dtype=torch.float).to(self.device)
        next_state_tensors = torch.stack([map_to_state_vector(next_state) for next_state in next_states]).to(self.device)
        done_tensors = torch.tensor(dones, dtype=torch.bool).to(self.device)

        # Compute Q(s, a) and Q(s', a')
        q_values = self.q_network(state_tensors).gather(1, action_tensors.unsqueeze(1)).squeeze()
        next_q_values = self.target_network(next_state_tensors).max(1)[0]
        next_q_values[done_tensors] = 0.0  # Zero out terminal states

        # Compute the expected Q values
        expected_q_values = reward_tensors + self.gamma * next_q_values

        # Compute the loss
        loss = nn.MSELoss()(q_values, expected_q_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update the target network
        self.step_count += 1
        if self.step_count % self.target_update_interval == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

    def run_episode(self, env: PacmanEnvironment) -> int:
        """
        Runs a single episode of training in the specified environment.

        Args:
            env (PacmanEnvironment): The environment in which Pac-Man operates.

        Returns: score
        """
        observation = env.reset()
        self.lastAction = None
        self.lastState = observation.map
        total_reward = 0

        while not observation.done:
            action = self.best_action(observation.map)
            if action is None:
                break
            next_observation = env.step(action)
            reward = next_observation.reward
            next_state = next_observation.map
            done = next_observation.done

            self.remember(self.lastState, action, reward, next_state, done)
            self.learn()

            self.lastState = next_state
            self.lastAction = action
            total_reward += reward
            observation = next_observation

        self.episodesPassed += 1
        if self.train_:
             self.epsilon = max(self.test_epsilon, self.epsilon * self.epsilon_decay)
        return observation.score

    def train(self, env: PacmanEnvironment) -> None:
        """
        Trains the agent over a specified number of episodes in the given environment.

        Args:
            env (PacmanEnvironment): The environment in which Pac-Man operates.
        """
        self.episodesPassed = 0
        self.train_ = True
        mean_score = 0.0

        pbar = tqdm(range(self.numTraining), total=self.numTraining)
        for i in pbar:
            score = self.run_episode(env)
            mean_score += score
            if (i + 1) % 100 == 0:
                pbar.set_description(f"Mean score on last 100 episodes: {mean_score / 100:.0f}, Epsilon: {self.epsilon:.4f}")
                mean_score = 0

        self.lastAction = None
        self.lastState = None

    def get_action(self, observation: Observation) -> ActionSpaceEnum | None:
        """
        Retrieves the next action to be taken based on the current observation.

        Args:
            observation (Observation): The current observation from the environment.

        Returns:
            ActionSpaceEnum | None: The selected action; returns None if no valid action is available.
        """
        self.train_ = False
        return self.best_action(observation.map)
