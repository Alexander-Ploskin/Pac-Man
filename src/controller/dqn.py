import random
from collections import deque
import logging
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

from src.controller import Controller
from src.state import Map, Observation, ActionSpaceEnum
from src.environment import PacmanEnvironment


MAX_ENV_STEPS = 200


def map_to_state_vector(map_object: Map) -> torch.Tensor:
    """
    Converts a Map object to a state tensor (vector) representing the grid.

    Each position in the grid is encoded as follows:
    - 0: Empty
    - 1: Pac-Man
    - 2: Pellet

    Args:
        map_object (Map): The Map object.

    Returns:
        torch.Tensor: A tensor representing the grid state.
    """

    return map_object.state_tensor.flatten()


def map_to_state_matrix(map_object: Map) -> torch.Tensor:
    """
    Converts a Map object to a state tensor (matrix) representing the grid.

    Each position in the grid is encoded as follows:
    - 0: Empty
    - 1: Pac-Man
    - 2: Pellet
    - -1: Wall

    Args:
        map_object (Map): The Map object.

    Returns:
        torch.Tensor: A tensor representing the grid state.
    """

    return map_object.state_tensor


class QNetworkDense(nn.Module):
    """
    A neural network for estimating Q-values of state-action pairs.
    """

    def __init__(self, state_size, action_size, hidden_size=64):
        """
        Initializes the Q-Network.

        Args:
            state_size (int): The size of the state space.
            action_size (int): The size of the action space.
            hidden_size (int): The number of units in the hidden layer.
        """
        super(QNetworkDense, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class QNetworkConv(nn.Module):
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
        super(QNetworkDense, self).__init__()
        pass

    def forward(self, state):
        pass


class DQNAgent(Controller):
    """
    Deep Q-Network agent.
    """

    def __init__(self, state_size, action_size, nn_type='dense', alpha=0.01,
                 gamma=0.98, train_epsilon=0.9, test_epsilon=0.05,
                 epsilon_decay=0.99997, replay_buffer_size=1000,
                 batch_size=64, target_update_interval=1000,
                 numTraining=100000, verbose=False, device="cpu"):
        """
        Initializes the DQN agent.

        Args:
            state_size (int): The size of the state space.
            action_size (int): The size of the action space.
            nn_type (str): type of the network to use.
            alpha (float): Learning rate.
            gamma (float): Discount factor.
            test_epsilon (float): Exploration rate during testing.
            epsilon (float): Exploration rate during training.
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
        self.nn_type = nn_type
        if self.nn_type == 'dense':
            self.state_converter = map_to_state_vector
            self.q_network = QNetworkDense(state_size, action_size)
            self.target_network = QNetworkDense(state_size, action_size)
        if self.nn_type == 'conv':
            self.state_converter = map_to_state_matrix
            self.q_network = QNetworkConv(state_size, action_size)
            self.target_network = QNetworkConv(state_size, action_size)
        self.q_network = self.q_network.to(self.device)
        self.target_network = self.target_network.to(self.device)

        self.target_network.load_state_dict(self.q_network.state_dict())  # Initialize target network with Q-network weights

        for param in self.target_network.parameters():
            param.requires_grad = False

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.alpha)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.numTraining * MAX_ENV_STEPS // self.target_update_interval)  # 200 is from maximum number of samples in environment

        self.loss = nn.MSELoss()

        # Replay Buffer
        self.replay_buffer = deque(maxlen=self.replay_buffer_size)

        # Training variables
        self.episodesPassed = 0
        self.lastAction: ActionSpaceEnum | None = None
        self.lastState: Map | None = None
        self.train_ = True
        self.step_count = 0

        # Logger setup
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler = logging.FileHandler('dqn_agent.log')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

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
            state = state.to(self.device)
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
        actions = list(Map.directions.keys())
        if ((self.train_ and random.random() < self.epsilon) or
                (not self.train_ and random.random() < self.test_epsilon)):
            return random.choice(actions)

        state_tensor = self.state_converter(state)
        with torch.no_grad():
            state_tensor = state_tensor.to(self.device)
            q_values = self.q_network(state_tensor)
            best_action_index = torch.argmax(q_values).item()
            best_action = actions[best_action_index]
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

    def learning_step(self):
        """
        Samples a minibatch from the replay buffer and performs a learning step.
        """
        # full_time_start = time.perf_counter()
        # measure_time = 0.0
        if len(self.replay_buffer) < self.batch_size:
            return

        # Sample a minibatch from the replay buffer
        minibatch = random.sample(self.replay_buffer, self.batch_size)

        # Convert the minibatch to tensors
        states, actions, rewards, next_states, dones = zip(*minibatch)

        encodings = {k: i for i, k in enumerate(Map.directions.keys())}

        
        state_tensors = torch.stack([self.state_converter(state) for state in states]).to(self.device)
        action_tensors = torch.tensor([encodings[action] for action in actions], dtype=torch.int64).to(self.device)
        reward_tensors = torch.tensor(rewards, dtype=torch.float).to(self.device)
        next_state_tensors = torch.stack([self.state_converter(next_state) for next_state in next_states]).to(self.device)
        done_tensors = torch.tensor(dones, dtype=torch.bool).to(self.device)

        # Compute Q(s, a) and Q(s', a')
        outputs = self.q_network(state_tensors)
        q_values = outputs[torch.arange(outputs.size(0)), action_tensors]
        # q_values = self.q_network(state_tensors).gather(1, action_tensors.unsqueeze(1)).squeeze()
        next_q_values, _ = torch.max(self.target_network(next_state_tensors), dim=-1)
        next_q_values[done_tensors] = 0.0  # Zero out terminal states

        # Compute the expected Q values
        expected_q_values = reward_tensors + self.gamma * next_q_values

        # Compute the loss
        loss = self.loss(q_values, expected_q_values)

        # Optimize the model
        # measure_start_time = time.perf_counter()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # measure_end_time = time.perf_counter()
        # measure_time += measure_end_time - measure_start_time

        # Update the target network
        self.step_count += 1
        if self.step_count % self.target_update_interval == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

            # for param in self.target_network.parameters():
            #     param.requires_grad = False
            self.scheduler.step()
        
            # Log the loss and learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.logger.info(f"Step: {self.step_count}, Loss: {loss.item():.4f}, Learning Rate: {current_lr:.6f}")
        
        # full_time_end = time.perf_counter()
        # print(full_time_end - full_time_start, measure_time, measure_time / (full_time_end - full_time_start))

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
            next_observation = env.step(action)
            reward = next_observation.reward
            next_state = next_observation.map
            done = next_observation.done

            self.remember(self.lastState, action, reward, next_state, done)
            self.learning_step()

            self.lastState = next_state
            self.lastAction = action
            total_reward += reward
            observation = next_observation

        self.episodesPassed += 1
        if self.train_:
             self.epsilon = max(self.test_epsilon, self.epsilon * self.epsilon_decay)
        return total_reward, observation.score

    def train(self, env: PacmanEnvironment) -> None:
        """
        Trains the agent over a specified number of episodes in the given environment.

        Args:
            env (PacmanEnvironment): The environment in which Pac-Man operates.
        """
        self.episodesPassed = 0
        self.train_ = True
        mean_score = 0.0
        mean_reward = 0.0

        pbar = tqdm(range(self.numTraining), total=self.numTraining)
        for i in pbar:
            total_reward, score = self.run_episode(env)
            mean_score += score
            mean_reward += total_reward
            if (i + 1) % 10 == 0:
                pbar.set_description(f"Mean score on last 10 episodes: {mean_score / 10:.0f}, Mean reward on last 10 episodes: {mean_reward / 10:.0f}, Epsilon: {self.epsilon:.4f}")
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

    def save(self, filename: str) -> None:
        """
        Saves the Q-network, target network, optimizer state, replay buffer, and other relevant parameters to a file.

        Args:
            filename (str): The path to the file where the agent's state will be saved.
        """
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'replay_buffer': self.replay_buffer,
            'epsilon': self.epsilon,
            'episodes_passed': self.episodesPassed,
            'step_count': self.step_count,
            'alpha': self.alpha,
            'gamma': self.gamma,
            'test_epsilon': self.test_epsilon,
            'epsilon_decay': self.epsilon_decay,
            'batch_size': self.batch_size,
            'target_update_interval': self.target_update_interval,
            'num_training': self.numTraining,
            'nn_type': self.nn_type
        }, filename)

    def load(self, filename: str) -> None:
        """
        Loads the Q-network, target network, optimizer state, replay buffer, and other relevant parameters from a file.

        Args:
            filename (str): The path to the file from which the agent's state will be loaded.
        """
        checkpoint = torch.load(filename, map_location=self.device)

        # Load network type
        self.nn_type = checkpoint.get('nn_type', 'dense')

        # Initialize networks based on loaded type
        if self.nn_type == 'dense':
            self.state_converter = map_to_state_vector
            self.q_network = QNetworkDense(self.state_size, self.action_size).to(self.device)
            self.target_network = QNetworkDense(self.state_size, self.action_size).to(self.device)
        elif self.nn_type == 'conv':
            self.state_converter = map_to_state_matrix
            self.q_network = QNetworkConv(self.state_size, self.action_size).to(self.device)
            self.target_network = QNetworkConv(self.state_size, self.action_size).to(self.device)
        else:
            raise ValueError(f"Unknown network type: {self.nn_type}")

        # Load state dictionaries
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])

        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # Manually move optimizer's state to the correct device
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)
        
        # Load scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.numTraining * MAX_ENV_STEPS // self.target_update_interval)
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # Load other parameters
        self.replay_buffer = checkpoint['replay_buffer']
        self.epsilon = checkpoint['epsilon']
        self.episodesPassed = checkpoint['episodes_passed']
        self.step_count = checkpoint['step_count']
        self.alpha = checkpoint['alpha']
        self.gamma = checkpoint['gamma']
        self.test_epsilon = checkpoint['test_epsilon']
        self.epsilon_decay = checkpoint['epsilon_decay']
        self.batch_size = checkpoint['batch_size']
        self.target_update_interval = checkpoint['target_update_interval']
        self.numTraining = checkpoint['num_training']
