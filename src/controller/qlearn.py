import random
from collections import defaultdict
import pickle

from tqdm import tqdm

from src.controller import Controller
from src.state import Map, Observation, ActionSpaceEnum
from src.environment import PacmanEnvironment


class QTable(defaultdict):
    """
    A table for storing Q-values for state-action pairs with serialization capabilities.
    Inherits from defaultdict, initializing with a default float value of 0.0 for new keys.
    """

    def __init__(self):
        """
        Initializes the QTable with a default float value for new entries.
        """
        super().__init__(float)

    def save(self, file_path: str) -> None:
        """
        Saves the Q-table to a file using pickle.
        
        Args:
            file_path (str): Path where to save the Q-table
        """
        with open(file_path, "wb") as f:
            pickle.dump(dict(self), f)

    @classmethod
    def load(cls, file_path: str) -> 'QTable':
        """
        Loads the Q-table from a file using pickle.
        
        Args:
            file_path (str): Path from where to load the Q-table
            
        Returns:
            QTable: New instance of QTable with loaded values
        """
        table = cls()
        with open(file_path, "rb") as f:
            loaded_dict = pickle.load(f)
            table.update(loaded_dict)
        return table


class QLearnAgent(Controller):
    """
    Controller that uses the Q-learning algorithm to select optimal actions in a given state.

    Example:
    qlearn = Qlearning()
    qlearn.train(env)
    observation = env.reset()
    while not observation.done:
        action = qlearn.best_action(observation)
        observation = env.step(action)
    """

    def __init__(self, alpha=0.3, train_epsilon=0.9, test_epsilon=0.2, gamma=0.98, gamma_eps=0.99997, numTraining = 100000, verbose=False):
        """
        Initializes the QLearnAgent with specified parameters.
        Args:
            alpha (float): learning rate
            train_epsilon (float): train exploration rate
            test_epsilon (float): test exploration rate
            gamma (float): discount factor
            gamma_eps (float): gamma factor for epsilon in training
            numTraining (int): number of training episodes
        """
        self.alpha = float(alpha)
        self.test_epsilon = float(test_epsilon)
        self.train_epsilon = float(train_epsilon)
        self.gamma = float(gamma)
        self.gamma_eps = float(gamma_eps)
        self.numTraining = int(numTraining)
        # Q-values
        self.q_value: QTable = QTable()
        # Number of passed episodes during the training
        self.episodesPassed: int = 0
        # Last actions in training
        self.lastAction: ActionSpaceEnum | None = None
        self.lastState: Map | None = None
        self.verbose = verbose

    def getQValue(self, state: Map, action: ActionSpaceEnum) -> float:
        """
        Retrieves the Q-value for a given state-action pair.

        Args:
            state (Position): The current state of Pac-Man.
            action (ActionSpaceEnum): The action taken.

        Returns:
            float: The Q-value associated with the given state-action pair.
        """
        return self.q_value[hash((state, action))]

    def getMaxQ(self, state: Map) -> float:
        """
        Calculates the maximum Q-value among all legal actions in a given state.

        Args:
            state (Map): The current state of Pac-Man.

        Returns:
            float: The maximum Q-value among legal actions; returns 0.0 if no legal actions are available.
        """
        q_list = [self.getQValue(state, action) for action in state.get_legal_actions()]
        return max(q_list) if q_list else 0.0

    def updateQ(self, state: Map, action: ActionSpaceEnum, reward: float, qmax: float) -> None:
        """
        Updates the Q-value for a given state-action pair based on the received reward and maximum future reward.

        Args:
            state (Position): The current state of Pac-Man.
            action (ActionSpaceEnum): The action taken.
            reward (float): The reward received after taking the action.
            qmax (float): The maximum Q-value for the next state.
        """
        q = self.getQValue(state, action)
        self.q_value[hash((state, action))] = q + self.alpha * (reward + self.gamma * qmax - q)

    def best_action(self, state: Map) -> ActionSpaceEnum | None:
        """
        Determines the best action to take in a given state based on current Q-values.

        Args:
            state (Map): The current state of Pac-Man's environment.

        Returns:
            ActionSpaceEnum | None: The best action to take; returns None if no legal actions are available.
        """
        legal_actions = state.get_legal_actions()
        if self.numTraining > 0 and self.episodesPassed / self.numTraining < 0.9 or not self.train_:
            if self.lastAction is not None:

                if self.lastAction.get_opposite() in legal_actions:
                    legal_actions.remove(self.lastAction.get_opposite())
        if ((self.train_ and random.random() < self.train_epsilon) or 
                (not self.train_ and random.random() < self.test_epsilon)):
            return random.choice(legal_actions) if legal_actions else None

        best_action = []
        best_q = float('-inf')

        for action in legal_actions:
            tmp_q = self.getQValue(state, action)
            if tmp_q > best_q:
                best_q = tmp_q
                best_action = [action]
            elif tmp_q == best_q:
                best_action.append(action)

        return random.choice(best_action) if best_action else None

    def run_episode(self, env: PacmanEnvironment) -> int:
        """
        Runs a single episode of training in the specified environment.

        Args:
            env (PacmanEnvironment): The environment in which Pac-Man operates.
    
        This method resets the environment and continues until the episode is done,
        updating Q-values based on actions taken and rewards received.

        returns: score
        """
        observation = env.reset()
        self.lastAction = None
        self.lastState = observation.map

        while not observation.done:
            action = self.best_action(observation.map)
            if action is None:
                break
            self.lastAction = action
            observation = env.step(action)
            qmax = self.getMaxQ(observation.map)
            self.updateQ(self.lastState, action, observation.reward, qmax)
            self.lastState = observation.map
        self.episodesPassed += 1

        return observation.score

    def train(self, env: PacmanEnvironment) -> None:
        """
        Trains the agent over a specified number of episodes in the given environment.

        Args:
            env (PacmanEnvironment): The environment in which Pac-Man operates.

        This method resets the Q-values and runs multiple episodes to train 
        the agent using the Q-learning algorithm.
        """
        self.q_value.clear()
        self.episodesPassed = 0
        self.train_ = True
        mean_score = 0.0

        pbar = tqdm(range(self.numTraining), total=self.numTraining)
        for i in pbar:
            if i < self.numTraining * 0.6:
                self.train_epsilon = self.train_epsilon * self.gamma_eps
            score = self.run_episode(env)
            mean_score += score
            if (i + 1) % 100 == 0:
                pbar.set_description(f"Mean score on last 100 eposodes: {mean_score / 100:.0f}, Train epsilon: {self.train_epsilon}")
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
        
        This method returns the best action based on current knowledge.
        """
        self.train_ = False

        action = self.best_action(observation.map)
        if self.lastState is not None and self.lastAction is not None:
            qmax = self.getMaxQ(observation.map)
            self.updateQ(self.lastState, self.lastAction, observation.reward, qmax)
            if self.verbose:
                print(observation.reward)
                print(f"Q-value:", self.getQValue(self.lastState, self.lastAction))
                print("Position:", self.lastState.pacman_position, self.lastAction)
                print("Hash:", hash((self.lastState, self.lastAction)))

        self.lastState = observation.map
        self.lastAction = action

        return action

    def save_model(self, file_path: str) -> None:
        """
        Save the Q-table to a file.
        
        Args:
            file_path (str): Path where to save the model
        """
        self.q_value.save(file_path)

    def load_model(self, file_path: str) -> None:
        """
        Load the Q-table from a file.
        
        Args:
            file_path (str): Path from where to load the model
        """
        self.q_value = QTable.load(file_path)
