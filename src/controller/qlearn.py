import random
from collections import defaultdict

from src.controller import Controller
from src.state import Position, Map, Observation, ActionSpaceEnum
from src.environment import PacmanEnvironment


class QTable(defaultdict):
    """
    A table for storing Q-values for state-action pairs.

    Inherits from defaultdict, initializing with a default float value of 0.0 for new keys.
    """

    def __init__(self):
        """
        Initializes the QTable with a default float value for new entries.
        """
        super().__init__(float)


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

    def __init__(self, alpha=0.3, train_epsilon=0.9, test_epsilon=0.1, gamma=0.98, numTraining = 50000):
        """
        Initializes the QLearnAgent with specified parameters.
        Args:
            alpha (float): learning rate
            train_epsilon (float): train exploration rate
            test_epsilon (float): test exploration rate
            gamma (float): discount factor
            numTraining (int): number of training episodes
        """
        self.alpha = float(alpha)
        self.test_epsilon = float(test_epsilon)
        self.train_epsilon = float(train_epsilon)
        self.gamma = float(gamma)
        self.numTraining = int(numTraining)
        # Q-values
        self.q_value: QTable = QTable()
        # Number of passed episodes during the training
        self.episodesPassed: int = 0
        # Last actions in training
        self.lastAction: ActionSpaceEnum | None = None
        self.lastState: Position | None = None

    def getQValue(self, state: Position, action: ActionSpaceEnum) -> float:
        """
        Retrieves the Q-value for a given state-action pair.

        Args:
            state (Position): The current state of Pac-Man.
            action (ActionSpaceEnum): The action taken.

        Returns:
            float: The Q-value associated with the given state-action pair.
        """
        return self.q_value[(state, action)]

    def getMaxQ(self, state: Position, map: Map) -> float:
        """
        Calculates the maximum Q-value among all legal actions in a given state.

        Args:
            state (Position): The current state of Pac-Man.
            map (Map): The game map containing legal actions.

        Returns:
            float: The maximum Q-value among legal actions; returns 0.0 if no legal actions are available.
        """
        q_list = [self.getQValue(state, action) for action in map.get_legal_actions(state)]
        return max(q_list) if q_list else 0.0

    def updateQ(self, state: Position, action: ActionSpaceEnum, reward: float, qmax: float) -> None:
        """
        Updates the Q-value for a given state-action pair based on the received reward and maximum future reward.

        Args:
            state (Position): The current state of Pac-Man.
            action (ActionSpaceEnum): The action taken.
            reward (float): The reward received after taking the action.
            qmax (float): The maximum Q-value for the next state.
        """
        q = self.getQValue(state, action)
        self.q_value[(state, action)] = q + self.alpha * (reward + self.gamma * qmax - q)

    def best_action(self, state: Position, map: Map) -> ActionSpaceEnum | None:
        """
        Determines the best action to take in a given state based on current Q-values.

        Args:
            state (Position): The current state of Pac-Man.
            map (Map): The game map containing legal actions.

        Returns:
            ActionSpaceEnum | None: The best action to take; returns None if no legal actions are available.
        """
        legal_actions = map.get_legal_actions(state)
        # in the first half of training, the agent is forced not to stop
        # or turn back while not being chased by the ghost
        if self.numTraining > 0 and self.episodesPassed / self.numTraining < 0.5 or not self.train_:
            if self.lastAction is not None:
                # distance0 = state.getPacmanPosition()[0]- state.getGhostPosition(1)[0]
                # distance1 = state.getPacmanPosition()[1]- state.getGhostPosition(1)[1]
                # if math.sqrt(distance0**2 + distance1**2) < 2:
                #     best_action = last_action.get_opposite()
                if self.lastAction.get_opposite() in legal_actions:
                    legal_actions.remove(self.lastAction.get_opposite())
        if ((self.train_ and random.random() < self.train_epsilon) or 
                (not self.train_ and random.random() < self.test_epsilon)):
            return random.choice(legal_actions) if legal_actions else None

        best_action = None
        best_q = float('-inf')

        for action in legal_actions:
            tmp_q = self.getQValue(state, action)
            if tmp_q > best_q:
                best_q = tmp_q
                best_action = action

        return best_action

    def run_episode(self, env: PacmanEnvironment) -> None:
        """
        Runs a single episode of training in the specified environment.

        Args:
            env (PacmanEnvironment): The environment in which Pac-Man operates.
    
        This method resets the environment and continues until the episode is done,
        updating Q-values based on actions taken and rewards received.
        """
        observation = env.reset()
        self.lastAction = None
        self.lastState = observation.map.pacman_position

        while not observation.done:
            action = self.best_action(observation.map.pacman_position, observation.map)
            if action is None:
                break
            self.lastAction = action
            observation = env.step(action)
            qmax = self.getMaxQ(observation.map.pacman_position, observation.map)
            self.updateQ(self.lastState, action, observation.reward, qmax)
            self.lastState = observation.map.pacman_position
        self.episodesPassed += 1

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

        for i in range(self.numTraining):
            if i < self.numTraining * 0.5:
                self.train_epsilon = self.train_epsilon * ((self.numTraining-(i % 25)) / self.numTraining)
            self.run_episode(env)
            if (i + 1) % 100 == 0:
                print(i + 1)

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

        action = self.best_action(observation.map.pacman_position, observation.map)
        self.lastAction = action

        return action
