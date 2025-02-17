from typing import Iterator
from tqdm import tqdm
import itertools
import os
from src.controller import Controller, QTable
from src.state import Map, Observation, ActionSpaceEnum, Position
from src.environment import PacmanEnvironment


class ValueIterationAgent(Controller):
    """
    Controller that uses Value Iteration to compute optimal policies.
    """

    def __init__(self, environment: PacmanEnvironment, gamma=0.98, theta=1e-6, max_iterations=1000):
        """
        Initializes the ValueIterationAgent with specified parameters.
        
        Args:
            gamma (float): Discount factor.
            theta (float): Convergence threshold.
            max_iterations (int): Maximum number of iterations.
        """
        self.gamma = gamma
        self.theta = theta
        self.max_iterations = max_iterations
        self.values = QTable()
        self.environment = environment

    def compute_value(self, state: Map, environment: PacmanEnvironment) -> float:
        """
        Computes the maximum value for a given state based on legal actions.

        Args:
            state (Map): The current state of Pac-Man.

        Returns:
            float: Maximum value for the state.
        """
        legal_actions = state.get_legal_actions()
        if not legal_actions:
            return 0.0

        max_value = float('-inf')
        for action in legal_actions:
            current_position = state.pacman_position
            candidate_position = state._state_after_action(action)
            reward = environment.compute_reward(state, current_position, candidate_position)
            value = reward + self.gamma * self.values[hash(candidate_position)]
            max_value = max(max_value, value)
        
        return max_value

    def simulate_action(self, state: Map, action: ActionSpaceEnum) -> Map:
        """
        Simulates taking an action in a given state and returns the resulting state.

        Args:
            state (Map): Current state.
            action (ActionSpaceEnum): Action to simulate.

        Returns:
            Map: Resulting state after taking the action.
        """
        new_position = state._state_after_action(action)
        if new_position in state.walls:
            return state  # No movement if there's a wall

        new_pellets = set(state.pellets)
        if new_position in new_pellets:
            new_pellets.remove(new_position)

        return Map(
            walls=state.walls,
            pellets=new_pellets,
            pacman_position=new_position
        )


    def train(self, env: PacmanEnvironment):
        """
        Runs Value Iteration to compute optimal values and policies.

        Args:
            env (PacmanEnvironment): The environment to train on.
        """
        pbar = tqdm(range(self.max_iterations), total=self.max_iterations)
        mean_delta = 0.0
        for i in pbar:
            delta = 0
            states = list(self.get_training_states(env))

            for state in states:
                old_value = self.values[hash(state)]
                new_value = self.compute_value(state, env)
                self.values[hash(state)] = new_value
                delta = max(delta, abs(old_value - new_value))
            
            mean_delta += delta
            if (i + 1) % 100 == 0:
                pbar.set_description(f"Mean delta on last 100 iterations: {mean_delta / 100:.0f}")
                mean_delta = 0

            if delta < self.theta:
                break

    
    def get_training_states(self, env: PacmanEnvironment) -> Iterator[Map]:
        default_map = env.reset().map
        walls = default_map.walls
        grid_size = env.get_grid_size()
        valid_positions = []
        
        for x in range(grid_size):
            for y in range(grid_size):
                pos = Position(x, y)
                if pos not in walls:
                    valid_positions.append(pos)

        for pacman_position in valid_positions:
            # Pellet candidates: all valid cells except where Pac-Man is located.
            pellet_candidates = [pos for pos in valid_positions if pos != pacman_position]
            # The power set (all subsets) of pellet_candidates gives every possibility of pellet placement.
            # Note: There are 2^(len(pellet_candidates)) possibilities.
            for r in range(len(pellet_candidates) + 1):
                for pellet_subset in itertools.combinations(pellet_candidates, r):
                    pellets = set(pellet_subset)
                    # Optionally, if you prefer that the cell where Pac-Man stands never has a pellet
                    # (as it is immediately consumed), this setup already enforces that.
                    yield Map(walls=walls, pellets=pellets, pacman_position=pacman_position)


    def get_action(self, observation: Observation) -> ActionSpaceEnum | None:
        """
        Retrieves the optimal action based on precomputed values.

        Args:
            observation (Observation): The current observation from the environment.

        Returns:
            ActionSpaceEnum | None: The selected action; returns None if no valid action is available.
        
        This method uses the precomputed policy to select actions.
        """
        best_action = None
        best_value = float('-inf')

        for action in observation.map.get_legal_actions():
            current_position = observation.map.pacman_position
            candidate_position = self.simulate_action(observation.map, action).pacman_position
            
            reward = self.environment.compute_reward(observation.map, current_position, candidate_position)
            value = reward + self.gamma * self.values[hash(candidate_position)]
            
            if value > best_value:
                best_value = value
                best_action = action

        return best_action

    def save_model(self, file_path: str) -> None:
        """
        Save the Q-table to a file.
        
        Args:
            file_path (str): Path where to save the model
        """
        file_path = os.path.join('checkpoints', 'value_iteration', file_path)
        self.values.save(file_path)

    def load_model(self, file_path: str) -> None:
        """
        Load the values table from a file.
    
        Args:
            file_path (str): Path from where to load the model
        """
        self.values = QTable.load(file_path)
