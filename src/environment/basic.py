import torch
from typing import Sequence

from src.environment import PacmanEnvironment
from src.state import Position, Observation, Map, ActionSpaceEnum, MapFullHash


def get_state_matrix(grid_size: int, walls: Sequence[Position],
                        pellets: Sequence[Position], pacman_position: Position) -> torch.Tensor:
    """
    Converts a Map's objects positions to a state tensor (matrix) representing the grid.

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

    state = torch.zeros((grid_size, grid_size), dtype=torch.float32)
    for pos in walls:
        state[pos.x, pos.y] = -1.0
    for pos in pellets:
        state[pos.x, pos.y] = 2.0
    state[pacman_position.x, pacman_position.y] = 1.0

    return state

class BasicPacmanEnvironment(PacmanEnvironment):
    """
    A basic grid-based implementation of the Pac-Man environment for RL.

    This environment sets up walls, pellets, and the Pac-Man, and defines
    simple dynamics such as movement, rewards for eating pellets, and penalties for
    hitting walls.
    """

    def __init__(self, grid_size=10, cell_size=40, max_steps=200, full_hash=False, inner_walls=True):
        """
        Initialize the environment parameters and reset it.

        Args:
            grid_size (int): Number of cells per side in the grid.
            cell_size (int): Pixel size of each cell (used for rendering).
            max_steps (int): Maximum number of steps allowed in an episode.
            full_hash (bool): Use full hashable maps
        """
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.max_steps = max_steps
        self.full_hash = full_hash
        self.inner_walls = inner_walls

    def get_grid_size(self):
        return self.grid_size

    def reset(self) -> Observation:
        """
        Reset the environment to its initial configuration.

        This method creates boundary walls, adds optional internal walls,
        places pellets in every non-wall cell, and initializes Pac-Man's
        starting position at the center.
        """
        # Create walls along the perimeter
        walls = set()
        for i in range(self.grid_size):
            walls.add(Position(0, i))
            walls.add(Position(self.grid_size - 1, i))
            walls.add(Position(i, 0))
            walls.add(Position(i, self.grid_size - 1))

        # Add internal walls (an example pattern)
        if self.inner_walls:
            for r in range(2, self.grid_size - 2, 2):
                walls.add(Position(r, self.grid_size // 2))

        # Populate the grid with pellets where there are no walls 
        pellets = set()
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if Position(r, c) not in walls:
                    pellets.add(Position(r, c))

        # Place Pac-Man at the grid center and remove any pellet at that position
        pacman_position = Position(self.grid_size // 2, self.grid_size // 2)
        if pacman_position in pellets:
            pellets.remove(pacman_position)

        # Initialize game state variables
        self.score = 0
        self.step_count = 0
        self.done = False

        self.max_score = len(pellets) * 10

        state_tensor = get_state_matrix(self.grid_size, walls, pellets, pacman_position)

        if self.full_hash:
            self.map = MapFullHash(
                walls=walls,
                pellets=pellets,
                pacman_position=pacman_position,
                ghost_positions=set(),
                ghost_position_to_color={},
                size=self.grid_size,
                state_tensor=state_tensor
            )
        else:
            self.map = Map(
                walls=walls,
                pellets=pellets,
                pacman_position=pacman_position,
                ghost_positions=set(),
                ghost_position_to_color={},
                size=self.grid_size,
                state_tensor=state_tensor
            )

        return Observation(
            reward=0,
            done=False,
            score=0,
            step_count=0,
            map=self.map,
        )

    def step(self, action: ActionSpaceEnum) -> Observation:
        """
        Perform one step in the environment given an action.

        Dynamics:
          - Maps the action to a directional change.
          - Applies a small movement penalty.
          - Penalizes hitting a wall (no movement happens).
          - Rewards eating a pellet and updates the score.
          - Increments the step counter and terminates the episode if there are
            no remaining pellets or if the maximum number of steps is reached.

        Args:
            action (ActionSpaceEnum): The action to perform.
            
        Returns:
            Observation: Contains updated reward, game state, and whether the episode 
                      has ended.
        """
        # If already finished, return current state with zero reward.
        if self.done:
            return Observation(
                reward=0,
                done=True,
                score=self.score,
                step_count=self.step_count,
                map=self.map
            )

        # Mapping for actions to directional moves: up, right, down, left.
        directions = {
            ActionSpaceEnum.UP: (0, -1),
            ActionSpaceEnum.RIGHT: (1, 0),
            ActionSpaceEnum.DOWN: (0, 1),
            ActionSpaceEnum.LEFT: (-1, 0)
        }
        delta = directions.get(action, (0, 0))
        
        current_position = self.map.pacman_position
        new_x = current_position.x + delta[0]
        new_y = current_position.y + delta[1]
        new_position = Position(new_x, new_y)
        
        reward = self.compute_reward(self.map, current_position, new_position)

        # Check if new position is a wall.
        if new_position in self.map.walls:
            new_position = current_position  # Remain in the same position.
        else:
            self.map.state_tensor[self.map.pacman_position.x, self.map.pacman_position.y] = 0.0
            self.map.state_tensor[new_position.x, new_position.y] = 1.0
            # Move Pac-Man to the new position.
            self.map.pacman_position = new_position
            # If a pellet is present, remove it.
            if new_position in self.map.pellets:
                self.map.pellets.remove(new_position)
                self.score += 10

        self.step_count += 1
        
        # Termination condition: either all pellets are eaten or max steps reached.
        if not self.map.pellets or self.step_count >= self.max_steps:
            self.done = True
        
        return Observation(
            reward=reward,
            done=self.done,
            score=self.score,
            step_count=self.step_count,
            map=self.map
        )

    @staticmethod
    def compute_reward(map_instance: Map, current: Position, candidate: Position) -> float:
        """
        Computes the reward for moving from the current position to a candidate new position.

        Args:
            map_instance (Map): The current map, including walls and pellets.
            current (Position): Pac-Man's current position.
            candidate (Position): The candidate new position based on the action.

        Returns:
            - reward (float): The reward to be applied.
        """
        reward = -1  # default step cost

        # Check if candidate hits a wall.
        if candidate in map_instance.walls:
            reward = -50
        else:
            # If there is a pellet at the candidate position, add reward but do not remove it here.
            if candidate in map_instance.pellets:
                reward += 10

        return reward
