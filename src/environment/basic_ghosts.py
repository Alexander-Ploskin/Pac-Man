from src.environment import PacmanEnvironment
from src.state import Position, Observation, Map, ActionSpaceEnum, MapFullHash, GhostColorEnum
from src.environment.ghosts import Ghost
from src.environment.ghosts.strategy import RandomGhostStrategy
from typing import Set
import random


class GhostsPacmanEnvironment(PacmanEnvironment):
    """
    An implementation of grid-based Pac-Man environment with ghosts for RL.

    This environment sets up walls, pellets, ghosts and the Pac-Man, and defines
    simple dynamics such as Pac-Man movement, ghosts behavior, rewards for eating pellets,
    and penalties for hitting walls and ghosts.
    """

    def __init__(
        self,
        grid_size=10,
        num_ghosts=1,
        max_steps=200,
        stability_rate=0.75,
        full_hash=False):
        """
        Initialize the environment parameters and reset it.

        Args:
            grid_size (int): Number of cells per side in the grid.
            num_ghosts (int): Number of ghosts moving on the map.
            max_steps (int): Maximum number of steps allowed in an episode.
            full_hash (bool): Use full hashable maps
        """
        self.grid_size = grid_size
        self.num_ghosts = num_ghosts
        self.max_steps = max_steps
        self.full_hash = full_hash
        self.ghosts_strategy = RandomGhostStrategy(stability_rate=stability_rate)
        
    def get_grid_size(self):
        return self.grid_size

    def reset(self, inner_walls: bool = True) -> Observation:
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
        if inner_walls:
            for x in range(2, self.grid_size - 2, 2):
                walls.add(Position(x, self.grid_size // 2))
        
        # Populate the grid with pellets where there are no walls
        pellets = set()
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if Position(x, y) not in walls:
                    pellets.add(Position(x, y))
        
        # Place Pac-Man at the grid center and remove any pellet at that position
        pacman_position = Position(self.grid_size // 2, self.grid_size // 2)
        if pacman_position in pellets:
            pellets.remove(pacman_position)

        # Initialize ghosts
        valid_ghost_positions = []
        ghosts: Set[Ghost] = set()
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                position = Position(x, y)
                if position not in walls and position != pacman_position:
                    valid_ghost_positions.append(Position(x, y))
        
        colors = [color.value for color in GhostColorEnum]
        if len(colors) < self.num_ghosts:
            raise ValueError(self.num_ghosts)
        for _, color in zip(range(self.num_ghosts), colors):
            if not valid_ghost_positions:
                raise ValueError("Not enough valid positions for ghosts")
            ghost_position = random.choice(valid_ghost_positions)
            ghosts.add(Ghost(ghost_position, self.ghosts_strategy, color))
            valid_ghost_positions.remove(ghost_position)
        
        self.ghosts = ghosts
        ghost_positions = {ghost.position for ghost in ghosts}
        ghost_position_to_ghost_color = {ghost.position: ghost.color for ghost in ghosts}
        
        # Initialize game state variables
        self.score = 0
        self.step_count = 0
        self.done = False
        
        if self.full_hash:
            self.map = MapFullHash(
                walls=walls,
                pellets=pellets,
                pacman_position=pacman_position,
                ghost_positions=ghost_positions,
                ghost_position_to_color=ghost_position_to_ghost_color,
                size=self.grid_size
            )
        else:
            self.map = Map(
                walls=walls,
                pellets=pellets,
                pacman_position=pacman_position,
                ghost_positions=ghost_positions,
                ghost_position_to_color=ghost_position_to_ghost_color,
                size=self.grid_size
            )
        
        return Observation(
            reward=0,
            done=False,
            score=0,
            step_count=0,
            map=self.map
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

        # Check if new position is a wall.
        if new_position in self.map.walls:
            new_position = current_position  # Remain in the same position.
        else:
            # Move Pac-Man to the new position.
            self.map.pacman_position = new_position
            # If a pellet is present, remove it.
            if new_position in self.map.pellets:
                self.map.pellets.remove(new_position)
                self.score += 10

        for ghost in self.ghosts:
            ghost.move(self.map)
        self.map.ghost_position_to_color = {ghost.position: ghost.color for ghost in self.ghosts}
        
        # Collision detection
        if self.map.pacman_position in self.map.ghost_positions:
            self.done = True

        reward = self.compute_reward(self.map, current_position, new_position)
        
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

        # check if the candidate is eaten by a ghost
        if candidate in map_instance.ghost_positions:
            return -100
        
        # Check if candidate hits a wall.
        if candidate in map_instance.walls:
            reward = -50
        else:
            # If there is a pellet at the candidate position, add reward but do not remove it here.
            if candidate in map_instance.pellets:
                reward += 10
                

        return reward
