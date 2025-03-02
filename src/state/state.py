from dataclasses import dataclass
from enum import Enum
from typing import Set, List
import numpy as np


class ActionSpaceEnum(int, Enum):
    """
    Enumeration representing possible actions for Pac-Man.

    Attributes:
        UP (int): Represents moving upward.
        RIGHT (int): Represents moving to the right.
        DOWN (int): Represents moving downward.
        LEFT (int): Represents moving to the left.
    """
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

    def get_opposite(self) -> 'ActionSpaceEnum':
        """
        Returns the opposite action.
        """
        if self == ActionSpaceEnum.UP:
            return ActionSpaceEnum.DOWN
        elif self == ActionSpaceEnum.DOWN:
            return ActionSpaceEnum.UP
        elif self == ActionSpaceEnum.LEFT:
            return ActionSpaceEnum.RIGHT
        elif self == ActionSpaceEnum.RIGHT:
            return ActionSpaceEnum.LEFT
        else:
            return self  # Return itself if no opposite is defined

@dataclass(frozen=True)
class Position:
    """
    Represents a coordinate on the game grid.

    Attributes:
        x (int): The x-coordinate (horizontal position).
        y (int): The y-coordinate (vertical position).
    """
    x: int
    y: int
    
    def __add__(self, other: 'Position') -> 'Position':
        """Adds two Positions component-wise, returning a new frozen instance."""
        if not isinstance(other, Position):
            return NotImplemented
        return Position(self.x + other.x, self.y + other.y)

@dataclass
class Map:
    """
    Encapsulates the current state of the game map.

    Attributes:
        walls (Set[Position]): Set of positions where walls are located.
        pellets (Set[Position]): Set of positions where pellets are located.
        pacman_position (Position): The current position of Pac-Man.
    """
    size: int
    walls: Set[Position]
    pellets: Set[Position]
    ghost_positions: Set[Position]
    pacman_position: Position

    directions = {
        ActionSpaceEnum.UP: (0, -1),
        ActionSpaceEnum.RIGHT: (1, 0),
        ActionSpaceEnum.DOWN: (0, 1),
        ActionSpaceEnum.LEFT: (-1, 0)
    }

    def __hash__(self):
        """
        Returns a hash value for the Map object.

        The hash is computed based on the frozen sets of walls, pellets and Pac-Man's position.
        This ensures that the Map object can be used in hash-based data structures.

        Returns:
            int: Hash value of the Map object.
        """
        return hash((self.pacman_position))

    def _state_after_action(self, action: ActionSpaceEnum) -> Position:
        """
        Calculates new position after taking a specified action.

        If no `state` is provided, the current position of Pac-Man is used as the starting point.

        Args:
            action (ActionSpaceEnum): The action to be taken.
            state (Optional[Position]): The starting position. Defaults to the current Pac-Man position.

        Returns:
            Position: The new position after taking the action.
        """
        state = self.pacman_position
        delta = self.directions.get(action, (0, 0))
        
        new_x = state.x + delta[0]
        new_y = state.y + delta[1]
        new_position = Position(new_x, new_y)

        return new_position

    def get_legal_actions(self) -> List[ActionSpaceEnum]:
        """
        Determines the list of legal actions that can be taken from a given state.

        An action is considered legal if the resulting position is not occupied by a wall, pellet.
        If no `state` is provided, the current position of Pac-Man is used as the starting point.

        Args:
            state (Optional[Position]): The starting position. Defaults to the current Pac-Man position.

        Returns:
            List[ActionSpaceEnum]: A list of legal actions from the given state.
        """

        legal_actions: List[ActionSpaceEnum] = []
        for action in self.directions.keys():
            next_state = self._state_after_action(action)
            if next_state in self.walls:
                continue
            legal_actions.append(action)
        return legal_actions


class MapFullHash(Map):
    def __hash__(self):
        return hash((frozenset(self.pellets), self.pacman_position))


@dataclass
class Observation:
    """
    Holds the result of an environment step.

    Attributes:
        reward (float): The reward received after taking an action.
        done (bool): Indicates whether the episode has terminated.
        score (int): The cumulative score.
        step_count (int): The number of steps taken.
        map (Map): The current state of the map.
    """
    reward: float
    done: bool
    score: int
    step_count: int
    map: Map
    
    def to_numpy(self):
        state = np.zeros((self.map.size, self.size))
        for wall in self.map.walls:
            state[wall.y, wall.x] = -1
        for pellet in self.map.pellets:
            state[pellet.y, pellet.x] = 1
        state[self.map.pacman_position.y, self.map.pacman_position.x] = 2
        for ghost in self.map.ghost_positions:
            state[ghost.x, ghost.y] = -2
