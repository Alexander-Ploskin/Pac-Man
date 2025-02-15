from dataclasses import dataclass, field
from enum import Enum
from typing import Set, List


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

@dataclass
class Map:
    """
    Encapsulates the current state of the game map.

    Attributes:
        walls (Set[Position]): Set of positions where walls are located.
        pellets (Set[Position]): Set of positions where pellets are located.
        pacman_position (Position): The current position of Pac-Man.
    """
    walls: Set[Position]
    pellets: Set[Position]
    pacman_position: Position
    ghosts: Set[Position] = field(default_factory=set)

    directions = {
        ActionSpaceEnum.UP: (-1, 0),
        ActionSpaceEnum.RIGHT: (0, 1),
        ActionSpaceEnum.DOWN: (1, 0),
        ActionSpaceEnum.LEFT: (0, -1)
    }

    def __hash__(self):
        """
        Returns a hash value for the Map object.

        The hash is computed based on the frozen sets of walls, pellets, ghosts and Pac-Man's position.
        This ensures that the Map object can be used in hash-based data structures.

        Returns:
            int: Hash value of the Map object.
        """
        return hash((frozenset(self.walls), frozenset(self.pellets), frozenset(self.ghosts), self.pacman_position))

    def _state_after_action(self, action: ActionSpaceEnum, state: Position | None = None) -> Position:
        """
        Calculates new position after taking a specified action.

        If no `state` is provided, the current position of Pac-Man is used as the starting point.

        Args:
            action (ActionSpaceEnum): The action to be taken.
            state (Optional[Position]): The starting position. Defaults to the current Pac-Man position.

        Returns:
            Position: The new position after taking the action.
        """
        if state is None:
            state = self.pacman_position
        delta = self.directions.get(action, (0, 0))
        
        new_x = state.x + delta[0]
        new_y = state.y + delta[1]
        new_position = Position(new_x, new_y)

        return new_position

    def get_legal_actions(self, state: Position | None = None) -> List[ActionSpaceEnum]:
        """
        Determines the list of legal actions that can be taken from a given state.

        An action is considered legal if the resulting position is not occupied by a wall, pellet, or ghost.
        If no `state` is provided, the current position of Pac-Man is used as the starting point.

        Args:
            state (Optional[Position]): The starting position. Defaults to the current Pac-Man position.

        Returns:
            List[ActionSpaceEnum]: A list of legal actions from the given state.
        """
        if state is None:
            state = self.pacman_position
        legal_actions: List[ActionSpaceEnum] = []
        for action in self.directions.keys():
            next_state = self._state_after_action(action, state)
            if next_state in self.walls or next_state in self.ghosts:
                continue
            legal_actions.append(action)
        return legal_actions

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
