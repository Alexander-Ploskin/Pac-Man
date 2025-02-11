from dataclasses import dataclass
from typing import Set


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
