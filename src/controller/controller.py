from abc import ABC, abstractmethod
from enum import Enum
from src.state import Observation


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


class Controller(ABC):
    """
    Abstract base class for controllers.

    This interface defines an expected method 'get_action' that takes an observation
    of the current environment state and returns an action from the defined action space.
    Any controller (including RL models) should subclass and implement this method.
    """

    @abstractmethod
    def get_action(self, observation: Observation) -> ActionSpaceEnum:
        """
        Determines and returns an action for Pac-Man based on the given observation.

        Args:
            observation (Observation): The current state of the environment. This 
                                       contains information about walls, pellets, 
                                       Pac-Man's position, etc.

        Returns:
            ActionSpaceEnum: The selected action from the available action space.
        """
        pass
