from abc import ABC, abstractmethod
from src.controller import ActionSpaceEnum
from src.state import Observation


class PacmanEnvironment(ABC):
    """
    Abstract base class defining the interface for a Pac-Man environment.

    Any concrete implementation must provide reset and step functionalities.
    """

    @abstractmethod
    def reset(self) -> None:
        """
        Reset the environment to its initial state.
        """
        pass

    @abstractmethod
    def step(self, action: ActionSpaceEnum) -> Observation:
        """
        Advance the environment one step based on the provided action.

        Args:
            action (ActionSpaceEnum): The action to be executed.
            
        Returns:
            Observation: The outcome of the action including reward and updated map data.
        """
        pass
