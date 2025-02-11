from abc import ABC, abstractmethod
from src.state import Map


class Drawer(ABC):
    """
    Abstract base class for drawing the game state.

    Implementations of this class must provide methods for drawing the current
    state and for cleaning up resources when done.
    """

    @abstractmethod
    def draw(self, map: Map) -> None:
        """
        Draw the environment using the provided map data.

        Args:
            map (Map): Contains current positions of walls, pellets, and Pac-Man.
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """
        Clean up any resources used by the drawer (e.g., close a window).
        """
        pass
