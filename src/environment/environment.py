# src/environment/environment.py
from abc import ABC, abstractmethod
from src.state import Observation, ActionSpaceEnum, Map, Position

class PacmanEnvironment(ABC):
    """
    Abstract base class defining the interface for a Pac-Man environment.
    """

    @abstractmethod
    def reset(self, inner_walls: bool) -> Observation:
        pass
    
    @abstractmethod
    def get_grid_size(self) -> int:
        pass

    @abstractmethod
    def step(self, action: ActionSpaceEnum) -> Observation:
        pass

    @staticmethod
    def compute_reward(map_instance: Map, current: Position, candidate: Position) -> float:
        raise NotImplementedError()
