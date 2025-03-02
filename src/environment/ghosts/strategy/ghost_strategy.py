from src.state import ActionSpaceEnum
from typing import List
from abc import ABC, abstractmethod


class GhostStrategy(ABC):
    @abstractmethod
    def get_action(self, prev_action: ActionSpaceEnum, possible_actions: List[ActionSpaceEnum]) -> ActionSpaceEnum:
        pass
    