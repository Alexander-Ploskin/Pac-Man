from src.state import ActionSpaceEnum
from typing import List
from abc import ABC, abstractmethod


class GhostStrategy(ABC):
    """
    Abstract base class for ghost movement strategies.
    """

    @abstractmethod
    def get_action(self, prev_action: ActionSpaceEnum, possible_actions: List[ActionSpaceEnum]) -> ActionSpaceEnum:
        """
        Determine the next action for a ghost based on its previous action and possible moves.

        Args:
            prev_action (ActionSpaceEnum): The ghost's previous action.
            possible_actions (List[ActionSpaceEnum]): List of possible actions the ghost can take.

        Returns:
            ActionSpaceEnum: The chosen action for the ghost to take.
        """
        pass
