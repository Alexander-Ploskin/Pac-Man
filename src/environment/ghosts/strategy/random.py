from src.environment.ghosts.strategy import GhostStrategy
from src.state import ActionSpaceEnum
import random
from typing import List, Optional


class RandomGhostStrategy(GhostStrategy):
    """
    A ghost movement strategy that chooses actions randomly with a tendency to maintain direction.

    Attributes:
        stability_rate (float): The probability of the ghost maintaining its previous direction.
    """

    def __init__(self, stability_rate: float):
        """
        Initialize the RandomGhostStrategy.

        Args:
            stability_rate (float): The probability (between 0 and 1) of maintaining the previous direction.

        Raises:
            ValueError: If stability_rate is not between 0 and 1.
        """
        if stability_rate < 0.0 or stability_rate > 1.0:
            raise ValueError('probability should be between 0 and 1')
        self.stability_rate = stability_rate
    
    def get_action(self, prev_action: Optional[ActionSpaceEnum], possible_actions: List[ActionSpaceEnum]) -> ActionSpaceEnum:
        """
        Choose the next action for the ghost.

        Args:
            prev_action (Optional[ActionSpaceEnum]): The ghost's previous action, if any.
            possible_actions (List[ActionSpaceEnum]): List of possible actions the ghost can take.

        Returns:
            ActionSpaceEnum: The chosen action for the ghost.

        Raises:
            ValueError: If there are no possible actions.
        """
        if not possible_actions:
            raise ValueError('there is no possible actions')
        
        if prev_action and prev_action in possible_actions and random.random() < self.stability_rate:
            return prev_action
        
        return random.choice(possible_actions)
