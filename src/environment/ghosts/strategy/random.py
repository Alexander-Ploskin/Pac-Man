from src.environment.ghosts.strategy import GhostStrategy
from src.state import ActionSpaceEnum
import random
from typing import List, Optional


class RandomGhostStrategy(GhostStrategy):
    def __init__(self, stability_rate: float):
        if stability_rate < 0.0 or stability_rate > 1.0:
            raise ValueError('probability should be between 0 and 1')
        self.stability_rate = stability_rate
        
    
    def get_action(self, prev_action: Optional[ActionSpaceEnum], possible_actions: List[ActionSpaceEnum]) -> ActionSpaceEnum:
        if not possible_actions:
            raise ValueError('there is no possible actions')
        
        if prev_action and prev_action in possible_actions and random.random() < self.stability_rate:
            return prev_action
        
        return random.choice(possible_actions)
