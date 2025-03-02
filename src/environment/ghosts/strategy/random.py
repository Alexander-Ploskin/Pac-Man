from src.environment.ghosts.strategy import GhostStrategy
from src.state import ActionSpaceEnum
import random
from typing import List, Optional


class RandomGhostStrategy(GhostStrategy):
    def __init__(self, continue_direction_probability: float):
        if continue_direction_probability < 0.0 or continue_direction_probability > 1.0:
            raise ValueError('probability should be between 0 and 1')
        self.continue_direction_probability = continue_direction_probability
        
    
    def get_action(self, prev_action: Optional[ActionSpaceEnum], possible_actions: List[ActionSpaceEnum]) -> ActionSpaceEnum:
        if not possible_actions:
            raise ValueError('there is no possible actions')
        
        if prev_action and prev_action in possible_actions and random.random() < self.continue_direction_probability:
            return prev_action
        
        return random.choice(possible_actions)
