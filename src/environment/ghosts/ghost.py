from src.environment.ghosts.strategy import GhostStrategy
from src.state import Position, Map, ActionSpaceEnum, GhostColorEnum
from typing import Optional
from enum import Enum


ghost_action_space = {
    ActionSpaceEnum.UP: (0, -1),
    ActionSpaceEnum.RIGHT: (1, 0),
    ActionSpaceEnum.DOWN: (0, 1),
    ActionSpaceEnum.LEFT: (-1, 0)
}

class Ghost:
    """
    Represents a ghost in the Pac-Man game.

    Attributes:
        prev_action (Optional[ActionSpaceEnum]): The ghost's previous action.
        position (Position): The current position of the ghost.
        strategy (GhostStrategy): The strategy used by the ghost for movement.
        color (GhostColorEnum): The color of the ghost.
    """

    prev_action: Optional[ActionSpaceEnum]
    position: Position
    
    def __init__(self, position: Position, strategy: GhostStrategy, color: GhostColorEnum):
        """
        Initialize a new Ghost instance.

        Args:
            position (Position): The initial position of the ghost.
            strategy (GhostStrategy): The movement strategy for the ghost.
            color (GhostColorEnum): The color of the ghost.
        """
        self.position = position
        self.prev_action = None
        self.strategy = strategy
        self.color = color
    
    def move(self, map: Map):
        """
        Move the ghost on the map based on its strategy.

        Args:
            map (Map): The current game map.
        """
        possible_actions = [
            action
            for action, (dx, dy) in ghost_action_space.items()
            if self.position + Position(dx, dy) not in map.walls.union(map.ghost_positions)
        ]
        
        if not possible_actions:
            return
        
        action = self.strategy.get_action(self.prev_action, possible_actions)
        self.prev_action = action
        dx, dy = ghost_action_space[action]
        x, y = self.position.x + dx, self.position.y + dy
        new_position = Position(x, y)
        map.ghost_positions.remove(self.position)
        map.ghost_positions.add(new_position)
        self.position = new_position
