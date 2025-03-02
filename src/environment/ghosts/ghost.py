from src.environment.ghosts.strategy import GhostStrategy
from src.state import Position, Map, ActionSpaceEnum
from typing import Optional


ghost_action_space = {
    ActionSpaceEnum.UP: (0, -1),
    ActionSpaceEnum.RIGHT: (1, 0),
    ActionSpaceEnum.DOWN: (0, 1),
    ActionSpaceEnum.LEFT: (-1, 0)
}


class Ghost:
    prev_action: Optional[ActionSpaceEnum]
    position: Position
    
    def __init__(self, position: Position, strategy: GhostStrategy):
        self.position = position
        self.prev_action = None
        self.strategy = strategy
    
    def move(self, map: Map):
        possible_actions = [
            action
            for action, (dx, dy) in ghost_action_space.items()
            if self.position + Position(dx, dy) not in map.walls.union(map.ghost_positions)
        ]
        print('possible actions: ', possible_actions)
        print('impossible positions: ', map.walls.union(map.ghost_positions))
        
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
