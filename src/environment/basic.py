from src.environment import PacmanEnvironment
from src.state import Position, Observation, Map


class BasicPacmanEnvironment(PacmanEnvironment):
    """
    A basic grid-based implementation of the Pac-Man environment for RL.

    This environment sets up walls, pellets, and the Pac-Man, and defines
    simple dynamics such as movement, rewards for eating pellets, and penalties for
    hitting walls.
    """

    def __init__(self, grid_size=10, cell_size=40, max_steps=200):
        """
        Initialize the environment parameters and reset it.

        Args:
            grid_size (int): Number of cells per side in the grid.
            cell_size (int): Pixel size of each cell (used for rendering).
            max_steps (int): Maximum number of steps allowed in an episode.
        """
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.max_steps = max_steps

    def reset(self):
        """
        Reset the environment to its initial configuration.

        This method creates boundary walls, adds optional internal walls,
        places pellets in every non-wall cell, and initializes Pac-Man's
        starting position at the center.
        """
        # Create walls along the perimeter
        walls = set()
        for i in range(self.grid_size):
            walls.add(Position(0, i))
            walls.add(Position(self.grid_size - 1, i))
            walls.add(Position(i, 0))
            walls.add(Position(i, self.grid_size - 1))
        
        # Add internal walls (an example pattern)
        for r in range(2, self.grid_size - 2, 2):
            walls.add(Position(r, self.grid_size // 2))
        
        # Populate the grid with pellets where there are no walls
        pellets = set()
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if Position(r, c) not in walls:
                    pellets.add(Position(r, c))
        
        # Place Pac-Man at the grid center and remove any pellet at that position
        pacman_position = Position(self.grid_size // 2, self.grid_size // 2)
        if pacman_position in pellets:
            pellets.remove(pacman_position)
        
        # Initialize game state variables
        self.score = 0
        self.step_count = 0
        self.done = False
        
        self.map = Map(
            walls=walls,
            pellets=pellets,
            pacman_position=pacman_position
        )
        
        return Observation(
            reward=0,
            done=False,
            score=0,
            step_count=0,
            map=Map
        )

    def step(self, action):
        """
        Perform one step in the environment given an action.

        Dynamics:
          - Maps the action to a directional change.
          - Applies a small movement penalty.
          - Penalizes hitting a wall (no movement happens).
          - Rewards eating a pellet and updates the score.
          - Increments the step counter and terminates the episode if there are
            no remaining pellets or if the maximum number of steps is reached.

        Args:
            action (ActionSpaceEnum): The action to perform.
            
        Returns:
            Observation: Contains updated reward, game state, and whether the episode 
                      has ended.
        """
        # If already finished, return current state with zero reward.
        if self.done:
            return Observation(
                reward=0,
                done=True,
                score=self.score,
                step_count=self.step_count,
                map=self.map
            )

        # Mapping for actions to directional moves: up, right, down, left.
        directions = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
        delta = directions.get(action, (0, 0))
        
        current_position = self.map.pacman_position
        new_x = current_position.x + delta[0]
        new_y = current_position.y + delta[1]
        new_position = Position(new_x, new_y)

        reward = -0.1  # Default step cost
        
        # Check if new position is a wall; if so, apply a collision penalty.
        if new_position in self.map.walls:
            reward = -5
            new_position = current_position  # Remain in the same position.
        else:
            # Move Pac-Man to the new position.
            self.map.pacman_position = new_position
            # If a pellet is present, remove it and update reward and score.
            if new_position in self.map.pellets:
                self.map.pellets.remove(new_position)
                reward += 10
                self.score += 10

        self.step_count += 1
        
        # Termination condition: either all pellets are eaten or max steps reached.
        if not self.map.pellets or self.step_count >= self.max_steps:
            self.done = True
        
        return Observation(
            reward=reward,
            done=self.done,
            score=self.score,
            step_count=self.step_count,
            map=self.map
        )
