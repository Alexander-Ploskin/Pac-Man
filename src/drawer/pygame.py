import pygame
from src.state import Map
from src.drawer import Drawer


class PygameDrawer(Drawer):
    """
    Implementation of the Drawer using the Pygame library.

    This class handles visualization of the game environment using a grid. It draws 
    grid lines, walls, pellets (as small circles), and the Pac-Man character.
    """

    def __init__(self, grid_size=10, cell_size=40, framerate=1):
        """
        Initializes the Pygame drawer with display settings.

        Args:
            grid_size (int): Number of cells on one side of the grid.
            cell_size (int): Pixel dimension of each grid cell.
            framerate (int): Frame rate for screen updates.
        """
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.framerate = framerate

        self.clock = pygame.time.Clock()
        pygame.init()

        # Set up the display window based on grid and cell size
        self.screen = pygame.display.set_mode(
            (self.grid_size * self.cell_size, self.grid_size * self.cell_size)
        )
        pygame.display.set_caption("Pac-Man RL Environment")

    def draw(self, map: Map):
        """
        Render the current game state on the Pygame display.

        This function draws grid lines, the walls, pellets, and Pac-Man on the screen.
        
        Args:
            map (Map): The current state of the game containing positions of 
                                walls, pellets, and Pac-Man.
        """
        # Define colors for the background and game elements
        bg_color = (0, 0, 0)
        wall_color = (50, 50, 50)
        pellet_color = (255, 255, 0)
        ghost_color = (255, 0, 255)
        pacman_color = (255, 255, 0)
        
        # Fill the background
        self.screen.fill(bg_color)
        
        # Draw grid lines for visual structure
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                rect = pygame.Rect(
                    c * self.cell_size, r * self.cell_size,
                    self.cell_size, self.cell_size
                )
                pygame.draw.rect(self.screen, (30, 30, 30), rect, 1)
        
        # Draw each wall as a filled rectangle
        for wall in map.walls:
            rect = pygame.Rect(
                wall.x * self.cell_size, wall.y * self.cell_size,
                self.cell_size, self.cell_size
            )
            pygame.draw.rect(self.screen, wall_color, rect)
        
        # Draw each pellet as a small circle
        for pellet in map.pellets:
            cx = pellet.x * self.cell_size + self.cell_size // 2
            cy = pellet.y * self.cell_size + self.cell_size // 2
            radius = self.cell_size // 8
            pygame.draw.circle(self.screen, pellet_color, (cx, cy), radius)

        for ghost in map.ghost_positions:
            cx = ghost.x * self.cell_size + self.cell_size // 2
            cy = ghost.y * self.cell_size + self.cell_size // 2
            radius = self.cell_size // 2 - 4
            pygame.draw.circle(self.screen, ghost_color, (cx, cy), radius)
        
        # Draw Pac-Man as a larger circle to highlight its presence
        cx = map.pacman_position.x * self.cell_size + self.cell_size // 2
        cy = map.pacman_position.y * self.cell_size + self.cell_size // 2
        radius = self.cell_size // 2 - 4
        pygame.draw.circle(self.screen, pacman_color, (cx, cy), radius)
        
        # Update the display and maintain constant framerate
        pygame.display.flip()
        self.clock.tick(self.framerate)

    def close(self):
        """
        Quit Pygame and clean up associated resources.
        """
        pygame.quit()
