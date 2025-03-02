import pygame
from src.state import Map, ActionSpaceEnum
from src.drawer import Drawer


class PygameDrawer(Drawer):
    """
    Implementation of the Drawer using the Pygame library.

    This class handles visualization of the game environment using a grid. It draws 
    grid lines, walls, pellets (as small circles), and the Pac-Man character.
    """

    def __init__(self, grid_size=10, cell_size=40, framerate=3):
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
        self._load_sprites()


    def _load_sprites(self):
        """Load and scale all game sprites once during initialization"""
        # Load base images (replace paths with your actual sprite files)
        pacman_sprites = {
            dir: pygame.image.load(f"sprites/pacman_{dir}.png").convert_alpha()
            for dir in ["RIGHT", "LEFT", "UP", "DOWN", "DEAD"]
        }
        self.pacman_imgs = {
            dir: pygame.transform.scale(img, (self.cell_size, self.cell_size))
            for dir, img in pacman_sprites.items()
        }
        ghost_sprites = {
            color: pygame.image.load(f"sprites/ghost_{color}.png").convert_alpha()
            for color in ["green", "red", "orange", "purple", "blue", "brown"]
        }
        self.ghost_imgs = {
            color: pygame.transform.scale(img, (self.cell_size, self.cell_size))
            for color, img in ghost_sprites.items()
        }

    def draw(self, map: Map, action: ActionSpaceEnum):
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
            if pellet in map.ghost_positions:
                continue
            cx = pellet.x * self.cell_size + self.cell_size // 2
            cy = pellet.y * self.cell_size + self.cell_size // 2
            radius = self.cell_size // 8
            pygame.draw.circle(self.screen, pellet_color, (cx, cy), radius)

        # Draw ghosts with different colored sprites
        for ghost_pos in map.ghost_positions:
            if ghost_pos == map.pacman_position:
                continue
            ghost_rect = pygame.Rect(
                ghost_pos.x * self.cell_size,
                ghost_pos.y * self.cell_size,
                self.cell_size,
                self.cell_size
            )
            self.screen.blit(self.ghost_imgs[map.ghost_position_to_color[ghost_pos]], ghost_rect)
        
        # Draw Pac-Man with directional sprites
        pacman_rect = pygame.Rect(
            map.pacman_position.x * self.cell_size,
            map.pacman_position.y * self.cell_size,
            self.cell_size,
            self.cell_size
        )
        pacman_dead = map.pacman_position in map.ghost_positions
        if pacman_dead:
            self.screen.blit(self.pacman_imgs["DEAD"], pacman_rect)
        else:
            self.screen.blit(self.pacman_imgs[action.name], pacman_rect)
        
        # Update the display and maintain constant framerate
        pygame.display.flip()
        framerate = self.framerate / 10 if pacman_dead else self.framerate
        self.clock.tick(framerate)

    def close(self):
        """
        Quit Pygame and clean up associated resources.
        """
        pygame.quit()
