import argparse
import pygame
from src.environment import BasicPacmanEnvironment
from src.drawer import PygameDrawer
from src.controller import BasicController, QLearnAgent


def run_random_controller(grid_size=10, cell_size=40, max_steps=200, framerate=10):
    """
    Runs the Pac-Man environment with a basic random controller.

    Args:
        grid_size (int): Size of the grid
        cell_size (int): Size of each cell in pixels
        max_steps (int): Maximum number of steps per episode
        framerate (int): Framerate of the game
    """
    # Initialize the environment, drawer, and controller.
    environment = BasicPacmanEnvironment(
        grid_size=grid_size,
        cell_size=cell_size,
        max_steps=max_steps
    )
    observation = environment.reset()
    drawer = PygameDrawer(
        grid_size=grid_size,
        cell_size=cell_size,
        framerate=framerate
    )
    controller = BasicController()
    running = True

    # Main game loop
    while running:
        # Process Pygame events to enable window closure.
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Get an action from the controller (random actions in this basic version).
        action = controller.get_action(observation)
        observation = environment.step(action)
        print(f"Step: {observation.step_count}, Action: {action}, Reward: {observation.reward}, Score: {observation.score}")

        # Render the current game state.
        drawer.draw(observation.map)

        # Reset the environment if the episode has ended.
        if observation.done:
            print("Episode finished. Resetting environment.")
            environment.reset()

    # Clean up the Pygame resources.
    drawer.close()


def run_qlearn_controller(grid_size=10, cell_size=40, max_steps=200, framerate=10, full_hash=False):
    """
    Runs the Pac-Man environment with a Q-learning controller.

    Args:
        grid_size (int): Size of the grid (default: 10).
        cell_size (int): Size of each cell in pixels (default: 40).
        max_steps (int): Maximum number of steps per episode (default: 200).
        framerate (int): Framerate of the game (default: 10).
        full_hash (bool): Use full hashable maps for states or not (default: False).
    """
    # Initialize the environment, drawer, and controller.
    environment = BasicPacmanEnvironment(
        grid_size=grid_size,
        cell_size=cell_size,
        max_steps=max_steps,
        full_hash=full_hash
    )
    drawer = PygameDrawer(
        grid_size=grid_size,
        cell_size=cell_size,
        framerate=framerate
    )
    controller = QLearnAgent()
    controller.train(environment)

    observation = environment.reset()
    running = True

    # Main game loop
    while running:
        # Process Pygame events to enable window closure.
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Get an action from the controller.
        action = controller.get_action(observation)
        observation = environment.step(action)
        print(f"Step: {observation.step_count}, Action: {action}, Reward: {observation.reward}, Score: {observation.score}")

        # Render the current game state.
        drawer.draw(observation.map)

        # Reset the environment if the episode has ended.
        if observation.done:
            print("Episode finished. Resetting environment.")
            environment.reset()

    # Clean up the Pygame resources.
    drawer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Pac-Man environment with different controllers.")
    parser.add_argument("controller", choices=["random", "qlearn"],
                        help="Choose the controller to run: 'random' or 'qlearn'.")
    parser.add_argument("--full_hash", action="store_true", help="Use full hashable maps for states or not")
    parser.add_argument("--grid_size", type=int, default=10, help="Size of the grid.")
    parser.add_argument("--cell_size", type=int, default=40, help="Size of each cell in pixels.")
    parser.add_argument("--max_steps", type=int, default=200, help="Maximum number of steps per episode.")
    parser.add_argument("--framerate", type=int, default=10, help="Framerate of the game.")

    args = parser.parse_args()

    if args.controller == "random":
        run_random_controller(args.grid_size, args.cell_size, args.max_steps, args.framerate)
    elif args.controller == "qlearn":
        run_qlearn_controller(args.grid_size, args.cell_size, args.max_steps, args.framerate, args.full_hash)
