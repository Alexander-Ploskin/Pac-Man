import pygame
import os
from uuid import uuid4
from src.environment import BasicPacmanEnvironment
from src.drawer import PygameDrawer
from src.controller import BasicController, QLearnAgent

def run_algorithm(environment, drawer, controller):
    """
    Runs the main game loop with the provided environment, drawer, and controller.

    Args:
        environment: Instance of the Pac-Man environment.
        drawer: Instance of the Pygame drawer.
        controller: Instance of the controller.
    """
    observation = environment.reset()
    running = True

    while running:
        # Process Pygame events to enable window closure.
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Get an action from the controller and perform a step in the environment.
        action = controller.get_action(observation)
        observation = environment.step(action)
        print(f"Step: {observation.step_count}, Action: {action}, Reward: {observation.reward}, Score: {observation.score}")

        # Render the current game state.
        drawer.draw(observation.map)

        # Reset the environment if the episode has ended.
        if observation.done:
            print("Episode finished. Resetting environment.")
            environment.reset()

    # Clean up Pygame resources.
    drawer.close()


def create_environment(environment_name, full_hash=True):
    """
    Creates and returns an instance of the specified environment.

    Args:
        environment_name (str): Name of the environment to create ('basic').

    Returns:
        An environment instance.
    """
    if environment_name == 'basic':
        environment = BasicPacmanEnvironment(full_hash=full_hash)
    else:
        raise ValueError(f"Invalid environment name: {environment_name}")

    return environment


def create_drawer(grid_size=10, cell_size=40, framerate=10):
    """
    Creates and returns an instance of the Pygame drawer.

    Args:
        grid_size (int): Size of the grid.
        cell_size (int): Size of each cell in pixels.
        framerate (int): Framerate of the game.

    Returns:
        PygameDrawer: Instance of the drawer.
    """
    return PygameDrawer(
        grid_size=grid_size,
        cell_size=cell_size,
        framerate=framerate
    )


def create_random_controller():
    """
    Creates and returns an instance of a random controller.

    Returns:
        BasicController: Instance of the controller.
    """
    return BasicController()


def create_qlearn_controller(environment, model_path, **params):
    """
    Creates and trains an instance of a Q-learning controller.

    Args:
        environment: Instance of the environment on which to train the controller.

    Returns:
        QLearnAgent: Instance of the controller.
    """
    controller = QLearnAgent(**params)
    if model_path is not None and os.path.exists(model_path):
        print(f"Loading model from {model_path}...")
        controller.load_model(model_path)
    else:
        print("Training QLearnAgent from scratch...")
        controller.train(environment)
        model_path = f'{uuid4()}.pkl'
        print(f"Saving best model to {model_path}")
        controller.save_model(model_path)
    return controller


def run_game(environment_name, controller_type, model_path=None, full_hash=True, **params):
    """
    Runs the Pac-Man game with the specified environment and controller type.

    Args:
        environment_name (str): Name of the environment to use ('basic').
        controller_type (str): Type of controller to use ('random' or 'qlearn').
        full_hash (bool): Use full hashable maps for states or not (only for qlearn with basic env).
    """
    environment = create_environment(environment_name, full_hash=full_hash)
    drawer = create_drawer()

    if controller_type == 'random':
        controller = create_random_controller()
    elif controller_type == 'qlearn':
        controller = create_qlearn_controller(environment, model_path, **params)
    else:
        raise ValueError(f"Invalid controller type: {controller_type}")

    run_algorithm(environment, drawer, controller)
