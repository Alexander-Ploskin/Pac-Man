import pygame
import os
import json
from uuid import uuid4
from src.environment import BasicPacmanEnvironment, GhostsPacmanEnvironment
from src.drawer import PygameDrawer
from src.controller import BasicController, QLearnAgent, ValueIterationAgent
from src.evaluation import evaluate_algorithm


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
        drawer.draw(observation.map, action)

        # Reset the environment if the episode has ended.
        if observation.done:
            print("Episode finished. Resetting environment.")
            environment.reset()

    # Clean up Pygame resources.
    drawer.close()


def create_environment(environment_name, **params):
    """
    Creates and returns an instance of the specified environment.

    Args:
        environment_name (str): Name of the environment to create ('basic').

    Returns:
        An environment instance.
    """
    if environment_name == 'basic':
        environment = BasicPacmanEnvironment(**params)
    elif environment_name == 'ghosts':
        environment = GhostsPacmanEnvironment(**params)
    else:
        raise ValueError(f"Invalid environment name: {environment_name}")

    return environment


def create_drawer(grid_size=10, cell_size=40, framerate=3):
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
        save_model(controller, params, 'qlearn')
    return controller


def create_value_iteration_controller(environment, model_path, **params):
    """
    Creates and trains an instance of a Value Iteration controller.

    Args:
        environment: Instance of the environment on which to train the controller.

    Returns:
        ValueIterationAgent: Instance of the controller.
    """
    controller = ValueIterationAgent(environment=environment, **params)
    if model_path is not None and os.path.exists(model_path):
        print(f"Loading model from {model_path}...")
        controller.load_model(model_path)
    else:
        print("Training ValueIterationAgent from scratch...")
        controller.train(environment)
        save_model(controller, params, 'value_iteration')
    return controller

def save_model(controller, params, method):
    checkpoint_folder = os.path.join('checkpoints', method, uuid4())
    os.mkdir(checkpoint_folder)
    model_path = os.path.join(checkpoint_folder, 'checkpoint.pkl')
    print(f"Saving best model to {model_path}")
    controller.save_model(model_path)
    with open(os.path.join(checkpoint_folder, 'params'), 'w+') as file:
        json.dump(params, file)


def create_controller(environment, controller_type, **params):
    if controller_type == 'random':
        return create_random_controller(**params)
    elif controller_type == 'qlearn':
        return create_qlearn_controller(environment, **params)
    elif controller_type == 'value_iteration':
        return create_value_iteration_controller(environment, **params)
    else:
        raise ValueError(f"Invalid controller type: {controller_type}")


def run_game(environment_args, controller_args, drawer_args):
    """
    Runs the Pac-Man game with the specified environment and controller type.

    Args:
        environment_name (str): Name of the environment to use ('basic').
        controller_type (str): Type of controller to use ('random', 'qlearn' or 'value_iteration').
        full_hash (bool): Use full hashable maps for states or not (only for qlearn with basic env).
    """
    environment = create_environment(**environment_args)
    drawer = create_drawer(**drawer_args)
    controller = create_controller(environment, **controller_args)

    run_algorithm(environment, drawer, controller)


def print_metrics(num_episodes, environment_args, controller_args):
    """
    Evaluate controller on environment with num_episodes runs.
    """
    environment = create_environment(**environment_args)
    controller = create_controller(environment, **controller_args)
    
    metrics = evaluate_algorithm(environment, controller, num_episodes)
    print(f"Environment: {environment_args['environment_name']}", f"Controller: {controller_args['controller_type']}", f"Metrics: {metrics}")
