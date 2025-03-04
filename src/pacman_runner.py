import pygame
import os
import json
from uuid import uuid4
from src.environment import BasicPacmanEnvironment, GhostsPacmanEnvironment
from src.drawer import PygameDrawer
from src.controller import BasicController, QLearnAgent, ValueIterationAgent, NeuralNetworkPolicy
from src.evaluation import evaluate_algorithm


def run_algorithm(environment, drawer, controller):
    """
    Execute the main game loop with the given environment, drawer, and controller.

    Args:
        environment: The Pac-Man environment instance.
        drawer: The Pygame drawer instance for rendering.
        controller: The game controller instance.
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
    Instantiate and return the specified Pac-Man environment.

    Args:
        environment_name (str): The type of environment to create ('basic' or 'ghosts').
        **params: Additional parameters for environment initialization.

    Returns:
        An instance of BasicPacmanEnvironment or GhostsPacmanEnvironment.

    Raises:
        ValueError: If an invalid environment name is provided.
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
    Create and return a Pygame drawer instance for rendering the game.

    Args:
        grid_size (int): The number of cells in each dimension of the grid.
        cell_size (int): The size of each cell in pixels.
        framerate (int): The target frame rate for the game.

    Returns:
        PygameDrawer: An instance of the Pygame drawer.
    """
    return PygameDrawer(
        grid_size=grid_size,
        cell_size=cell_size,
        framerate=framerate
    )


def create_random_controller():
    """
    Create and return a random action controller.

    Returns:
        BasicController: An instance of the random controller.
    """
    return BasicController()


def create_qlearn_controller(environment, model_path, **params):
    """
    Create and optionally train a Q-learning controller.

    Args:
        environment: The environment instance for training.
        model_path (str): Path to load a pre-trained model (if it exists).
        **params: Additional parameters for Q-learning agent initialization.

    Returns:
        QLearnAgent: An instance of the Q-learning controller.
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
    Create and optionally train a Value Iteration controller.

    Args:
        environment: The environment instance for training.
        model_path (str): Path to load a pre-trained model (if it exists).
        **params: Additional parameters for Value Iteration agent initialization.

    Returns:
        ValueIterationAgent: An instance of the Value Iteration controller.
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
    """
    Save the trained model and its parameters to a unique directory.

    Args:
        controller: The trained controller instance.
        params (dict): The parameters used for training.
        method (str): The training method ('qlearn' or 'value_iteration').
    """
    checkpoint_folder = os.path.join('checkpoints', method, str(uuid4()))
    os.mkdir(checkpoint_folder)
    model_path = os.path.join(checkpoint_folder, 'checkpoint.pkl')
    print(f"Saving best model to {model_path}")
    controller.save_model(model_path)
    with open(os.path.join(checkpoint_folder, 'params'), 'w+') as file:
        json.dump(params, file)


def create_controller(environment, controller_type, **params):
    """
    Create and return the specified type of game controller.

    Args:
        environment: The environment instance for potential training.
        controller_type (str): The type of controller to create ('random', 'qlearn', or 'value_iteration').
        **params: Additional parameters for controller initialization.

    Returns:
        A controller instance of the specified type.

    Raises:
        ValueError: If an invalid controller type is provided.
    """
    if controller_type == 'random':
        return create_random_controller(**params)
    elif controller_type == 'qlearn':
        return create_qlearn_controller(environment, **params)
    elif controller_type == 'value_iteration':
        return create_value_iteration_controller(environment, **params)
    elif controller_type == "reinforce":
        return NeuralNetworkPolicy()
    else:
        raise ValueError(f"Invalid controller type: {controller_type}")


def run_game(environment_args, controller_args, drawer_args):
    """
    Set up and run the Pac-Man game with specified components.

    Args:
        environment_args (dict): Arguments for creating the game environment.
        controller_args (dict): Arguments for creating the game controller.
        drawer_args (dict): Arguments for creating the game renderer.
    """
    environment = create_environment(**environment_args)
    drawer = create_drawer(**drawer_args)
    controller = create_controller(environment, **controller_args)

    run_algorithm(environment, drawer, controller)


def print_metrics(num_episodes, environment_args, controller_args):
    """
    Evaluate a controller on an environment and print performance metrics.

    Args:
        num_episodes (int): The number of episodes to run for evaluation.
        environment_args (dict): Arguments for creating the game environment.
        controller_args (dict): Arguments for creating the game controller.
    """
    environment = create_environment(**environment_args)
    controller = create_controller(environment, **controller_args)
    
    metrics = evaluate_algorithm(environment, controller, num_episodes)
    print(f"Environment: {environment_args['environment_name']}", f"Controller: {controller_args['controller_type']}", f"Metrics: {metrics}")
