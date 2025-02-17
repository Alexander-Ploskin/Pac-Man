import statistics
from typing import Any

from src.environment import PacmanEnvironment
from src.controller import Controller


def evaluate_algorithm(environment: PacmanEnvironment, controller: Controller, num_episodes=1000) -> dict[str, Any]:
    """
    Evaluates the specified controller in the given environment over a number of episodes.

    Args:
        environment (PacmanEnvironment): environment to run.
        controller_name (str): Name of the controller to use.
        num_episodes (int): Number of episodes to run for evaluation.
        **kwargs: Keyword arguments to pass to the controller.

    Returns:
        float: The average score over all episodes.
    """

    scores = []
    for _ in range(num_episodes):
        observation = environment.reset()

        while not observation.done:
            action = controller.get_action(observation)
            observation = environment.step(action)
        
        scores.append(observation.score)

    return {
        "Mean-score": statistics.mean(scores)
    }
