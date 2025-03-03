import statistics
from typing import Any

from src.environment import PacmanEnvironment, BasicPacmanEnvironment
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
    steps = []
    rewards = []
    min_num_steps_on_max = None
    for _ in range(num_episodes):
        observation = environment.reset()
        
        tmp_reward = 0.0
        while not observation.done:
            action = controller.get_action(observation)
            observation = environment.step(action)
            tmp_reward += observation.reward
        
        scores.append(observation.score)
        steps.append(observation.step_count)
        rewards.append(tmp_reward)
        if isinstance(environment, BasicPacmanEnvironment) and observation.score == environment.max_score:
            if min_num_steps_on_max is None or observation.step_count < min_num_steps_on_max:
                min_num_steps_on_max = observation.step_count

    stats = {
        "Mean-score": statistics.mean(scores),
        "Mean-reward": statistics.mean(rewards),
        "Mean-steps": statistics.mean(steps)
    }

    if min_num_steps_on_max is not None:
        stats["Mean-number-of-steps-on-max"] = min_num_steps_on_max

    return stats
