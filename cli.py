import click
from src.pacman_runner import run_game

@click.group()
def cli():
    """
    Runs the environment with different controllers.
    """
    pass

@cli.command()
@click.option('--environment', type=click.Choice(['basic']), default='basic', help='The environment to run.')
@click.option('--grid_size', type=int, default=10, help='number of cells in the grid')
def random(environment, grid_size):
    """
    Runs the game with a random controller.
    """
    run_game(environment, grid_size, 'random')

@cli.command()
@click.option('--environment', type=click.Choice(['basic']), default='basic', help='The environment to run.')
@click.option('--full_hash', is_flag=True, help='Use full hashable maps for states or not.')
@click.option('--alpha', type=float, default=0.3, help='learning rate')
@click.option('--train_epsilon', type=float, default=0.9, help='train exploration rate')
@click.option('--test_epsilon', type=float, default=0.2, help='test exploration rate')
@click.option('--gamma', type=float, default=0.98, help='discount factor')
@click.option('--gamma_eps', type=float, default=0.99997, help='gamma factor for epsilon in training')
@click.option('--numTraining', type=int, default=100000, help='number of training episodes')
@click.option('--verbose', is_flag=True, help='Print Q-value, reward and position for debug on each test action')
@click.option('--model_path', type=str, default=None, help='Path to load/save the Q-learning model.')
@click.option('--grid_size', type=int, default=10, help='number of cells in the grid')
def qlearn(environment, full_hash, alpha, train_epsilon, test_epsilon, gamma, gamma_eps, numtraining, verbose, model_path, grid_size):
    """
    Runs the game with a Q-learning controller.
    """
    run_game(environment, 'qlearn', grid_size, full_hash=full_hash,
             alpha=alpha, train_epsilon=train_epsilon,
             test_epsilon=test_epsilon, gamma=gamma,
             gamma_eps=gamma_eps, numTraining=numtraining,
             verbose=verbose, model_path=model_path)

@cli.command()
@click.option('--environment', type=click.Choice(['basic']), default='basic', help='The environment to run.')
@click.option('--full_hash', is_flag=True, help='Use full hashable maps for states or not.')
@click.option('--gamma', type=float, default=0.98, help='discount factor')
@click.option('--theta', type=float, default=1e-6, help='convergence threshold')
@click.option('--numTraining', type=int, default=20, help='number of training episodes')
@click.option('--grid_size', type=int, default=10, help='number of cells in the grid')
def value_iteration(environment, full_hash, gamma, theta, numtraining, grid_size):
    """
    Runs the game with a Value Iteration controller.
    """
    run_game(
        environment,
        'value_iteration',
        grid_size,
        full_hash=full_hash,
        gamma=gamma,
        theta=theta,
        max_iterations=numtraining,
    )


if __name__ == '__main__':
    cli()
