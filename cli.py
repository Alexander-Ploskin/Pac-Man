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
def random(environment):
    """
    Runs the game with a random controller.
    """
    run_game(environment, 'random')

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
def qlearn(environment, full_hash, alpha, train_epsilon, test_epsilon, gamma, gamma_eps, numtraining, verbose):
    """
    Runs the game with a Q-learning controller.
    """
    run_game(environment, 'qlearn', full_hash=full_hash,
             alpha=alpha, train_epsilon=train_epsilon,
             test_epsilon=test_epsilon, gamma=gamma,
             gamma_eps=gamma_eps, numTraining=numtraining,
             verbose=verbose)


if __name__ == '__main__':
    cli()
