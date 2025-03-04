from src.pacman_runner import run_game, print_metrics
import hydra


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg):
    """
    Unified entry point for running the game with Hydra configuration
    """
    controller = cfg.controller.type
    
    controller_args = {
        'controller_type': controller,
    }
    environment_args = {
        'environment_name': cfg.environment.type,
        'grid_size': cfg.grid_size,
    }
    drawer_args = {
        'grid_size': cfg.grid_size,
        'cell_size': cfg.cell_size,
        'framerate': cfg.framerate,
    }
    
    if controller == 'random':
        controller_args.update({})
    elif controller == 'qlearn':
        controller_args.update({
            'alpha': cfg.controller.alpha,
            'train_epsilon': cfg.controller.train_epsilon,
            'test_epsilon': cfg.controller.test_epsilon,
            'gamma': cfg.controller.gamma,
            'gamma_eps': cfg.controller.gamma_eps,
            'numTraining': cfg.controller.numTraining,
            'verbose': cfg.controller.verbose,
            'model_path': cfg.controller.model_path,
        })
    elif controller == 'value_iteration':
        controller_args.update({
            'gamma': cfg.controller.gamma,
            'theta': cfg.controller.theta,
            'max_iterations': cfg.controller.max_iterations,
            'model_path': cfg.controller.model_path,
        })
    elif controller == "reinforce":
        controller_args.update({
            'model_path': cfg.controller.model_path,
        })
    else:
        raise ValueError(f"Unknown controller: {controller}")

    environment = cfg.environment.type
    if environment == 'basic':
        environment_args.update({
            'full_hash': cfg.environment.full_hash,
        })
    elif environment == 'ghosts':
        environment_args.update({
            'num_ghosts': cfg.environment.num_ghosts,
            'stability_rate': cfg.environment.stability_rate,
            'full_hash': cfg.environment.full_hash,
        })
    else:
        raise ValueError(f"Unknown environment: {environment}")

    if cfg.eval:
        print_metrics(
            num_episodes=cfg.num_episodes,            
            environment_args=environment_args,
            controller_args=controller_args,
        )
    else:
        run_game(
            environment_args=environment_args,
            controller_args=controller_args,
            drawer_args=drawer_args,
        )

if __name__ == '__main__':
    main()
