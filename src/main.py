import pygame
from src.environment import BasicPacmanEnvironment
from src.drawer import PygameDrawer
from src.controller import BasicController


if __name__ == "__main__":
    grid_size = 10
    cell_size = 40
    max_steps = 200
    framerate = 10
    
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
