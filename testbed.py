from Environment import Tron
import sys
import numpy as np
import pygame

# main function to run env
def main():

    pygame.init()

    env = Tron(size=64)
    env.render_mode = "human"
    obs, info = env.reset()

    running = True

    # take inputs
    key_to_action = {
        pygame.K_RIGHT: 1, #(0,0) is top left of pygame grid -> (y,x) instead of (x,y)
        pygame.K_UP: 2,
        pygame.K_LEFT: 3,
        pygame.K_DOWN: 0,
    }
    while running:
        # Handle Pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key in key_to_action:
                    action = key_to_action[event.key]
                    action2 = np.random.randint(4)
                    if action == 1:
                        actionTaken = "right"
                    elif action == 2:
                        actionTaken = "up"
                    elif action == 3:
                        actionTaken = "left"
                    elif action == 0:
                        actionTaken = "down"
                    if action2 == 1:
                            action2Taken = "right"
                    elif action2 == 2:
                        action2Taken = "up"
                    elif action2 == 3:
                        action2Taken = "left"
                    elif action2 == 0:
                        action2Taken = "down"
                    obs, (r_blue, r_red), terminated, truncated, info = env.step(action2, action)
                    print(f"Blue: Action: {actionTaken}, Reward: {r_blue}, Distance: {info['distance']}")
                    print(f"Red: Action: {action2Taken}, Reward: {r_red}, Distance: {info['distance']}")
                    if terminated:
                        print("Target reached! Resetting environment.")
                        obs, info = env.reset()

        # Render
        env.render()

    env.close()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()