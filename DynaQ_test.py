from Environment import Tron
from DynaQ import DynaQ
import sys
import numpy as np
import pygame
import time
import random

#taken from MCTS agent code
def wait_for_space_or_quit():
    """Pause until SPACE is pressed or window is closed."""
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    return
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    raise SystemExit
        time.sleep(0.05)
# also taken from MCTS
def safe_random_opponent_action(env):
    """
    Chooses a random action for the opponent that avoids immediate death.
    Avoids stepping into its own trail, the agent's trail, or staying in place due to wall clipping.
    """
    safe_actions = []
    tx, ty = env._target_location
    size = env.size
    trails = set(env.agent_trail) | set(env.target_trail)

    for a in range(4):
        dx, dy = env._action_to_direction[a]
        nx = max(0, min(size - 1, tx + dx))
        ny = max(0, min(size - 1, ty + dy))
        new_pos = (nx, ny)

        # Avoid staying in place due to wall clipping
        if new_pos == (tx, ty):
            continue
        # Avoid stepping into any trail
        if new_pos in trails:
            continue

        safe_actions.append(a)

    # If boxed in, return any action to keep simulation moving
    return random.choice(safe_actions if safe_actions else [0, 1, 2, 3])

def main():
    env = Tron(size=16)
    env.render_mode = "human"

    #cr8s 2 dynaQ bbs
    dyna_q_agent1 = DynaQ(
        env_size=env.size,
        gamma=0.9,
        step_size=0.3,
        epsilon=0.05,
        max_model_step=150
    )

    obs, info = env.reset()
    dyna_q_agent1.resetEp()
    total_reward = 0.0
    step = 0

    dir_map = {0: "Right", 1: "Up", 2: "Left", 3: "Down"}

    env.render()
    pygame.init()

    while True:
        time.sleep(0.01)

        dyna_q_agent1.updateTrailInfo(env.agent_trail, env.target_trail)

        state_index = dyna_q_agent1.state2index(obs["agent1"])
        our_action = dyna_q_agent1.chooseAct(state_index, training=True)

        opp_action = safe_random_opponent_action(env)

        obs, (r_our, r_opp),terminated, truncated, info = env.step( our_action, opp_action)
        total_reward += r_our

        dyna_q_agent1.updateTrailInfo(env.agent_trail, env.target_trail)

        #update dynaQ agent
        next_state_index = dyna_q_agent1.state2index(obs["agent1"])

        dyna_q_agent1.update(state_index, our_action, r_our, next_state_index, terminated)
        dyna_q_agent1.planning()

        env.render()

        blue_pos = tuple(env._agent_location.tolist())
        red_pos = tuple(env._target_location.tolist())

        """        print(f"Step {step:03d} | Blue: {dir_map[our_action]:<5} | Red: {dir_map[opp_action]:<5} "
          f"| Reward (Blue,Red): ({r_our:+5.1f}, {r_opp:+5.1f}) "
          f"| Blue Pos: {blue_pos} | Red Pos: {red_pos} "
          f"| Total Blue Reward: {total_reward:+6.1f}")"""

        step += 1

        if terminated or truncated:
            print("\n--- Episode Ended ---")
            print(f"Final Blue Reward: {total_reward:+.2f}\n")
            total_reward = 0.0
            step = 0
            obs, info = env.reset()
            dyna_q_agent1.resetEp()

if __name__ == "__main__":
    main()