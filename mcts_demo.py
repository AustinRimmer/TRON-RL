# mcts_demo.py
from Environment import Tron
from mcts_agent import MCTSAgent
import random
import pygame
import time

# ==================== TOGGLE HERE ====================
STEP_MODE = False   # True = press SPACE to advance step-by-step
# False = run automatically
# =====================================================

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

def main():
    env = Tron(size=15)
    env.render_mode = "human"
    agent = MCTSAgent(
        simulations_per_move=200,
        c_ucb=1.4,
        rollout_depth=50,
        gamma=1.0
    )

    obs, info = env.reset()
    agent.reset_tree()
    total_reward = 0.0
    step = 0

    dir_map = {0: "Right", 1: "Up", 2: "Left", 3: "Down"}

    print("\n=== MCTS Demo ===")
    print("Press [SPACE] to advance a step (if STEP_MODE=True). Press [ESC] to quit.\n")

    env.render()
    pygame.init()

    while True:
        if STEP_MODE:
            wait_for_space_or_quit()
        else:
            time.sleep(0.05)  # small delay for smoother rendering

        # our move via MCTS
        our_action = agent.act(env)
        # opponent (safe random can be used here)
        opp_action = random.randint(0, 3)

        # take one environment step
        obs, (r_our, r_opp), terminated, truncated, info = env.step(our_action, opp_action)
        total_reward += r_our

        env.render()

        # get current positions for printing
        blue_pos = tuple(env._agent_location.tolist())
        red_pos = tuple(env._target_location.tolist())

        print(f"Step {step:03d} | Blue: {dir_map[our_action]:<5} | Red: {dir_map[opp_action]:<5} "
              f"| Reward (Blue,Red): ({r_our:+5.1f}, {r_opp:+5.1f}) "
              f"| Blue Pos: {blue_pos} | Red Pos: {red_pos} "
              f"| Total Blue Reward: {total_reward:+6.1f}")

        step += 1

        if terminated or truncated:
            print("\n--- Episode Ended ---")
            print(f"Final Blue Reward: {total_reward:+.2f}\n")
            total_reward = 0.0
            step = 0
            obs, info = env.reset()
            agent.reset_tree()

if __name__ == "__main__":
    main()
