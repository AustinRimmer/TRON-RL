# testbed_mcts_vs_dynaq.py

import numpy as np
from Environment import Tron
from mcts_agent import MCTSAgent
from DynaQ import DynaQ

# ----------------- Config -----------------
NUM_EPISODES     = 5000
MAX_STEPS        = 10_000
RENDER_FIRST_N   = 1    # render first 3 games
RENDER_LAST_N    = 3     # and last 3 games
GRID_SIZE        = 32    # Tron(size=GRID_SIZE)
# ------------------------------------------


def make_red_obs(env: Tron):
    """
    Build an observation dict from RED's point of view for DynaQ.
    DynaQ's state2index expects:
      - observation["agent1"] = 'self' position
      - trails passed via updateTrailInfo(self_trail, opp_trail)
    For red, 'self' is env._target_location, and its trail is env.target_trail.
    """
    return {
        "agent1": np.array(env._target_location, dtype=int),
        "agent2": np.array(env._agent_location, dtype=int),
    }


def run():
    env = Tron(size=GRID_SIZE)
    env.render_mode = "human"

    # Blue = MCTS
    mcts = MCTSAgent(
        simulations_per_move=200,
        c_ucb=1.4,
        rollout_depth=50,
        gamma=1.0,
    )

    # Red = Dyna-Q
    dynaq = DynaQ(
        env_size=GRID_SIZE,
        gamma=0.95,
        step_size=0.2,
        epsilon=0.05,
        max_model_step=100,
    )

    blue_total_return = 0.0
    red_total_return = 0.0
    blue_wins = 0
    red_wins = 0
    draws = 0

    for ep in range(NUM_EPISODES):
        obs, info = env.reset()
        mcts.reset_tree()
        dynaq.resetEp()

        # Update DynaQ with *red* and *blue* trails (self, opp)
        dynaq.updateTrailInfo(env.target_trail, env.agent_trail)

        ep_blue_return = 0.0
        ep_red_return = 0.0
        done = False
        steps = 0
        last_r_blue = 0.0
        last_r_red = 0.0

        # Decide if this episode should be rendered
        render_this_ep = (
            ep < RENDER_FIRST_N
            or ep >= NUM_EPISODES - RENDER_LAST_N
        )

        if render_this_ep:
            print(f"\n=== Episode {ep+1}/{NUM_EPISODES} (render) ===")
            env.render()
        else:
            print(f"\n=== Episode {ep+1}/{NUM_EPISODES} ===")

        while not done and steps < MAX_STEPS:
            # ----- BLUE (MCTS) chooses action -----
            blue_action = mcts.act(env)

            # ----- RED (Dyna-Q) chooses action -----
            red_obs = make_red_obs(env)
            red_state_idx = dynaq.state2index(red_obs)
            red_action = dynaq.chooseAct(red_state_idx, training=True)

            # ----- Environment step -----
            obs, (r_blue, r_red), terminated, truncated, info = env.step(
                blue_action, red_action
            )
            done = terminated or truncated
            steps += 1

            # Update trails from env for DynaQ's local perception
            dynaq.updateTrailInfo(env.target_trail, env.agent_trail)

            # Next state for RED
            next_red_obs = make_red_obs(env)
            next_red_state_idx = dynaq.state2index(next_red_obs)

            # ----- Dyna-Q learning (for RED) -----
            dynaq.update(
                state_index=red_state_idx,
                action=red_action,
                reward=r_red,
                next_state_index=next_red_state_idx,
                done=done,
            )
            dynaq.planning()

            # ----- Track rewards -----
            ep_blue_return += r_blue
            ep_red_return += r_red
            last_r_blue = r_blue
            last_r_red = r_red

            if render_this_ep:
                env.render()

        # Episode finished: stats
        blue_total_return += ep_blue_return
        red_total_return += ep_red_return

        # Determine winner from final rewards
        if last_r_blue > last_r_red:
            blue_wins += 1
            outcome = "Blue (MCTS) WIN"
        elif last_r_red > last_r_blue:
            red_wins += 1
            outcome = "Red (Dyna-Q) WIN"
        else:
            draws += 1
            outcome = "DRAW"

        print(
            f"Episode {ep+1} finished in {steps} steps | "
            f"Blue Return: {ep_blue_return:+.2f} | "
            f"Red Return: {ep_red_return:+.2f} | Outcome: {outcome}"
        )

    # ------------------ Final summary ------------------
    print("\n================= SUMMARY =================")
    print(f"Episodes: {NUM_EPISODES}")
    print(f"Blue (MCTS) total return: {blue_total_return:+.2f}")
    print(f"Red  (Dyna-Q) total return: {red_total_return:+.2f}")
    print(f"Blue (MCTS) avg return/ep: {blue_total_return / NUM_EPISODES:+.2f}")
    print(f"Red  (Dyna-Q) avg return/ep: {red_total_return / NUM_EPISODES:+.2f}")
    print(f"Blue (MCTS) wins : {blue_wins}")
    print(f"Red  (Dyna-Q) wins : {red_wins}")
    print(f"Draws             : {draws}")
    print(f"Blue (MCTS) winrate: {blue_wins / NUM_EPISODES * 100:.1f}%")
    print("===========================================")


if __name__ == "__main__":
    run()
