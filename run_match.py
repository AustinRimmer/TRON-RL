import time
from Environment import Tron
from mcts_agent import MCTSAgent
from DynaQ import DynaQ

def run_match(
        episodes=50,
        render=True,
        sleep_time=0.05,
):
    env = Tron()


    blue_agent = MCTSAgent(simulations_per_move=200,
                           c_ucb=1.4,
                           rollout_depth=50,
                           gamma=1.0)   # Blue
    red_agent  = DynaQ(env_size=env.size,
                       gamma=0.95,
                       step_size=0.1,
                       epsilon=0.3,
                       max_model_step=10)   # Red

    results = {"blue_wins": 0, "red_wins": 0, "ties": 0}

    for ep in range(episodes):
        obs, _ = env.reset()
        terminated = False
        ep_reward_blue = 0
        ep_reward_red = 0

        while not terminated:
            blue_obs, red_obs = obs

            #both agents choose action

            blue_action = blue_agent.act(env)

            state_idx = red_agent.state2index(obs["agent2"])
            red_action = red_agent.chooseAct(state_idx, training=True)

            #step
            obs, (r_blue, r_red), terminated, truncated, info = env.step(blue_action, red_action)

            ep_reward_blue += r_blue
            ep_reward_red += r_red

            # Dyna-Q update (MCTS doesn't learn here)
            next_state_idx = red_agent.state2index(obs["agent1"])
            red_agent.update(state_idx, red_action, ep_reward_red, next_state_idx, terminated)

            if render:
                env.render()      # ← your pygame renderer
                time.sleep(sleep_time)

        # Episode finished — count result
        if ep_reward_blue > ep_reward_red:
            results["blue_wins"] += 1
            print(f"Episode {ep+1}: BLUE wins")
        elif ep_reward_red > ep_reward_blue:
            results["red_wins"] += 1
            print(f"Episode {ep+1}: RED wins")
        else:
            results["ties"] += 1
            print(f"Episode {ep+1}: TIE")

    print("\n=== FINAL RESULTS ===")
    print(results)
    return results


if __name__ == "__main__":
    run_match(episodes=50, render=True)