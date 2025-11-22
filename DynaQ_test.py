from Environment import Tron
from DynaQ import DynaQ
import sys
import numpy as np
import pygame

def main():
    pygame.init()

    env = Tron(size=32)
    env.render_mode = "human"

    #cr8s 2 dynaQ bbs
    dyna_q_agent1 = DynaQ(
        env_size=env.size,
        gamma=0.95,
        step_size=0.1,
        epsilon=0.3,
        max_model_step=10
    )
    #this boys a lil funky wit it
    dyna_q_agent2 = DynaQ(
        env_size=env.size,
        gamma=0.95,
        step_size=0.4,
        epsilon=0.5,
        max_model_step=20
    )


    episodes = 1000

    print("Training Dyna-Q agents...")


    for episode in range(episodes):
        obs, info = env.reset()
        state_idx1 = dyna_q_agent1.state2index(obs["agent1"])
        state_idx2 = dyna_q_agent2.state2index(obs["agent2"])

        dyna_q_agent1.resetEp()
        dyna_q_agent2.resetEp()

        terminated = False
        truncated = False
        total_reward1 = 0
        total_reward2 = 0
        steps = 0

        while not (terminated or truncated) and steps < 500:
            #each agents selected action
            action1 = dyna_q_agent1.chooseAct(state_idx1, training=True)
            action2 = dyna_q_agent2.chooseAct(state_idx2, training=True)


            action_directions = {0: "down", 1: "right", 2: "up", 3: "left"}
            action1_taken = action_directions[action1]
            action2_taken = action_directions[action2]

            #step
            obs, (reward1, reward2), terminated, truncated, info = env.step(action1, action2)

            #next state indices
            next_state_idx1 = dyna_q_agent1.state2index(obs["agent1"])
            next_state_idx2 = dyna_q_agent2.state2index(obs["agent2"])

            #prints
            print(f"Blue: Action: {action1_taken}, Reward: {reward1:.1f}, Distance: {info.get('distance', 'N/A')}")
            print(f"Red: Action: {action2_taken}, Reward: {reward2:.1f}, Distance: {info.get('distance', 'N/A')}")

            #updates
            dyna_q_agent1.update(state_idx1, action1, reward1, next_state_idx1, terminated)
            dyna_q_agent2.update(state_idx2, action2, reward2, next_state_idx2, terminated)

            #let em plan
            dyna_q_agent1.planning()
            dyna_q_agent2.planning()

            #render only sometimes (lol)
            if episode % 50 == 0:
                env.render()
                pygame.time.delay(50)

            total_reward1 += reward1
            total_reward2 += reward2
            state_idx1 = next_state_idx1
            state_idx2 = next_state_idx2
            steps += 1

            if terminated:
                print("Game over! Resetting environment.")
                print(f"Episode {episode} completed in {steps} steps")
                print(f"Total Rewards - Blue: {total_reward1:.1f}, Red: {total_reward2:.1f}")
                break

        #print summary
        if episode % 10 == 0:
            print(f"=== Episode {episode} Summary ===")
            print(f"Steps: {steps}, Total Blue Reward: {total_reward1:.1f}, Total Red Reward: {total_reward2:.1f}")
            print("=" * 40)

    print("Training completed!")

    env.close()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()