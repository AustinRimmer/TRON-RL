import numpy as np
from Environment import Tron
def main():#test for collisions
    env = Tron(size=64) #smaller just to force cllsion faster)
    observation, info = env.reset()


    #test loop, simulates singluar episode
    terminated = False
    i=0
    while not terminated:
        #agent moves right (action 0)
        #Target moves left (action 2)

        agent_action = 0  #move right
        target_action = 2  #ove left
        if i==14:
            agent_action=1

        observation, reward, terminated, truncated, info = env.step(agent_action, target_action)
        env.render()

        #delay to visualize movement better
        import time
        time.sleep(0.5)
        i+=1
    time.sleep(5)  #keep final state for 5 seconds before closing
    env.close()

if __name__ == "__main__":
    main()
