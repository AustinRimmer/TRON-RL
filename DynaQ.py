
"""
OLD A2 DYNA-Q:
FROM AUSTINS A2


import numpy as np
def DynaQ(env,gamma, step_size, epsilon, max_episode, max_model_step):
    #this was such a pain to make run in under 1 minute. inital implementation ran in 216 secs lol
    #init Q(s,a) all s in S, all a in A Q terminal = 0
    S = env.n_states
    A = env.n_actions
    rng = np.random.default_rng()
    q = rng.uniform(low=1e-2, high=1e-2, size=(S, A))

    model = {}
    observedStateActions = []
    #needed to go fast
    seen = set()

    #Loop forever (for each episode)
    for episode in range(max_episode):
        #init S
        state, _ = env.reset()
        finished = False
        #loop for each step of episode
        while not finished:
            #choose A from S using policy derived from Q
            if rng.random() < epsilon:
                action = rng.integers(0, A)
            else:
                bestActions = np.flatnonzero(q[state] == np.max(q[state]))
                action = int(rng.choice(bestActions))

            #take action A, observe R, S'
            nextS,reward,terminated,truncated, _ = env.step(action)
            finished = terminated or truncated

            #target = R + gamma max Q(S',a')
            bestQ = np.max(q[nextS])
            target = reward + gamma * bestQ

            #Q(S,A) = Q(S,A) + a[R + gamma max Q(S',a) - Q(S,A)]
            if finished:
                q[state,action] += step_size *(reward - q[state, action])
            else:
                q[state,action] += step_size * (target - q[state, action])

            if(state,action) not in seen:
                #for speed only
                seen.add((state,action))
                observedStateActions.append((state,action))

            model[(state, action)] = (nextS, reward, finished)

            #Planning
            #this is where i had to make some changes
            if observedStateActions and max_model_step > 0:
                observedArray = np.array(observedStateActions)
                #sample prev obsrvd states
                indexs = rng.choice(len(observedArray), size=min(max_model_step, len(observedArray)), replace=False)
                for index in indexs:
                    sP, aP = observedArray[index]
                    nextSP,rP, finishedP = model[(sP,aP)]

                    #target = R + gamma * max Q(S',a')
                    bestQP = np.max(q[nextSP])
                    target = rP + gamma * bestQP
                    #Q(S,A) = Q(S,A) + a[R + gamma max Q(S',a) - Q(S,A)]
                    if finishedP:
                        q[sP,aP] += step_size * (rP - q[sP, aP])
                    else:
                        q[sP,aP] += step_size * (target - q[sP,aP])
            #S = S'
            state = nextS

    #getting greedy from Q
    greedyAs = np.argmax(q,axis=1)
    Pi = np.zeros((S,S*A))
    for s in range(S):
        Pi[s,s * A + greedyAs[s]] = 1.0

    qCol = q.reshape((S*A,1))

    return Pi, qCol
"""
#TODO: make helper functs like in A2 (that sucks)
#current Dyna-Q attempt No.13

# a more understandable dumbed down explination of Dyna-Q: https://compneuro.neuromatch.io/tutorials/W3D4_ReinforcementLearning/student/W3D4_Tutorial4.html
# this helped me out a bit on A2 aswell but was super usefull here with the gross implementation
import numpy as np


class DynaQ:
    def __init__ (self, env_size, gamma=0.95, step_size=0.1, epsilon=0.1, max_model_step=5):
        """
        This is for the Dyna-Q agent
        :param env_size: size of grid env
        :param gamma: discount (duh)
        :param step_size: learn r8
        :param epsilon: explore r8
        :param max_model_step: num planning steps
        """
        self.env_size = env_size
        self.gamma = gamma
        self.step_size = step_size
        self.epsilon = epsilon
        self.max_model_step = max_model_step

        self.S = env_size * env_size
        self.A = 4

        self.q = np.zeros((self.S, self.A))

        self.model = {}

        self.visited_sa = []

        #important so agent doesnt do illegal move
        self.prev_act = None

        #dict of opposing actions so agent doesnt 180 and kill itself every 2 secs
        self.op_acts = {
            0: 2, #R -> L
            1: 3, #Up -> Dwn
            2: 0, #L -> R
            3: 1  #Dwn -> Up
        }

        self.rng = np.random.default_rng()

    #LLM assisted with debugging this stupid dumb sillybilly error bs i was having with hashsize conversion overflow
    def state2index(self,grid_obs):

        if isinstance(grid_obs, tuple) and len(grid_obs) == 2:
            #if pos only
            x, y = grid_obs
            return x * self.env_size + y

        #compress
        grid_flat = grid_obs.flatten()

        #convert grid to a base 3 num
        state_index = 0
        for i, cell in enumerate(grid_flat):
            state_index = state_index * 3 + (cell % 3)  # Use modulo to ensure small values

        return state_index % self.S

    def getValidAct(self, state_index):
        acts = [0,1,2,3]

        if self.prev_act is not None:
            op_act = self.op_acts[self.prev_act]
            valid_acts = [action for action in acts if action != op_act]
            return valid_acts
        return acts

    def chooseAct(self,state_index, training=True):
        valid_acts = self.getValidAct(state_index)

        if training and self.rng.random() < self.epsilon:
            action = int(self.rng.choice(valid_acts))
        else:
            valid_q_vals = [self.q[state_index, a] for a in valid_acts]
            max_q = np.max(valid_q_vals)

            best_acts = [a for a in valid_acts if self.q[state_index, a] == max_q]
            action = int(self.rng.choice(best_acts))

        self.prev_act = action
        return action

    def update(self, state_index, action, reward, next_state_index, done):
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q[next_state_index])


        #Q learning update
        self.q[state_index,action] += self.step_size*(target - self.q[state_index,action])

        #mode l update
        self.model[(state_index, action)] = (next_state_index, reward, done)

        #tracking visited
        if (state_index, action) not in self.visited_sa:
            self.visited_sa.append((state_index,action))

    def planning(self):
        if len(self.visited_sa) == 0 or self.max_model_step <= 0:
            return

        for _ in range (self.max_model_step):
            state_index,action = self.rng.choice(self.visited_sa)

            if(state_index,action) in self.model:
                next_state_index,reward, done = self.model[(state_index,action)]

                if done:
                    target = reward
                else:
                    target = reward + self.gamma * np.max(self.q[next_state_index])

                #updating q val based on plan
                self.q[state_index,action] += self.step_size * (target - self.q[state_index,action])

    def resetEp(self):
        self.prev_act = None

