
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

        self.S = env_size * env_size * 16
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
        self.agent_trail = []
        self.target_trail = []

    #this should let dyna Q actually compete with our MCTS implementation (before mine was basically blind)
    def getLocalObsrv(self,agent_pos, agent_trail, target_trail, env_size):
        x,y = agent_pos
        local_obs = 0
        #check surrounding sqrs arnd agent current pos
        for dx in [-1, 0,1]:
            for dy in [-1,0, 1]:
                nx, ny = x + dx, y + dy
                grid_pos_index = (dx + 1) * 3 + (dy + 1)

                #check if out of bounds or going into tail
                if nx < 0 or nx >= env_size or ny < 0 or ny >= env_size or  (nx, ny) in agent_trail or (nx, ny) in target_trail:
                    local_obs = local_obs | (1 << grid_pos_index)

        return local_obs

    def state2index(self,observation):

        if isinstance(observation, dict):
            agent_pos = observation["agent1"]
            agent_trail = getattr(self, 'agent_trail', [])
            target_trail = getattr(self, 'target_trail', [])
            env_size = getattr(self, 'env_size', 15)

            local_obs = self.getLocalObsrv(agent_pos, agent_trail, target_trail, env_size)
            x,y = agent_pos
            pos_index = x * self.env_size + y
            return (pos_index * 512 + local_obs) % self.S

        elif isinstance(observation, tuple) and len(observation) == 2:
            #og pos only fallback
            x, y = observation
            return (x * self.env_size + y) % self.S
        else:
            #from last vers
            return hash(str(observation)) % self.S

    def updateTrailInfo(self, agent_trail, target_trail):
        self.agent_trail = list(agent_trail)
        self.target_trail = list(target_trail)

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