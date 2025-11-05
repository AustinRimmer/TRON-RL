import numpy as np
from scipy.linalg import block_diag
import matplotlib
from matplotlib import pyplot as plt
import gymnasium as gym
from gymnasium.error import DependencyNotInstalled
from typing import Optional

#inital setup took from this: https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation

class Tron(gym.Env):
    #define metadata dictionary for self.clock.tick(self.metadata["render_fps"]) in render()
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, size: int = 64, reward_scale: float = 1.0, step_reward: float = 0.1):

        #The size of the square grid (5x5 by default)
        self.size = size
        self.reward_scale = reward_scale
        self.step_reward = step_reward

        #Initialize positions - will be set randomly in reset()
        #Using -1,-1 as "uninitialized" state
        #self._agent_location = np.array([-1, -1], dtype=np.int32)
        #self._target_location = np.array([-1, -1], dtype=np.int32)

        #Define what the agent can observe
        #Dict space gives us structured, human-readable observations

        #will have to define this better, eventually will see other agent, most likely will need map
        self.observation_space = gym.spaces.Dict(
            {
                "agent": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),   #[x, y] coordinates
                "target": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),  #[x, y] coordinates
            }
        )

        #Define what actions are available (4 directions)
        self.action_space = gym.spaces.Discrete(4)

        #Map action numbers to actual movements on the grid
        #This makes the code more readable than using raw numbers
        self._action_to_direction = {
            0: np.array([1, 0]),   #Move right (positive x)
            1: np.array([0, 1]),   #Move up (positive y)
            2: np.array([-1, 0]),  #Move left (negative x)
            3: np.array([0, -1]),  #Move down (negative y)
        }

        #init rendering variables for PyGame
        self.window = None
        self.clock = None
        self.window_size = (512, 512)  #Fixed window size for rendering
        self.window_cell_size = self.window_size[0] // self.size  #Size of each grid cell

        self.render_mode = 'human' #add default redner mode
        #placeholder for agent positions
        self._agent_location = np.array([self.size - 1, self.size // 2], dtype=int)
        self._target_location = np.array([self.size - 1, self.size // 2], dtype=int)

    def _get_obs(self):
        """Convert internal state to observation format.

        Returns:
            dict: Observation with agent and target positions
        """
        #this means our agent can see its own and its targets location, we can just make this target the other agent and update
        return {"agent1": self._agent_location, "agent2": self._target_location}


    def _get_info(self):
        """Compute auxiliary information for debugging.

        Returns:
            dict: Info with distance between agent and target
        """
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Start a new episode.

        Args:
            seed: Random seed for reproducible episodes
            options: Additional configuration (unused in this example)

        Returns:
            tuple: (observation, info) for the initial state
        """
        #IMPORTANT: Must call this first to seed the random number generator
        super().reset(seed=seed)

        #places agent1 on right side vertically centered
        self._agent_location = np.array([self.size // 2, self.size - 1], dtype=int)

        #Randomly place target, ensuring it's different from agent position
        #will have to change these to pre determined positions later
        #editited to just be opposite agent
        self._target_location = np.array([self.size // 2, 0], dtype=int)

        # while np.array_equal(self._target_location, self._agent_location):
        #     self._target_location = self.np_random.integers(
        #         0, self.size, size=2, dtype=int
        #     )

        # initialize trails with starting positions
        self.trail_length = 1000 #trail length of 3 squares
        self.agent_trail = [tuple(self._agent_location)]
        self.target_trail = [tuple(self._target_location)]

        observation = self._get_obs()
        info = self._get_info()

        return observation, info


    #rendering stuff:

    #rendering stuff:
    def render(self):
        """Render the environment for human viewing."""
        #PyGame has a different coordinate system (flip)
        try:
            import pygame
        except ImportError as e:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gymnasium[classic-control]`"
            ) from e

        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(self.window_size)
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface(self.window_size)
        canvas.fill((0, 0, 50))

        #drawing target first time
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
               self.to_screen_xy(self._target_location),
                (self.window_cell_size, self.window_cell_size),
            )
        )

        #draw agent
        ax, ay = self._agent_location
        cx, cy = self.to_screen_xy((ax, ay))
        center = (cx + self.window_cell_size*0.5, cy + self.window_cell_size*0.5)
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            center,
            self.window_cell_size / 3,
            )

        #draw trails
        for pos in self.agent_trail:
            pygame.draw.rect(
                canvas,
                (0, 0, 255),
                pygame.Rect(
                    self.to_screen_xy(pos),
                    (self.window_cell_size, self.window_cell_size),
                    ),
            )

        for pos in self.target_trail:
            pygame.draw.rect(
                canvas,
                (255, 0, 0),
                pygame.Rect(
                    self.to_screen_xy(pos),
                    (self.window_cell_size, self.window_cell_size),
                    ),
            )

        #grid
        for i in range(self.size):
            pygame.draw.line(
                canvas,
                0,
                (0, self.window_cell_size * i),
                (self.window_size[1], self.window_cell_size * i),
                width=1,
            )
        for i in range(self.size):
            pygame.draw.line(
                canvas,
                0,
                (self.window_cell_size * i, 0),
                (self.window_cell_size * i, self.window_size[0]),
                width=1,
            )

        if self.render_mode == "human":
            #The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            #We need to ensure that human-rendering occurs at the predefined framerate.
            #The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  #rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def to_screen_xy(self, pos):
        """Convert (x, y) in the grid coords to top left origin pygame pixels"""
        x, y = int(pos[0]), int(pos[1])
        # invert y because y increases as you go down
        ys = (self.size - 1 - y)
        return (x * self.window_cell_size, ys * self.window_cell_size)

    #agent behaviour stuff:
    def step(self, agent_action, target_action):
        """Execute one timestep within the environment.

        Args:
            action: The action to take (0-3 for directions)

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        #initialize rewards to step reward
        r_blue = self.step_reward
        r_red = self.step_reward

        #Map the discrete action (0-3) to a movement direction fro moth agent and target(other agent)
        agent_direction = self._action_to_direction[agent_action]
        target_direction = self._action_to_direction[target_action]

        #update agent position, making sure it stays in bounds
        #np.clip prevents the agent from walking off the edge
        agent_pos = np.clip(self._agent_location + agent_direction, 0, self.size - 1)
        target_pos = np.clip(self._target_location + target_direction, 0, self.size - 1)

        new_agent_pos = tuple(agent_pos)
        new_target_pos = tuple(target_pos)
        #terminate if the agent has caught the target
        terminated = np.array_equal(self._agent_location, self._target_location)

        loss = False

        #terminate if there is a collision with enemy/self trail
        if (new_agent_pos in self.target_trail or new_target_pos in self.agent_trail or #touching a trail
                new_agent_pos in self.agent_trail or new_target_pos in self.target_trail): #touching own trail
            loss = True
            terminated = True

        if not terminated:
            #update positions for both agents
            self._agent_location = agent_pos
            self._target_location = target_pos

            #update trail positions
            self.agent_trail.append(tuple(self._agent_location))
            self.target_trail.append(tuple(self._target_location))

            if len(self.agent_trail) > self.trail_length:
                self.agent_trail.pop(0)
            if len(self.target_trail) > self.trail_length:
                self.target_trail.pop(0)

        #reward structure, super simple atm
        #have to call init for it to work
        if terminated: #wil investigate further differentiating betweene loses and wins
            blue_hit = new_agent_pos in self.target_trail or new_agent_pos in self.agent_trail
            red_hit = new_target_pos in self.agent_trail or new_target_pos in self.target_trail

            if blue_hit and red_hit:
                r_blue = r_red = 0
            elif blue_hit:
                r_blue, r_red = -20, 20
            elif red_hit:
                r_blue, r_red = 20, -20
            else:
                r_blue = r_red = self.step_reward
        truncated = False #not used in simple environment

        observation = self._get_obs()
        info = self._get_info()
        return observation, (r_blue, r_red), terminated, truncated, info


    #close
    def close(self):
        if self.window is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
