import numpy as np
from scipy.linalg import block_diag
import matplotlib
from matplotlib import pyplot as plt
import gymnasium as gym
from gymnasium.error import DependencyNotInstalled
from typing import Optional

class GridWorldEnv(gym.Env):
    # define metadata dictionary for self.clock.tick(self.metadata["render_fps"]) in render()
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, size: int = 64, reward_scale: float = 1.0, step_reward: float = 0.1):
        # The size of the square grid (5x5 by default)
        self.size = size
        self.reward_scale = reward_scale
        self.step_reward = step_reward
        # Initialize positions - will be set randomly in reset()
        # Using -1,-1 as "uninitialized" state
        #self._agent_location = np.array([-1, -1], dtype=np.int32)
        #self._target_location = np.array([-1, -1], dtype=np.int32)

        # Define what the agent can observe
        # Dict space gives us structured, human-readable observations

        #will have to define this better, eventually will see other agent, most likely will need map
        self.observation_space = gym.spaces.Dict(
            {
                "agent": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),   # [x, y] coordinates
                "target": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),  # [x, y] coordinates
            }
        )

        # Define what actions are available (4 directions)
        self.action_space = gym.spaces.Discrete(4)

        # Map action numbers to actual movements on the grid
        # This makes the code more readable than using raw numbers
        self._action_to_direction = {
            0: np.array([1, 0]),   # Move right (positive x)
            1: np.array([0, 1]),   # Move up (positive y)
            2: np.array([-1, 0]),  # Move left (negative x)
            3: np.array([0, -1]),  # Move down (negative y)
        }

        #init rendering variables for PyGame
        self.window = None
        self.clock = None
        self.window_size = (512, 512)  # Fixed window size for rendering
        self.window_cell_size = self.window_size[0] // self.size  # Size of each grid cell

    def _get_obs(self):
        """Convert internal state to observation format.

        Returns:
            dict: Observation with agent and target positions
        """
        #this means our agent can see its own and its targets location, we can just make this target the other agent and update
        return {"agent": self._agent_location, "target": self._target_location}


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
        # IMPORTANT: Must call this first to seed the random number generator
        super().reset(seed=seed)

        #places agent1 on right side vertically centered
        self._agent_location = np.array([self.size - 1, self.size // 2], dtype=int)

        # Randomly place target, ensuring it's different from agent position
        #will have to change these to pre determined positions later
        self._target_location = self._agent_location

        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(
                0, self.size, size=2, dtype=int
            )

        observation = self._get_obs()
        info = self._get_info()

        return observation, info


    #rendering stuff:

    #rendering stuff:
    def render(self):
        """Render the environment for human viewing."""
        # PyGame has a different coordinate system (flip)
        try:
            import pygame
        except ImportError as e:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gymnasium[classic-control]`"
            ) from e

        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(np.flip(self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface(np.flip(self.window_size))
        canvas.fill((255, 255, 255))

        #drawing target first time
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                self.window_cell_size * np.flip(self._target_location),
                (self.window_cell_size, self.window_cell_size),
                ),
        )

        #draw agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (np.flip(self._agent_location) + 0.5) * self.window_cell_size,
            self.window_cell_size / 3,
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
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
    #agent behaviour stuff:

    def step(self, action):
        """Execute one timestep within the environment.

        Args:
            action: The action to take (0-3 for directions)

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # Map the discrete action (0-3) to a movement direction
        direction = self._action_to_direction[action]

        #update agent position, making sure it stays in bounds
        # np.clip prevents the agent from walking off the edge
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )

        #checking if agent reached the target
        #we will have to change this pretty significantly
        terminated = np.array_equal(self._agent_location, self._target_location)

        # We don't use truncation in this simple environment
        # (could add a step limit here if desired)
        truncated = False

        #reward structure, super simple atm
        #have to call init for it to work
        if terminated:
            reward = self.reward_scale  # Success reward
        else:
            reward = -self.step_reward  # Step penalty (0 by default)

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    #close
    def close(self):
        if self.window is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
