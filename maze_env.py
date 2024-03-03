import numpy as np
import gym
from gym import spaces
import matplotlib.pyplot as plt
from typing import Tuple, Optional

# from typing import (
#     TYPE_CHECKING,
#     Any,
#     Dict,
#     Generic,
#     List,
#     Optional,
#     SupportsFloat,
#     Tuple,
#     TypeVar,
#     Union,
# )


class MazeEnv(gym.Env):
    """Custom Environment that follows gym interface"""

    metadata = {"render.modes": ["human", "rgb_array"]}
    reward_range = (-float("inf"), float("inf"))

    def __init__(self):
        super(MazeEnv, self).__init__()
        self.action_space = spaces.Discrete(4)  # 0: up, 1: right, 2: down, 3: left
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(9, 9), dtype=np.float32
        )

        self.maze = np.zeros((9, 9))
        self.maze[1, 1:8] = 1  # Example wall
        self.maze[3, 1:8] = 1  # Example wall
        self.maze[5, 1:8] = 1  # Example wall
        self.maze[7, 1:8] = 1  # Example wall
        self.agent_pos = [0, 0]  # Start at top-left corner

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        assert self.action_space.contains(action), "%r (%s) invalid" % (
            action,
            type(action),
        )
        y, x = self.agent_pos
        if action == 0:  # up
            y = max(0, y - 1)
        elif action == 1:  # right
            x = min(x + 1, 8)
        elif action == 2:  # down
            y = min(y + 1, 8)
        elif action == 3:  # left
            x = max(0, x - 1)

        if self.maze[y, x] == 0:  # If not a wall
            self.agent_pos = [y, x]

        terminated = self.agent_pos == [8, 8]  # Check for goal
        truncated = (
            False  # In this simple example, we don't have a condition for truncation
        )
        reward = 1 if terminated else -0.01  # Reward for reaching the goal or moving
        info = {}

        return np.array(self.agent_pos), reward, terminated, truncated, info

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.agent_pos = [0, 0]
        return np.array(
            self.agent_pos
        ), {}  # Return initial observation and an empty info dict

    def render(self, mode="human"):
        if mode == "human":
            self._render_human()
        elif mode == "rgb_array":
            return self._render_rgb_array()

    def _render_human(self):
        maze_copy = np.copy(self.maze)
        maze_copy[self.agent_pos[0], self.agent_pos[1]] = 2  # Mark agent's position
        plt.imshow(maze_copy, cmap="viridis")
        plt.show()

    def _render_rgb_array(self):
        # This method can be implemented if needed for returning an image array
        pass

    def close(self):
        plt.close()
