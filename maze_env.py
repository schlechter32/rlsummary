import numpy as np
import gym
from gym import spaces
import matplotlib.pyplot as plt
import torch
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
        # 0: up, 1: right, 2: down, 3: left
        self.action_space = spaces.Discrete(4)
        # self.observation_space = spaces.Box(
        #     low=0, high=1, shape=(9, 9, 2), dtype=np.float32
        # )

        self.step_count = 0
        self.maze = np.zeros((9, 9))
        self.maze[1, 1:8] = 1  # Example wall
        self.maze[3, 1:8] = 1  # Example wall
        self.maze[5, 1:8] = 1  # Example wall
        self.maze[7, 1:8] = 1  # Example wall

        # self.observation_space[0][1, 1:8] = 1  # Example wall
        # self.observation_space[0][3, 1:8] = 1  # Example wall
        # self.observation_space[0][5, 1:8] = 1  # Example wall
        # self.observation_space[0][7, 1:8] = 1  # Example wall
        self.agent_pos = np.zeros(2)  # Start at top-left corner
        self.observation_space = self.maze
        # self.observation_space[0, :, :] = self.maze
        print(self.agent_pos)
        print(self.observation_space)
        self.observation_space = np.concatenate(
            (self.observation_space.flatten(), self.agent_pos)
        )
        print(self.observation_space)
        self.update_observation_from_state()

    def update_observation_from_state(self):
        self.observation_space[81] = self.agent_pos[0]

        self.observation_space[82] = self.agent_pos[1]
        self.observation_space_tensor = torch.tensor(
            self.observation_space, dtype=torch.float32
        )

        return self.observation_space_tensor

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        assert self.action_space.contains(action), "%r (%s) invalid" % (
            action,
            type(action),
        )
        y, x = self.agent_pos
        if action == 0:  # up
            y = max(0, y - 1)
            y = y - 1 if y - 1 > 0 else 8
        elif action == 1:  # right
            x = min(x + 1, 8)
        elif action == 2:  # down
            y = min(y + 1, 8)
        elif action == 3:  # left
            x = max(0, x - 1)

        if self.maze[y, x] == 0:  # If not a wall
            self.agent_pos = [y, x]

        self.update_observation_from_state()
        terminated = (
            self.agent_pos == [8, 8] or self.maze[y, x] == 1 or self.step_count > 3000
        )

        # terminated = self.agent_pos == [8, 8] or self.step_count > 1000
        self.step_count += 1
        # terminated = True
        # terminated = self.agent_pos == [8, 8]  # Check for goal
        truncated = (
            False  # In this simple example, we don't have a condition for truncation
        )
        # Reward for reaching the goal or moving
        goal_reached = True if self.agent_pos == [8, 8] else False
        if goal_reached:
            print(f" Goal reached ")
            pass
        reward = 1 if self.agent_pos == [8, 8] else -0.01
        if self.maze[y, x] == 1:
            reward -= 1
        info = {}
        # print(self.agent_pos)
        # reward -= self.calc_distance(self.agent_pos, [8, 8])

        # print(self.agent_pos)
        # print(self.maze)
        # print(self.observation_space)
        # print(self.observation_space_tensor)

        return self.observation_space_tensor, reward, terminated, truncated, info

    def calc_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.agent_pos = [0, 0]
        self.step_count = 0
        self.update_observation_from_state()
        return self.observation_space_tensor

        def render(self, mode="human"):
            if mode == "human":
                self._render_human()
            elif mode == "rgb_array":
                return self._render_rgb_array()

        def _render_human(self):
            maze_copy = np.copy(self.maze)
            # Mark agent's position
            maze_copy[self.agent_pos[0], self.agent_pos[1]] = 2
            plt.imshow(maze_copy, cmap="viridis")
            plt.show()

        def _render_rgb_array(self):
            # This method can be implemented if needed for returning an image array
            pass

        def close(self):
            plt.close()

        def flatten(self):
            return self.maze.flatten()


env = MazeEnv()
print("Maze")
print(env.maze)
print("Observation")
print(env.observation_space)
