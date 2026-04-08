import numpy as np

class MazeEnv:
    def __init__(self):
        self.grid_size = 5
        self.start = (0, 0)
        self.goal = (4, 4)
        self.max_steps = 50
        self.agent_pos = list(self.start)
        self.steps = 0

    def reset(self):
        self.agent_pos = list(self.start)
        self.steps = 0
        obs = np.array(self.agent_pos, dtype=np.float32)
        return obs, {}  # ✅ returns (np.array, dict)

    def step(self, action):
        x, y = self.agent_pos

        if action == 0: x -= 1
        elif action == 1: x += 1
        elif action == 2: y -= 1
        elif action == 3: y += 1

        x = max(0, min(self.grid_size - 1, x))
        y = max(0, min(self.grid_size - 1, y))

        self.agent_pos = [x, y]
        self.steps += 1

        obs = np.array(self.agent_pos, dtype=np.float32)

        if (x, y) == self.goal:
            return obs, 10.0, True, False, {}
        elif self.steps >= self.max_steps:
            return obs, -5.0, False, True, {}
        else:
            return obs, -1.0, False, False, {}