import numpy as np

class MazeEnv:
    def __init__(self):
        self.grid_size = 5
        self.start = (0, 0)
        self.goal = (4, 4)
        self.max_steps = 50
        self.reset()

    def reset(self):
        self.agent_pos = list(self.start)
        self.steps = 0
        return tuple(self.agent_pos)

    def step(self, action):
        x, y = self.agent_pos

        # Actions: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
        if action == 0:
            x -= 1
        elif action == 1:
            x += 1
        elif action == 2:
            y -= 1
        elif action == 3:
            y += 1

        # Boundary check
        x = max(0, min(self.grid_size - 1, x))
        y = max(0, min(self.grid_size - 1, y))

        self.agent_pos = [x, y]
        self.steps += 1

        # Reward logic
        if (x, y) == self.goal:
            return (x, y), 10, True
        elif self.steps >= self.max_steps:
            return (x, y), -5, True
        else:
            return (x, y), -1, False