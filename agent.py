import random

class RandomAgent:
    def __init__(self):
        self.actions = [0, 1, 2, 3]

    def choose_action(self):
        return random.choice(self.actions)