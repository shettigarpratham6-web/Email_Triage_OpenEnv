import numpy as np
import random

class QLearningAgent:
    def __init__(self):
        self.q_table = {}
        self.actions = [0, 1, 2, 3]
        self.lr = 0.1
        self.gamma = 0.9
        self.epsilon = 0.2

    def get_q(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        q_values = [self.get_q(state, a) for a in self.actions]
        return self.actions[np.argmax(q_values)]

    def update(self, state, action, reward, next_state):
        old_q = self.get_q(state, action)
        next_max = max([self.get_q(next_state, a) for a in self.actions])
        new_q = old_q + self.lr * (reward + self.gamma * next_max - old_q)
        self.q_table[(state, action)] = new_q