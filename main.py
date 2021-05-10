"""
Inspired from:
https://github.com/ankonzoid/LearningX/blob/master/classical_RL/multiarmed_bandit/multiarmed_bandit.py
"""


import numpy as np
from datetime import datetime

np.random.seed(datetime.now().microsecond)


class Environment:

    def __init__(self, probs):
        self.probs = probs

    def get_reward(self, action):
        """
        Suppose the success rate of a particular action is 0.37. It is
        equivalent to [0.0, 0.37) as favourable space of [0.0, 1.0].
        """
        return 1 if (np.random.random() < self.probs[action]) else 0


class Agent:

    def __init__(self, n_actions, eps):
        # n_actions <- total number of all possible action
        self.n_action = n_actions
        self.eps = eps
        self.counts = np.zeros(n_actions, dtype=np.int64)
        self.Q = np.zeros(n_actions, dtype=np.float64)

    def update_Q(self, action, reward):
        self.counts[action] += 1
        self.Q[action] += (1.0 / self.counts[action]) * (reward - self.Q[action])

    def get_action(self):
        # Epsilon-greedy policy
        if np.random.random() < self.eps:
            return np.random.randint(self.n_action)
        else:
            # return np.random.choice([k for k in range(self.n_action) if self.Q[k] == self.Q.max()])
            return np.random.choice(np.flatnonzero(self.Q == self.Q.max()))


if __name__ == '__main__':
    probs = [0.10, 0.50, 0.60, 0.80, 0.10,
             0.25, 0.60, 0.45, 0.75, 0.65]
    env = Environment(probs)
    agent = Agent(len(probs), 0.25)
    action_tracker = np.zeros(len(probs), np.int64)
    reward_tracker = np.zeros(len(probs), np.int64)
    for episode in range(500):
        action = agent.get_action()
        action_tracker[action] += 1

        reward = env.get_reward(action)
        reward_tracker[action] += reward
        agent.update_Q(action, reward)
    # print(agent.Q)
    print(action_tracker, reward_tracker, reward_tracker.sum())

