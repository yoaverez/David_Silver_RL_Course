from utils import *
from environment import Easy21
import random
import numpy as np


class Agent:
    def __init__(self, N0=100, log_path='', **kwargs):
        self.log = False if log_path == '' else True
        self.log_path = log_path
        if self.log:
            with open(self.log_path, 'w', newline='\n') as f:
                # clean file
                pass
        self.env = Easy21()
        self.N0 = N0
        self.N = np.zeros(shape=(CARDS_NUM + 1, LEGAL_SUM_FIELD + 1, NUM_OF_ACTIONS))  # (state, action)
        self.Q = np.zeros_like(self.N)

    @staticmethod
    def calc_value_func_from_action_value_func(Q):
        n_d1, n_d2, n_d3 = Q.shape
        V = np.amax(Q, axis=2)
        # sanity check:
        assert V.shape == (n_d1, n_d2)
        return V

    def eps_greedy_action(self, state):
        if self.env.done:
            return "not_important", -1
        i, j = state
        N_state = np.sum(self.N[i, j])
        eps_t = self.N0/(self.N0 + N_state)
        if random.random() < eps_t:
            # pick random action
            if random.random() < 0.5:
                return "hit", ACTION_SPACE.actions_to_int["hit"]
            else:
                return "stick", ACTION_SPACE.actions_to_int["stick"]
        else:
            # pick greedy action
            int_action = np.argmax(self.Q[i, j])
            return ACTION_SPACE.int_to_actions[int_action], int_action

    def train_agent(self, episodes, **kwargs):
        raise Exception("VIRTUAL FUNCTION NOT IMPLEMENTED")

    def plot_value_function(self, title):
        x = np.arange(0, 10, 1)
        y = np.arange(0, 21, 1)
        z = self.calc_value_func_from_action_value_func(self.Q)[1:, 1:].transpose()

        plot_3D(x, y, z, title=title)

    def plot_action_value_function(self, title, single_plot=False):
        x = np.arange(0, 10, 1)
        y = np.arange(0, 21, 1)
        z = np.transpose(self.Q[1:, 1:], axes=(1, 0, 2))

        plot_3D(x, y, z, title=title, zlabel="Q(s,a)", titles=["Q(s,a=hit)", "Q(s,a=stick)"], single_plot=single_plot)

    def set_log_path(self, log_path):
        if log_path == "":
            self.log = False
            self.log_path = log_path
        else:
            self.log = True
            self.log_path = log_path

    def reset(self):
        self.N = np.zeros(shape=(CARDS_NUM + 1, LEGAL_SUM_FIELD + 1, NUM_OF_ACTIONS))  # (state, action)
        self.Q = np.zeros_like(self.N)
