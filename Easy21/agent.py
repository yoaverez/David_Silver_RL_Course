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

    @staticmethod
    def hit_until_17(state):
        action = "hit" if state[1] < 17 else "stick"
        int_action = ACTION_SPACE.actions_to_int[action]
        return action, int_action

    def greedy_action(self, state):
        i, j = state
        int_action = np.argmax(self.Q[i, j])
        return ACTION_SPACE.int_to_actions[int_action], int_action

    def eps_greedy_action(self, state, eps_t=None):
        if self.env.done:
            return "not_important", -1
        i, j = state
        if eps_t is None:
            N_state = np.sum(self.N[i, j])
            eps_t = self.N0/(self.N0 + N_state)
        elif type(eps_t) != int and type(eps_t) != float:
            print(type(eps_t))
            raise Exception("eps_t must be None of a number!!!")

        if random.random() < eps_t:
            # pick random action
            if random.random() < 0.5:
                return "hit", ACTION_SPACE.actions_to_int["hit"]
            else:
                return "stick", ACTION_SPACE.actions_to_int["stick"]
        else:
            # pick greedy action
            return self.greedy_action(state)

    def train_agent(self, episodes, **kwargs):
        raise Exception("VIRTUAL FUNCTION NOT IMPLEMENTED")

    def eval(self, episodes, log_path=''):
        log = False if log_path == '' else True
        count_wins, count_loss, count_tie = 0, 0, 0
        reward = 0
        for episode in range(1, episodes+1):
            print(f"running episode {episode}/{episodes}...")

            # sampling stage:
            state = self.env.reset()

            while not self.env.done:
                action, int_action = self.greedy_action(state)
                # action, int_action = self.hit_until_17(state)

                new_state, reward, log_str = self.env.step(state, action)

                # update log
                s = f"dealer open card: {state[0]:3} {' ':3}{'|':3} player sum: {state[1]:3} {' ':3}{'|':3} " \
                    f"action: {action:6} {' ':3}{'|':3} reward: {reward:3}\n"
                if log and episodes - episode <= 100:
                    with open(log_path, 'a', newline='\n') as f:
                        f.write(s)
                        f.write(log_str)
                elif episodes - episode <= 100:
                    print(s)
                    print(log_str)

                state = new_state

            if reward > 0:
                count_wins += 1
            elif reward < 0:
                count_loss += 1
            else:
                count_tie += 1

        log_str = f"player won {count_wins} games\n" \
                  f"player lost {count_loss} games\n" \
                  f"{count_tie} games ended with a tie\n"
        if log:
            with open(log_path, 'a', newline='\n') as f:
                f.write(log_str)
        else:
            print(log_str)

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

    def load_q(self, q_path):
        self.Q = np.load(q_path)
