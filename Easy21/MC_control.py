from environment import Easy21
from utils import ActionSpace
import numpy as np
import random

ACTION_SPACE = ActionSpace()
CARDS_NUM = 10
LEGAL_SUM_FIELD = 21
NUM_OF_ACTIONS = 2


class MCControlAgent:

    def __init__(self, N0=100, log_path=''):
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

    def train_agent(self, episodes):
        count_wins, count_loss, count_tie = 0, 0, 0
        if self.log:
            print(f"logging to {self.log_path}")
        for episode in range(1, episodes+1):
            print(f"running episode {episode}/{episodes}...")
            # sampling stage:
            state = self.env.reset()
            returns = []
            while not self.env.done:
                # run more step
                action, int_action = self.eps_greedy_action(state)
                # i, j = state
                # self.N[i, j, int_action] += 1
                new_state, reward, log_str = self.env.step(state, action)
                returns.append((state, int_action, reward))

                if self.log:
                    with open(self.log_path, 'a', newline='\n') as f:
                        s = f"dealer open card: {state[0]:3} {' ':3}{'|':3} player sum: {state[1]:3} {' ':3}{'|':3} " \
                            f"action: {action:6} {' ':3}{'|':3} reward: {reward:3}\n"
                        f.write(s)
                        f.write(log_str)

                state = new_state

            # updating action-value function (i.e. Q)
            g_t = returns[-1][2]  # only the last reward is not zero and no discount
            for state, int_action, reward in returns:
                i, j = state
                self.N[i, j, int_action] += 1
                alpha_t = 1 / self.N[i, j, int_action]
                error_t = g_t - self.Q[i, j, int_action]
                self.Q[i, j, int_action] += alpha_t * error_t
            if g_t > 0:
                count_wins += 1
            elif g_t < 0:
                count_loss +=1
            else:
                count_tie += 1
            if self.log:
                with open(self.log_path, 'a', newline='\n') as f:
                    f.write(f"player won {count_wins} games\n"
                            f"player lost {count_loss} games\n"
                            f"{count_tie} games ended with a tie\n")

        print(self.Q)

    def plot_frame(self, ax):


        X = np.arange(0, 10, 1)
        Y = np.arange(0, 21, 1)
        X, Y = np.meshgrid(X, Y)
        Z = self.calc_value_func_from_action_value_func(self.Q)[1:, 1:].transpose()

        surf = ax.plot_surface(X, Y, Z, cmap=cm.bwr, antialiased=False)
        return surf


#%% Train and generate the value function
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import os


episodes = 1000*1000
log_path = os.path.join("logs", f"{int(episodes/1000)}K_episodes.txt")
agent = MCControlAgent(log_path=log_path)
agent.train_agent(episodes=episodes)

fig = plt.figure()
ax = fig.add_subplot(111,projection ='3d')
agent.plot_frame(ax)
plt.title('value function after 1M episodes yoav')
ax.set_xlabel('Dealer showing')
ax.set_ylabel('Player sum')
ax.set_zlabel('V(s)')
ax.set_xticks(range(1, 10+1))
ax.set_yticks(range(1, 21+1))
# plt.legend()
plt.show()
plt.savefig('Value function_50k.png')