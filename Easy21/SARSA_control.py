from utils import *

from agent import Agent

import numpy as np
import random


class SarsaControlAgent(Agent):

    def __init__(self, N0=100, discount=1, log_path='', **kwargs):

        super().__init__(N0, log_path)

        self.gamma = discount

        self.eligibility_traces = np.zeros_like(self.N)

    def train_agent(self, episodes, lambda_=0, **kwargs):
        count_wins, count_loss, count_tie = 0, 0, 0
        if self.log:
            print(f"logging to {self.log_path}")
        for episode in range(1, episodes+1):
            print(f"running episode {episode}/{episodes}...")
            # sampling stage:
            self.reset_eligibility_traces()
            state = self.env.reset()
            action, int_action = self.eps_greedy_action(state)
            while not self.env.done:
                new_state, reward, log_str = self.env.step(state, action)

                i1, j1 = state
                i2, j2 = new_state

                self.N[i1, j1, int_action] += 1

                new_action, new_int_action = self.eps_greedy_action(new_state)

                q_prediction = reward
                if not self.env.done:
                    q_prediction += self.gamma * self.Q[i2, j2, new_int_action]

                delta_t = q_prediction - self.Q[i1, j1, int_action]
                self.eligibility_traces[i1, j1, int_action] += 1

                alpha_t = 1 / self.N[i1, j1, int_action]
                # update all the action-value function elements
                # and all the eligibility_traces elements

                # for i, rows in enumerate(self.eligibility_traces):
                #     for j, cols in enumerate(rows):
                #         for a, _ in enumerate(cols):
                #             self.Q[i, j, a] += alpha_t*delta_t*self.eligibility_traces[i, j, a]
                #             self.eligibility_traces[i, j, a] *= self.gamma * lambda_

                # alternative more fast update
                self.Q += alpha_t*delta_t*self.eligibility_traces
                self.eligibility_traces *= self.gamma * lambda_

                # update log
                if self.log and episodes - episode <= 100:
                    with open(self.log_path, 'a', newline='\n') as f:
                        s = f"dealer open card: {state[0]:3} {' ':3}{'|':3} player sum: {state[1]:3} {' ':3}{'|':3} " \
                            f"action: {action:6} {' ':3}{'|':3} reward: {reward:3}\n"
                        f.write(s)
                        f.write(log_str)

                state = new_state
                action, int_action = new_action, new_int_action

            if reward > 0:
                count_wins += 1
            elif reward < 0:
                count_loss += 1
            else:
                count_tie += 1
        if self.log:
            with open(self.log_path, 'a', newline='\n') as f:
                f.write(f"player won {count_wins} games\n"
                        f"player lost {count_loss} games\n"
                        f"{count_tie} games ended with a tie\n")

        print(self.Q)

    def reset_eligibility_traces(self):
        self.eligibility_traces = np.zeros_like(self.N)
