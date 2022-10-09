from agent import Agent
from utils import *
import numpy as np
import random
import os


class MCControlAgent(Agent):

    def __init__(self, N0=100, log_path='', **kwargs):
        super().__init__(N0, log_path)

    def train_agent(self, episodes, check_points=[], **kwargs):
        count_wins, count_loss, count_tie = 0, 0, 0
        save_format = os.path.join("MC_data", "Q_functions", "NUMBER_episodes_LAMBDA.npy")
        check_point_idx = 0
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

                if self.log and episodes - episode <= 100:
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
                count_loss += 1
            else:
                count_tie += 1

            # saving if wanted:
            if check_point_idx < len(check_points) and episode == check_points[check_point_idx]:
                check_point_idx += 1
                size = (1000, 'K') if episode < 1e6 else (1000 * 1000, 'M')
                save_path = save_format.replace("NUMBER", f"{int(episode / size[0])}{size[1]}")
                np.save(save_path, self.Q)

        if self.log:
            with open(self.log_path, 'a', newline='\n') as f:
                f.write(f"player won {count_wins} games\n"
                        f"player lost {count_loss} games\n"
                        f"{count_tie} games ended with a tie\n")

