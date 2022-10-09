from MC_control import MCControlAgent
import os
import numpy as np
from utils import plot_3D


def plot_value_function(episodes=1000*1000, log=False):
    size = (1000*1000, 'M') if episodes >= 1e6 else (1000, 'K')
    if log:
        log_path = os.path.join("logs", f"{int(episodes / size[0])}{size[1]}_episodes.txt")
        # log_path = os.path.join("logs", f"{int(episodes/size[0])}{size[1]}_episodes_dealer_sum_17.txt")
        agent = MCControlAgent(log_path=log_path)
    else:
        agent = MCControlAgent()

    agent.train_agent(episodes=episodes)

    x = np.arange(0, 10, 1)
    y = np.arange(0, 21, 1)
    z = agent.calc_value_func_from_action_value_func(agent.Q)[1:, 1:].transpose()

    title = f"value function after {int(episodes/size[0])}{size[1]} episodes"

    plot_3D(x, y, z, title=title)


def plot_action_value_function(episodes=1000*1000, log=False):
    size = (1000*1000, 'M') if episodes >= 1e6 else (1000, 'K')
    if log:
        log_path = os.path.join("logs", f"{int(episodes / size[0])}{size[1]}_episodes.txt")
        # log_path = os.path.join("logs", f"{int(episodes/size[0])}{size[1]}_episodes_dealer_sum_17.txt")
        agent = MCControlAgent(log_path=log_path)
    else:
        agent = MCControlAgent()

    agent.train_agent(episodes=episodes)

    x = np.arange(0, 10, 1)
    y = np.arange(0, 21, 1)
    z = np.transpose(agent.Q[1:, 1:], axes=(1, 0, 2))

    title = f"action-value function after {int(episodes/size[0])}{size[1]} episodes"

    plot_3D(x, y, z, title=title, zlabel="Q(s,a)", titles=["Q(s,a=hit)", "Q(s,a=stick)"], single_plot=False)


def main():
    # plot_value_function(episodes=50*1000)
    plot_action_value_function(episodes=5*1000)


if __name__ == "__main__":
    main()
