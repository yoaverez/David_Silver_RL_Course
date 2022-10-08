from MC_control import MCControlAgent
import os
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
# from import mpl_toolkits.mplot3d import Axes3D


def plot_3D(x, y, z, title, xlabel='Dealer showing', ylabel='Player sum', zlabel='V(s)', titles=None,
            single_plot=True):
    ncols = 1 if z.ndim == 2 or single_plot else 2
    is_2_plots = True if ncols == 2 else False
    fig, axs = plt.subplots(nrows=1, ncols=ncols, subplot_kw=dict(projection='3d'))

    if not is_2_plots:
        axs = [axs]

    x_, y_ = np.meshgrid(x, y)

    # surf = ax.plot_surface(X, Y, Z, cmap=plt.cm.cividis)
    for i, ax in enumerate(axs):
        # if is_2_plots:
        #     surf = ax.plot_surface(x_, y_, z[:, :, i])
        # else:
        #     surf = ax.plot_surface(x_, y_, z[:, :, i], cmap=plt.cm.bwr)
        if not is_2_plots:
            if z.ndim == 3:
                surf1 = ax.plot_surface(x_, y_, z[:, :, i], cmap='winter')
                surf2 = ax.plot_surface(x_, y_, z[:, :, 1 - i], cmap='autumn')
            else:
                surf = ax.plot_surface(x_, y_, z, cmap=plt.cm.bwr)
        else:
            surf = ax.plot_surface(x_, y_, z[:, :, i], cmap='winter')

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)
        ax.set_xticks([i + 1 for i in x])
        ax.set_yticks([i + 1 for i in y[::2]])
        if titles is not None:
            ax.title.set_text(titles[i])

    fig.tight_layout()
    plt.suptitle(title)

    if not is_2_plots or z.ndim == 3:
        fig.colorbar(surf, shrink=0.5, aspect=8, pad=0.1)
    plt.show()


def plot_value_function(episodes=1000*1000):
    size = (1000*1000, 'M') if episodes >= 1e6 else (1000, 'K')
    log_path = os.path.join("logs", f"{int(episodes/size[0])}{size[1]}_episodes.txt")
    agent = MCControlAgent(log_path=log_path)
    agent.train_agent(episodes=episodes)

    x = np.arange(0, 10, 1)
    y = np.arange(0, 21, 1)
    z = agent.calc_value_func_from_action_value_func(agent.Q)[1:, 1:].transpose()

    title = f"value function after {int(episodes/size[0])}{size[1]} episodes"

    plot_3D(x, y, z, title=title)


def plot_action_value_function(episodes=1000*1000):
    size = (1000*1000, 'M') if episodes >= 1e6 else (1000, 'K')
    log_path = os.path.join("logs", f"{int(episodes/size[0])}{size[1]}_episodes.txt")
    # log_path = os.path.join("logs", f"{int(episodes/size[0])}{size[1]}_episodes_dealer_sum_17.txt")

    agent = MCControlAgent(log_path=log_path)
    agent.train_agent(episodes=episodes)

    x = np.arange(0, 10, 1)
    y = np.arange(0, 21, 1)
    z = np.transpose(agent.Q[1:, 1:], axes=(1, 0, 2))

    title = f"action-value function after {int(episodes/size[0])}{size[1]} episodes"

    plot_3D(x, y, z, title=title, zlabel="Q(s,a)", titles=["Q(s,a=hit)", "Q(s,a=stick)"], single_plot=False)


def main():
    # plot_value_function(episodes=5*1000)
    plot_action_value_function(episodes=5*1000)


if __name__ == "__main__":
    main()
