import os
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
# from import mpl_toolkits.mplot3d import Axes3D


class ActionSpace:
    def __init__(self):
        self.int_to_actions = {0: "hit", 1: "stick"}
        self.actions_to_int = {"hit": 0, "stick": 1}


# global constants
ACTION_SPACE = ActionSpace()
CARDS_NUM = 10
LEGAL_SUM_FIELD = 21
NUM_OF_ACTIONS = 2


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
