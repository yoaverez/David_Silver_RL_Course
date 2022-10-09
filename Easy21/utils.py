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


def plot_2D(x, ys, ys_ids, font_size=16, fig_size=(10, 8), same_plot=True, x_label='', y_label='',
            title='', ys_colors=None, culomns=1, x_labels=[], y_labels=[], multiple_curves=False, curve_labels=[],
            save_fig_path=''):
    """
    print multiple curves on a single plot or multiple subplots

    :param x: The x_axis points. if same_plot=False than it's should be a list of x_axis points.
    Each list is x_axis points for one subplot.
    :param ys: List in which each element is a list of y_axis points.
    :param ys_ids: :List of names for each plot. if same_plot=True than it will appear as legends else
    as a title to specific subplot.
    :param font_size: The font size of the title and legends. Default is 16.
    In subplots the recommended font size is 10.
    :param fig_size: Adjust the size of the plot or each subplot.


    :param same_plot: If True, we plot all the ys curves on a single plot. Else plot each element of
    (x,y) in zip(x,ys) as a different subplot. Default is True.
    :param x_label: Used only if same_plot=True and is the title of the x_axis.
    :param y_label: Used only if same_plot=True and is the title of the y_axis.
    :param title: Used only if same_plot=True. The plot title.
    :param ys_colors: Used only if same_plot=True. List of colors (e.g. 'b', 'r') for each curve.

    :param culomns: Used only if same_plot=False. The number of subplots in each row. Default to 1.
    Deafult to None, meaning choose automatic colors.
    :param x_labels: Used only if same_plot=False. List of names for each subplot x_axis.
    :param y_labels: Used only if same_plot=False. List of names for each subplot y_axis.
    :param multiple_curves: Used only if same_plot=False. If true, each subplot will have multiple curves
    else, each subplot will have a single curve. Default is False.
    :param curve_labels: Used only if same_plot=False and multiple_curves=True. Contaions labels for each curve in each subplot.
    It's shape should be the same as ys shape.
    """
    if same_plot:
        fig, ax = plt.subplots(1, 1, figsize=fig_size, )
        plt.rcParams['font.size'] = f'{font_size}'
        for i, (y, y_id) in enumerate(zip(ys, ys_ids)):
            if ys_colors is None:
                ax.plot(x, y, label=y_id, marker='*')
            else:
                ax.plot(x, y, label=y_id, color=ys_colors[i], marker='*')

        ax.set_xlabel(x_label, fontsize=font_size)
        ax.set_ylabel(y_label, fontsize=font_size)
        ax.grid()
        ax.legend()
        ax.set_title(title)
    else:
        rows = len(ys) // culomns + 1

        fig, ax = plt.subplots(rows, culomns, figsize=fig_size, )
        plt.rcParams['font.size'] = f'{font_size}'
        for i in range(rows):
            for j in range(culomns):
                index = i * culomns + j
                if index >= len(ys):
                    ax[i, j].remove()
                    continue
                if multiple_curves:
                    for curve, curve_label in zip(ys[index], curve_labels[index]):
                        ax[i][j].plot(x[index], curve, label=curve_label, marker='*')
                    ax[i][j].legend()
                else:
                    ax[i][j].plot(x[index], ys[index])
                ax[i][j].set_xlabel(x_labels[index], fontsize=font_size)
                ax[i][j].set_ylabel(y_labels[index], fontsize=font_size)
                ax[i][j].set_title(ys_ids[index])
                ax[i][j].grid()
    fig.tight_layout()
    if save_fig_path != '':
        plt.savefig(save_fig_path)
    plt.show()
