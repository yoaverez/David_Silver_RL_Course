from utils import *
from MC_control import MCControlAgent
from SARSA_control import SarsaControlAgent
from FA_Sarsa import FaSarsaAgent

import os
import numpy as np


agents = {"MC": MCControlAgent, "Sarsa": SarsaControlAgent, "FA_sarsa": FaSarsaAgent}


def test_plots(agent_type='MC', episodes=1000*1000, log=False, check_points=[], **kwargs):
    size = (1000 * 1000, 'M') if episodes >= 1e6 else (1000, 'K')
    if log:
        log_path = os.path.join(f"{agent_type}_data", "logs", f"{int(episodes / size[0])}{size[1]}_episodes.txt")
        agent = agents[agent_type](log_path=log_path, **kwargs)
    else:
        agent = agents[agent_type](**kwargs)
    agent.train_agent(episodes, check_points=check_points, **kwargs)

    if agent_type == "FA_sarsa":
        agent.set_action_value_func_to_q()

    agent.plot_value_function(f"value function after {int(episodes/size[0])}{size[1]} episodes")
    agent.plot_action_value_function(f"action-value function after {int(episodes/size[0])}{size[1]} episodes")


def run_and_save_different_lambda(agent_type='Sarsa', episodes=1000, check_points=[1000],
                                  lambdas=np.arange(0, 1.1, 0.1)):
    size = (1000 * 1000, 'M') if episodes >= 1e6 else (1000, 'K')
    log_path = os.path.join(f"{agent_type}_data", "logs", f"{int(episodes / size[0])}{size[1]}_episodes_LAMBDA.txt")
    q_function_path = os.path.join(f"{agent_type}_data", "Q_functions",
                                   f"{int(episodes / size[0])}{size[1]}_episodes_LAMBDA.npy")
    agent = agents[agent_type]()

    for lambda_ in lambdas:
        log_path_temp = log_path.replace("LAMBDA", f"lambda_{lambda_:.1f}")
        # q_function_path_temp = q_function_path.replace("LAMBDA", f"lambda_{lambda_:.1f}")

        agent.reset()
        agent.set_log_path(log_path_temp)
        agent.train_agent(episodes=episodes, lambda_=lambda_, check_points=check_points)


def plot_mse_with_different_lambdas(agent_type='Sarsa', lambdas=np.arange(0, 1.1, 0.1), episodes=1000):
    size = (1000 * 1000, 'M') if episodes >= 1e6 else (1000, 'K')
    q_function_path = os.path.join(f"{agent_type}_data", "Q_functions",
                                   f"{int(episodes / size[0])}{size[1]}_episodes_LAMBDA.npy")
    mc_path = os.path.join("MC_data", "Q_functions", f"1M_episodes.npy")
    Q_star = np.load(mc_path)[1:]
    mse = []
    for lambda_ in lambdas:
        q_function_path_temp = q_function_path.replace("LAMBDA", f"lambda_{lambda_:.1f}")
        Q = np.load(q_function_path_temp)[1:]
        mse_t = Q**2 - 2 * Q * Q_star + Q_star**2
        mse.append(mse_t.sum())

    x_label = r"$\lambda$"
    y_label = r"$\sum_{s,a}{(Q(s,a)-Q^{*}(s,a))^2}$"
    title = r"mse for different $\lambda$ values"
    fig_path = os.path.join(f"{agent_type}_data", "pics",
                            f"mse_for_different_lambdas.png")
    plot_2D(x=lambdas, ys=[mse], x_label=x_label, y_label=y_label, title=title, ys_ids=["mse"], save_fig_path=fig_path)


def plot_mse_for_different_number_of_episodes(episodes_list, agent_type='Sarsa', train=False, log_path=""):
    if type(episodes_list) == list and not episodes_list:
        raise Exception("Must have episode list that contains at least one element")
    if train:
        agent = agents[agent_type](log_path=log_path)
        agent.train_agent(episodes=episodes_list[-1], check_points=episodes_list, lambda_=0)
        agent.reset()
        agent.train_agent(episodes=episodes_list[-1], check_points=episodes_list, lambda_=1)

    mc_path = os.path.join("MC_data", "Q_functions", f"1M_episodes.npy")
    Q_star = np.load(mc_path)[1:]

    x = episodes_list
    ys = [[], []]
    save_format = os.path.join(f"{agent_type}_data", "Q_functions", "NUMBER_episodes_LAMBDA.npy")
    for lambda_ in [0, 1]:
        for num_of_episodes in episodes_list:
            size = (1000, 'K') if num_of_episodes < 1e6 else (1000 * 1000, 'M')
            save_path = save_format.replace("NUMBER", f"{int(num_of_episodes / size[0])}{size[1]}")
            save_path = save_path.replace("LAMBDA", f"lambda_{lambda_:0.1f}")
            Q = np.load(save_path)[1:]

            mse_t = Q ** 2 - 2 * Q * Q_star + Q_star ** 2
            ys[lambda_].append(mse_t.sum())

    x_label = r"episodes"
    y_label = r"$\sum_{s,a}{(Q(s,a)-Q^{*}(s,a))^2}$"
    title = r"mse for different number of episodes"
    fig_path = os.path.join(f"{agent_type}_data", "pics",
                            f"mse_for_different_number_of_episodes.png")
    plot_2D(x=x, ys=ys, x_label=x_label, y_label=y_label, title=title,
            ys_ids=[r"$\lambda=0$", r"$\lambda=1$"], save_fig_path=fig_path)


def test_eval(episodes=100):
    agent = SarsaControlAgent()
    q_path = os.path.join("Sarsa_data", "Q_functions", "1M_episodes_lambda_1.0.npy")
    log_path = os.path.join("Sarsa_data", "logs", "test_eval.txt")
    agent.load_q(q_path)
    agent.eval(episodes, log_path=log_path)


def main():
    episodes = 1000 * 1000
    # test_plots(agent_type="MC", episodes=1000 * 1000)
    # test_plots(agent_type="Sarsa", episodes=50 * 1000, lambda_=0.2)
    # run_and_save_different_lambda()
    # plot_mse_with_different_lambdas()
    # plot_mse_for_different_number_of_episodes(episodes_list=np.arange(100, 10000+1, 100, dtype=int),
    #                                           train=True)
    # test_eval()
    # run_and_save_different_lambda(agent_type="FA_sarsa", episodes=episodes, check_points=[episodes],
                                  lambdas=[0.2])
    # plot_mse_with_different_lambdas(agent_type="FA_sarsa")
    # plot_mse_for_different_number_of_episodes(episodes_list=np.arange(100, 10000 + 1, 100, dtype=int),
    #                                           train=True, agent_type="FA_sarsa")
    # test_plots(agent_type="FA_sarsa", episodes=episodes, lambda_=0.2, check_points=[episodes])


if __name__ == "__main__":
    main()
