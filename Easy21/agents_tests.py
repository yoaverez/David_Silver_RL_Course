from MC_control import MCControlAgent
from SARSA_control import SarsaControlAgent
import os

agents = {"mc": MCControlAgent, "sarsa": SarsaControlAgent}


def test_plots(agent_type='MC', episodes=1000*1000, log=False, **kwargs):
    size = (1000 * 1000, 'M') if episodes >= 1e6 else (1000, 'K')
    if log:
        log_path = os.path.join("Sarsa_data", "logs", f"{int(episodes / size[0])}{size[1]}_episodes.txt")
        agent = agents[agent_type](log_path=log_path, **kwargs)
    else:
        agent = agents[agent_type](**kwargs)
    agent.train_agent(episodes)
    agent.plot_value_function(f"value function after {int(episodes/size[0])}{size[1]} episodes")
    agent.plot_action_value_function(f"action-value function after {int(episodes/size[0])}{size[1]} episodes")


def main():
    test_plots(agent_type="sarsa", episodes=50 * 1000, lambda_=0.2)


if __name__ == "__main__":
    main()
