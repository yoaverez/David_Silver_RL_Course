from MC_control import MCControlAgent
import os


def main():
    episodes = 100*1000
    log_path = os.path.join("logs", f"{int(episodes/1000)}K_episodes.txt")
    agent = MCControlAgent(log_path=log_path)
    agent.train_agent(episodes=episodes)


if __name__ == "__main__":
    main()
