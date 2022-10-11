from utils import *
import numpy as np
from SARSA_control import SarsaControlAgent

NUM_OF_FEATURES = 36


class FaSarsaAgent(SarsaControlAgent):
    def __init__(self, alpha=0.01, epsilon=0.05, **kwargs):
        super().__init__(**kwargs)
        self.w = np.zeros(shape=NUM_OF_FEATURES)
        self.eligibility_traces = np.zeros_like(self.w)
        self.alpha = alpha
        self.eps = epsilon
        # self.to_indices, self.from_indices = built_state_action_to_indices_vocab()

    @staticmethod
    def dealer_index(dealer_first_card):
        indices = []
        if dealer_first_card in range(1, 4 + 1):
            indices.append(0)
        if dealer_first_card in range(4, 7 + 1):
            indices.append(1)
        if dealer_first_card in range(7, 10 + 1):
            indices.append(2)
        return indices

    @staticmethod
    def player_sum_index(player_sum):
        indices = []
        if player_sum in range(1, 6 + 1):
            indices.append(0)
        if player_sum in range(4, 9 + 1):
            indices.append(1)
        if player_sum in range(7, 12 + 1):
            indices.append(2)
        if player_sum in range(10, 15 + 1):
            indices.append(3)
        if player_sum in range(13, 18 + 1):
            indices.append(4)
        if player_sum in range(16, 21 + 1):
            indices.append(5)
        return indices

    @staticmethod
    def action_index(action):
        int_action = ACTION_SPACE.actions_to_int[action]
        return [int_action]

    @staticmethod
    def get_full_feature_vector(state, action):
        dealer_first_card, player_sum = state
        i = FaSarsaAgent.dealer_index(dealer_first_card)
        j = FaSarsaAgent.player_sum_index(player_sum)
        k = FaSarsaAgent.action_index(action)
        feature_vector = np.zeros(shape=NUM_OF_FEATURES)
        for i_t in i:
            for j_t in j:
                for k_t in k:
                    index = i_t * 12 + j_t * 2 + k_t
                    feature_vector[index] = 1
        return feature_vector

    @staticmethod
    def built_state_action_to_indices_vocab():
        to_indices = {}
        from_indices = {}
        for dealer_first_card in range(1, CARDS_NUM + 1):
            for player_sum in range(1, LEGAL_SUM_FIELD + 1):
                for action in ["hit", "stick"]:
                    state = (dealer_first_card, player_sum)
                    to_indices[(dealer_first_card, player_sum, action)] = FaSarsaAgent.get_full_feature_vector(state, action)
        for key, value in to_indices.items():
            from_indices[value.tobytes()] = key
        return to_indices, from_indices

    def greedy_action(self, state):
        fv0 = FaSarsaAgent.get_full_feature_vector(state, "hit")
        fv1 = FaSarsaAgent.get_full_feature_vector(state, "stick")
        int_action = np.argmax([fv0 @ self.w, fv1 @ self.w])
        return ACTION_SPACE.int_to_actions[int_action], int_action

    def train_agent(self, episodes, lambda_=0, check_points=[], **kwargs):
        count_wins, count_loss, count_tie = 0, 0, 0
        save_format = os.path.join("FA_sarsa_data", "Q_functions", "NUMBER_episodes_LAMBDA.npy")
        check_point_idx = 0
        if self.log:
            print(f"logging to {self.log_path}")
        for episode in range(1, episodes+1):
            print(f"running episode {episode}/{episodes}...")
            # sampling stage:
            self.reset_eligibility_traces()
            state = self.env.reset()
            action, int_action = self.eps_greedy_action(state, eps_t=self.eps)
            while not self.env.done:
                new_state, reward, log_str = self.env.step(state, action)
                fv = self.get_full_feature_vector(state, action)

                i1, j1 = state
                i2, j2 = new_state

                new_action, new_int_action = self.eps_greedy_action(new_state, eps_t=self.eps)

                q_prediction = reward
                if not self.env.done:
                    fv_new = self.get_full_feature_vector(new_state, new_action)
                    q_prediction += self.gamma * (fv_new @ self.w)

                delta_t = q_prediction - fv @ self.w
                self.eligibility_traces += fv

                # update all the action-value function elements
                # and all the eligibility_traces elements
                self.w += self.alpha * delta_t * self.eligibility_traces
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

            # saving if wanted:
            if check_point_idx < len(check_points) and episode == check_points[check_point_idx]:
                self.set_action_value_func_to_q()
                check_point_idx += 1
                size = (1000, 'K') if episode < 1e6 else (1000*1000, 'M')
                save_path = save_format.replace("NUMBER", f"{int(episode/size[0])}{size[1]}")
                save_path = save_path.replace("LAMBDA", f"lambda_{lambda_:0.1f}")
                np.save(save_path, self.Q)

        if self.log:
            with open(self.log_path, 'a', newline='\n') as f:
                f.write(f"player won {count_wins} games\n"
                        f"player lost {count_loss} games\n"
                        f"{count_tie} games ended with a tie\n")

    def reset_eligibility_traces(self):
        self.eligibility_traces = np.zeros_like(self.w)

    def set_action_value_func_to_q(self):
        self.Q = np.zeros_like(self.Q)
        for i, rows in enumerate(self.Q[1:, 1:]):
            for j, cols in enumerate(rows):
                for k, _ in enumerate(cols):
                    state = i, j
                    action = ACTION_SPACE.int_to_actions[k]
                    fv = self.get_full_feature_vector(state, action)
                    self.Q[i + 1, j + 1, k] = fv @ self.w
        return self.Q

    def reset(self):
        self.N = np.zeros_like(self.N)  # (state, action)
        self.Q = np.zeros_like(self.N)
        self.w = np.zeros_like(self.w)