import random

class Easy21:

    def __init__(self):
        self.done = False

    @staticmethod
    def hit():
        card_val = random.randint(1, 10)
        # card_color = -1 if random.random() < (1/3.0) else 1
        card_color = -1 if random.randint(1, 3) == 3 else 1
        return card_color * card_val

    @staticmethod
    def goes_bust(cards_sum):
        return cards_sum > 21 or cards_sum < 1

    def is_done(self):
        return self.done

    def dealer_turn(self, state):
        log_str = ""
        dealer_open_card, player_sum = state
        dealer_sum = dealer_open_card
        while dealer_sum < 17:
            dealer_sum += self.hit()
            log_str += f"dealer sum: {dealer_sum:3} {' ':3}{'|':3} player sum: {player_sum:3}\n"
            if self.goes_bust(dealer_sum):
                log_str += f"dealer got bust with sum {dealer_sum}\n\n"
                return (dealer_open_card, player_sum), +1, log_str
        reward = 0
        if player_sum > dealer_sum:
            reward = 1
        elif player_sum < dealer_sum:
            reward = -1
        return (dealer_open_card, player_sum), reward, log_str + '\n'

    def step(self, state, action: str):
        log_str = ''
        if self.done:
            print(f"You are in terminal state")
            return state, 0, log_str  # next state, reward, log_str

        dealer_open_card, player_sum = state
        if action.lower() == "hit":
            player_sum += self.hit()
            if self.goes_bust(player_sum):
                log_str += f"player got bust with sum {player_sum}\n"
                self.done = True
                return (dealer_open_card, player_sum), -1, log_str  # next state, reward, log_str

            else:
                return (dealer_open_card, player_sum), 0, log_str

        elif action.lower() == "stick":
            # no more player turns - player is done
            self.done = True
            # dealer turn
            return self.dealer_turn(state=(dealer_open_card, player_sum))  # next state, reward, log_str

    def reset(self):
        player_sum = random.randint(1, 10)
        dealer_open_card = random.randint(1, 10)
        self.done = False
        return dealer_open_card, player_sum
