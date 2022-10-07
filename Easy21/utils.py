class ActionSpace:
    def __init__(self):
        self.int_to_actions = {0: "hit", 1: "stick"}
        self.actions_to_int = {"hit": 0, "stick": 1}