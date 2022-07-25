class Agent:
    def __init__(self, action_set=None, slot_set=None) -> None:
        self.action_set = action_set
        self.slot_set = slot_set

    def initialize_episode(self):
        """
        Initialize a new episode.
        This function is called every time a new episode is run.
        """
        self.current_action = {}
        self.cuurent_action["diaact"] = None
        self.current_action["inform_slots"] = {}
        self.current_action["request_slots"] = {}
        self.current_action["turn"] = 0

    def state_to_action(self):
        pass

    def set_nlg_model(self, nlg_model_path):
        # TODO - add loading model
        pass

    def set_nlu_model(self, nlu_model_path):
        # TODO - add loading model
        pass
