import random


class User:
    """
    Parent class for all user sims to inherit from.
    It's just a blueprint for RealUser and TestUser.
    """

    def __init__(self, action_set=None, slot_set=None, start_set=None):
        """Constructor shared by all user simulators"""

        self.action_set = action_set
        self.slot_set = slot_set
        self.start_set = start_set
        self.action_history = None

    def initialize_episode(self):
        """Initialize a new episode (dialog)"""

        print("initialize episode called, generating goal")
        self.goal = random.choice(self.start_set)
        self.goal["request_slots"]["ticket"] = "UNK"
        episode_over, user_action = self._sample_action()
        assert episode_over != 1, " but we just started"
        return user_action

    def next(self, system_action):
        pass

    def set_nlg_model(self, nlg_model):
        self.nlg_model = nlg_model

    def set_nlu_model(self, nlu_model):
        self.nlu_model = nlu_model

    def add_nl_to_action(self, user_action):
        """Add NL to User Dia_Act"""

        user_nlg_sentence = self.nlg_model.convert_diaact_to_nl(user_action, "usr")
        user_action["nl"] = user_nlg_sentence

        if self.simulator_act_level == 1:
            user_nlu_res = self.nlu_model.generate_dia_act(user_action["nl"])  # NLU
            if user_nlu_res != None:
                # user_nlu_res['diaact'] = user_action['diaact'] # or not?
                user_action.update(user_nlu_res)
