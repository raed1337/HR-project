from users.user import User


class RealUser(User):
    """
    A RealUser is a user that interacts with the system through terminal.
    """

    def __init__(self, action_set: list = None, slot_set: list = None, start_set: list = None, nlu_model=None):
        """
        Consturctor for RealUser class.

        Args:
            action_set (list, optional): _description_. Defaults to None.
            slot_set (list, optional): _description_. Defaults to None.
            start_set (list, optional): _description_. Defaults to None.
            nlu_model (_type_, optional): The NLU model which is responsible for understanding the user's response. Defaults to None.
        """

        self.action_set = action_set
        self.slot_set = slot_set
        self.start_set = start_set
        self.nlu_model = nlu_model
        self.action_history = []

    def initialize_episode(self) -> dict:
        """
        This function should only be called once and in the very beginning of the dialogue.
        It will take the first user action by calling the _sample_action function.
        Returns:
            dict: user_action
        """

        self.state = {}
        self.state["history_slots"] = {}
        self.state["inform_slots"] = {}
        self.state["request_slots"] = {}
        self.state["rest_slots"] = []
        self.state["turn"] = 0

        self.episode_over = False

        user_action = self._sample_action()

        return user_action

    def _sample_action(self):
        """sample a start action"""
        user_action = self.generate_diaact_from_nl("hi pam")
        user_action["turn"] = self.state["turn"]

        self.state["diaact"] = user_action["diaact"]
        self.state["inform_slots"] = user_action["inform_slots"]
        self.state["request_slots"] = user_action["request_slots"]

        return user_action

    def next(self, turn: int) -> dict:
        """
        This function is called everytime it is the user's turn.
        It will take the user's input and generate a dia_act by passing the reponse to user's NLU.
        Args:
            turn (int): an integer which takes track of how many turns it has taken till now.

        Returns:
            dict: NLU's output from user's response.
        """

        self.state["turn"] = turn
        self.episode_over = False

        user_input = input("\033[1;34m" + "+ User: " + "\033[0m")

        response_action = self.generate_diaact_from_nl(user_input)

        response_action["turn"] = self.state["turn"]
        self.state["diaact"] = response_action["diaact"]
        self.state["inform_slots"] = response_action["inform_slots"]
        self.state["request_slots"] = response_action["request_slots"]

        self.action_history.append(response_action)

        return response_action

    def generate_diaact_from_nl(self, user_input: str) -> dict:
        """
        This function calls user's NLU.

        Args:
            user_input (str): user's input

        Returns:
            dict: NLU's output.
        """

        user_action = {}
        user_action["diaact"] = "UNK"
        user_action["inform_slots"] = {}
        user_action["request_slots"] = {}

        if len(user_input) > 0:
            user_action = self.nlu_model.generate_dia_act(user_input)

        user_action["nl"] = user_input

        return user_action
