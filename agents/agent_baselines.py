from agents.agent import Agent
import dialogue_system.dialogue as dc


class BasicAgent(Agent):
    def initialize_episode(
        self,
    ):
        """
        Inherited from Agent class.
        The request set contains the first two actions which the agent should take at the begning of each episode.
        """
        self.state = {}
        self.state["diaact"] = "UNK"
        self.state["inform_slots"] = {}
        self.state["request_slots"] = {}
        self.state["turn"] = -1
        self.current_slot_id = 0
        self.dialogue = dc.Dialogue()
        self.dialogue.initialize_dialogue()
        self.request_set = []
        self.phase = 0

    def state_to_action(self):
        """Run current policy on state and produce an action"""

        self.state["turn"] += 2
        if self.current_slot_id < len(self.request_set):
            slot = self.request_set[self.current_slot_id].id

            act_slot_response = {}
            act_slot_response["diaact"] = "request"
            act_slot_response["inform_slots"] = {}
            act_slot_response["request_slots"] = {slot: "UNK"}
            act_slot_response["turn"] = self.state["turn"]

            self.current_slot_id += 1
            self.dialogue.question_num += 1

        elif self.phase == 0:
            act_slot_response = {
                "diaact": "inform",
                "inform_slots": {"taskcomplete": "PLACEHOLDER"},
                "request_slots": {},
                "turn": self.state["turn"],
            }
            self.phase += 1

        elif self.phase == 1:
            act_slot_response = {
                "diaact": "thanks",
                "inform_slots": {},
                "request_slots": {},
                "turn": self.state["turn"],
            }

        else:
            raise Exception("THIS SHOULD NOT BE POSSIBLE (AGENT CALLED IN UNANTICIPATED WAY)")

        return {"act_slot_response": act_slot_response, "act_slot_value_response": None}
