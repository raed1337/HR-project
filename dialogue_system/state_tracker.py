import numpy as np
import copy


class StateTracker:
    def __init__(self) -> None:
        self.initialize_episode()

    def initialize_episode(self):
        """ Initialize a new episode (dialog), flush the current state and tracked slots """

        self.action_dimension = 10
        self.history_vectors = np.zeros((1, self.action_dimension))
        self.history_dictionaries = []
        self.turn_count = 0
        self.current_slots = {}

        self.current_slots["inform_slots"] = {}
        self.current_slots["request_slots"] = {}
        self.current_slots["proposed_slots"] = {}
        self.current_slots["agent_request_slots"] = {}

    def dialog_history_dictionaries(self):
        """  Return the dictionary representation of the dialog history (includes values) """
        return self.history_dictionaries

    def get_state_for_agent(self):
        """ Get the state representatons to send to agent """

        state = {
            "user_action": self.history_dictionaries[-1],
            "current_slots": self.current_slots,
            "turn": self.turn_count,
            "history": self.history_dictionaries,
            "agent_action": self.history_dictionaries[-2] if len(self.history_dictionaries) > 1 else None,
        }
        return copy.deepcopy(state)

    # TODO  - CHECK UPDATE FUNCTION
    def update(self, agent_action=None, user_action=None):
        """ Update the state based on the latest action """

        ########################################################################
        #  Make sure that the function was called properly
        ########################################################################
        assert not (user_action and agent_action)
        assert user_action or agent_action

        ########################################################################
        #   Update state to reflect a new action by the agent.
        ########################################################################
        if agent_action:

            ####################################################################
            #   Handles the act_slot response (with values needing to be filled)
            ####################################################################
            if agent_action["act_slot_response"]:
                response = copy.deepcopy(agent_action["act_slot_response"])
                inform_slots = {}

                agent_action_values = {
                    "turn": self.turn_count,
                    "speaker": "agent",
                    "diaact": response["diaact"],
                    "inform_slots": inform_slots,
                    "request_slots": response["request_slots"],
                }

                agent_action["act_slot_response"].update(
                    {
                        "diaact": response["diaact"],
                        "inform_slots": inform_slots,
                        "request_slots": response["request_slots"],
                        "turn": self.turn_count,
                    }
                )

            elif agent_action["act_slot_value_response"]:
                agent_action_values = copy.deepcopy(agent_action["act_slot_value_response"])
                # print("Updating state based on act_slot_value action from agent")
                agent_action_values["turn"] = self.turn_count
                agent_action_values["speaker"] = "agent"

            ####################################################################
            #   This code should execute regardless of which kind of agent produced action
            ####################################################################
            for slot in agent_action_values["inform_slots"].keys():
                self.current_slots["proposed_slots"][slot] = agent_action_values["inform_slots"][slot]
                self.current_slots["inform_slots"][slot] = agent_action_values["inform_slots"][
                    slot
                ]  # add into inform_slots
                if slot in self.current_slots["request_slots"].keys():
                    del self.current_slots["request_slots"][slot]

            for slot in agent_action_values["request_slots"].keys():
                if slot not in self.current_slots["agent_request_slots"]:
                    self.current_slots["agent_request_slots"][slot] = "UNK"

            self.history_dictionaries.append(agent_action_values)
            current_agent_vector = np.ones((1, self.action_dimension))
            self.history_vectors = np.vstack([self.history_vectors, current_agent_vector])

        ########################################################################
        #   Update the state to reflect a new action by the user
        ########################################################################
        elif user_action:

            ####################################################################
            #   Update the current slots
            ####################################################################
            for slot in user_action["inform_slots"].keys():
                self.current_slots["inform_slots"][slot] = user_action["inform_slots"][slot]
                if slot in self.current_slots["request_slots"].keys():
                    del self.current_slots["request_slots"][slot]

            for slot in user_action["request_slots"].keys():
                if slot not in self.current_slots["request_slots"]:
                    self.current_slots["request_slots"][slot] = "UNK"

            self.history_vectors = np.vstack([self.history_vectors, np.zeros((1, self.action_dimension))])
            new_move = {
                "turn": self.turn_count,
                "speaker": "user",
                "request_slots": user_action["request_slots"],
                "inform_slots": user_action["inform_slots"],
                "diaact": user_action["diaact"],
            }
            self.history_dictionaries.append(copy.deepcopy(new_move))

        ########################################################################
        #   This should never happen if the asserts passed
        ########################################################################
        else:
            pass

        ########################################################################
        #   This code should execute after update code regardless of what kind of action (agent/user)
        ########################################################################
        self.turn_count += 1
