import os, json, time, logging

import numpy as np

from dialogue_system.state_tracker import StateTracker
from users.user import User
from users.test_user import TestUser
from users.test_user import RealUser
from agents.agent import Agent
from agents.agent_baselines import BasicAgent
from decision_maker.dataframe_creator import DataFrameMaker
from decision_maker.decision_tree_pred import DecisionTreePred
from pandas import DataFrame

"""
The dialogue manager is responsible for coordinating the conversation between the agent and the user.
IT IS NOT RESPONSIBLE FOR FIGURING OUT WHICH QUESTION TO ASK FROM THE USER.
IF YOU ARE LOOKING FOR THAT, IT IS IN dialogue.py
"""


class DialogueManager:
    """
    The dialogue manager is responsible for coordinating the conversation between the agent and the user.
    It will recieve an agent and a user object as input and will then use them to run the dialogue.
    It has an internal component called the StateTracker which keeps track of the conversation happening between agent and user.
    This ocmponent will be used when the conversation is over and a decision must be made.
    """

    def __init__(self, agent: BasicAgent, user: User) -> None:
        """
        Constructor for the DialogueManager class.
        along side intilizing everything, it will create a StateTracker object and a log file.
        Args:
            agent (Agent): Rule Based agent
            user (User): Either real user which will interact through terminal or a test user which will go through a csv file.
        """
        try:
            self.agent = agent
            self.user = user
            self.action_set = None
            self.slot_set = None
            self.state_tracker = StateTracker()
            self.user_action = None
            self.episode_over = False
            self.slot_history = []
            self.main_problem = None
            self.responses = {}
            self.initialize_logs()
            self.conversation_over = False
            self.operation_log.info("Dialogue Manager Initialized")
        except Exception:
            self.operation_log.exception("Error in Dialogue Manager Initialization")

    def initialize_logs(self):
        # initilize conversation log
        if not os.path.exists(os.path.join("log", "conversation_log")):
            os.makedirs(os.path.join("log", "conversation_log"))
        self.conversation_log_file_name = (
            os.path.join("log", "conversation_log")
            + f"/{'RealUser' if type(self.user) == RealUser else 'TestUser'}_"
            + f"{time.strftime('%Y_%m_%d__%H_%M_%S')}"
            + ".csv"
        )
        self.conversation_log_file = open(self.conversation_log_file_name, "a")
        self.conversation_log_file.write(
            "Agent Query ID,Agent Query Inform,User Response DiaAct,User Response Request Slot,User Response NL\n"
        )
        self.conversation_log_file.close()

        # initialize opeartion log
        if not os.path.exists(os.path.join("log", "operation_log")):
            os.makedirs(os.path.join("log", "operation_log"))
        logging.basicConfig(
            filename=os.path.join("log", "operation_log")
            + f"/{'RealUser' if type(self.user) == RealUser else 'TestUser'}_"
            + f"{time.strftime('%Y_%m_%d__%H_%M_%S')}"
            + ".log",
            filemode="a",
            format="%(name)s - %(module)s - %(funcName)s - %(asctime)s - %(levelname)s - %(message)s",
            level=logging.INFO,
        )

        self.operation_log = logging.getLogger(__name__)

    def log_conversation(self):
        """
        This function is called after each conversation step.
        A step is defined as an agent query from the user and user's reponse to it.
        """
        request_id = list(self.agent_action["act_slot_response"]["request_slots"].keys())[0]
        self.conversation_log_file = open(self.conversation_log_file_name, "a")
        self.conversation_log_file.write(
            f"{request_id}"
            + ","
            + f"{self.agent.dialogue.all_dialogues[request_id]['inform_slots']}"
            + ","
            + f"{self.user_action['diaact']}"
            + ","
            + f"{self.user_action['inform_slots']}"
            + ","
            + f"{self.user_action['nl']}"
            + "\n"
        )
        self.conversation_log_file.close()

    def initialize_episode(self):
        """
        This fuunction must be called once at the very beginning of the dialogue.
        It will initilize the state_tracker, take the first action ("hi pam") to initilize the conversation and get the agent rolling.
        """
        try:
            self.episode_over = False
            self.state_tracker.initialize_episode()
            self.user_action = self.user.initialize_episode()
            self.state_tracker.update(user_action=self.user_action)
            self.agent.initialize_episode()
            self.conversation_over = False
            self.main_problem_set = False
            self.main_problem = None
            self.sub_problem = None
            self.operation_log.info("Episode Initialized")
        except Exception:
            self.operation_log.exception("Error in Episode Initialization")

    def next_turn(self):
        """
        This function is reponsible for driving the conversation forward.
        Every time this function is called, it will take the next question from the set of agent quuestion and take user's reponse.
        If user's response is not what it should be accroding to the "inform_slots" defined in each questions "infor_slot" in the dialogs.json,
        dialogue manager will ask the user to repeat the question.
        """
        try:
            self.state = self.state_tracker.get_state_for_agent()
            self.operation_log.info(f"State for agent: {self.state}")
        except Exception:
            self.operation_log.exception("Error in getting state for agent")

        # Getting the question that agent must ask from the user happens in these two lines of code

        # at the begning of the question lists, lies the question that must be asked
        try:
            agent_query = self.agent.dialogue.questions.pop(0)
            self.operation_log.info(f"Agent Query: {agent_query.id} - {agent_query.text}")
        except Exception:
            self.operation_log.exception("Error in getting agent query")
        # we really don't need this for decision making. It is only kept for helping when tracing the program by debugging.
        try:
            self.agent.request_set.extend([agent_query])
            self.operation_log.info(f"Agent Request Set Extended")
        except Exception:
            self.operation_log.exception("Error in extending agent request set")

        # returns an empty dia_act dict which we fill with user's response later in the conversation
        try:
            self.agent_action = self.agent.state_to_action()
            self.operation_log.info(f"Agent Action: {self.agent_action}")
        except Exception:
            self.operation_log.exception("Error in getting agent action")

        # state_tracker is used to keep track of whats happening between the agent and the user.
        # we really don't use it. it is kept for tracing the program when debugging.
        try:
            self.state_tracker.update(agent_action=self.agent_action)
            self.sys_action = self.state_tracker.dialog_history_dictionaries()[-1]
            self.operation_log.info(f"State Tracker updated with agent action")
        except Exception:
            self.operation_log.exception("Error in updating state tracker with agent action")

        print("\033[92m" + f"{agent_query.text}" + "\033[0m")

        # user's input is processed in the following line
        try:
            self.user_action = self.user.next(self.user_action["turn"] + 2)
            self.operation_log.info(f"User Action (NLU's ouput): {self.user_action}")
        except Exception:
            self.operation_log.exception("Error in getting user action")

        # This for condtion and the following elif, are repsonsible for checking if user's input contains
        # the information the agent asked about.
        if agent_query.answer_condition == "and":
            # if a question is asking for multiple inform_slotsand all of them should be given by the user,
            # the user's input will be checked for each of them. If even of them is not given, the user will be asked to repeat the answer with a different phrasing.
            try:
                while not all(
                    element in list(self.user_action["inform_slots"].keys()) for element in list(agent_query.answer)
                ):
                    # In case we are using the TestUser, One failure is enough to terminate the test.
                    if type(self.user) == TestUser:
                        print("\033[91m" + "Test Failed: Agent couldn't understand the utterance" + "\033[0m")
                        self.log_conversation()
                        exit()

                    print(
                        "\033[92m"
                        + "I'm sorry but I wasn't able to understand what you said. Please rephrase your response."
                        + "\033[0m"
                    )
                    self.log_conversation()
                    self.user_action = self.user.next(self.user_action["turn"] + 2)

            except Exception:
                self.operation_log.exception("Error in processing user action with 'and' inform condition")

        # if a question is asking for multiple inform_slots and at least one of them should be given by the user,
        # the user's input will be checked for each of them. If even of them is given, the criteria is satisfied and conversation is resumed.
        # if none of them is given, the user will be asked to repeat the answer with a different phrasing.
        elif agent_query.answer_condition == "or":
            try:
                while not any(
                    element in list(self.user_action["inform_slots"].keys()) for element in list(agent_query.answer)
                ):
                    # In case we are using the TestUser, One failure is enough to terminate the test.
                    if type(self.user) == TestUser:
                        print("\033[91m" + "Test Failed: Agent couldn't understand the utterance" + "\033[0m")
                        self.log_conversation()
                        exit()

                    print(
                        "\033[92m"
                        + "I'm sorry but I wasn't able to understand what you said. Please rephrase your response."
                        + "\033[0m"
                    )
                    self.log_conversation()
                    self.user_action = self.user.next(self.user_action["turn"] + 2)
            except Exception:
                self.operation_log.exception("Error in processing user action with 'or' inform condition")
        # The Common.RequestConcern question is the flag we're using for indicating the flow of convesation has reached its end.
        # After seeing it, we start calling the decision sub-process.
        if agent_query.id == "Common.RequestConcern":
            try:
                self.conversation_over = True
                # call decision process
                df = self.recommend_action()

                print("Auf wiedersehen!")
                self.operation_log.info("Conversation over. Decsion process finished successfully")
            except Exception:
                self.operation_log.exception("Error in calling decision process")

        # Until the conversation is over, we keep taking user's input and agent's response. If they have passed the criteria above,
        # it means the conversation is not over yet and we must ask the next question.
        if not self.conversation_over:
            try:
                self.state_tracker.update(user_action=self.user_action)

                self.agent.dialogue.manage_dialogue(
                    question_asked=agent_query, user_response=self.user_action["inform_slots"]
                )

                self.log_conversation()
                # We record user_response in this way to only record the necessary information for the decision tree.
                # The user's response may contain multiple inform_slots and some of them may not be relevant to the agent's question.
                # We only record the ones that are relevant to the agent's question.
                user_responses = {
                    key: value for key, value in self.user_action["inform_slots"].items() if key in agent_query.answer
                }
                self.slot_history.append([self.sys_action["request_slots"], user_responses])
                self.operation_log.info(f"Dialogue moving forward")

            except Exception as e:
                # If any exception happens during testing, we terminate the test
                if type(self.user) == TestUser:
                    print("\033[91m" + "Test Failed: unexpected error occured" + "\033[0m")
                    self.log_conversation()
                    self.operation_log.exception("Error in agent's dialogue maodule")
                    print(e)
                    exit()
                else:
                    self.log_conversation()
                    self.operation_log.exception("Error in agent's dialogue maodule")

    def recommend_action(self) -> DataFrame:
        conv_info_extractor = DataFrameMaker(self.slot_history, self.slot_history[1][1]["problem"])
        j = conv_info_extractor.data_processor()
        # j.to_csv("test.csv")
        dtpred = DecisionTreePred(j)
        final_result = dtpred.predictor(self.slot_history[1][1]["problem"])
        DataFrame(np.array(final_result)).to_csv("recomendation.csv")
        for f in final_result:
            print("\033[95m", f, "\033[0m", "\n")
        return j
