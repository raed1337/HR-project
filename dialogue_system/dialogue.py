import json
from tkinter import N

"""
This module contains the heart and soul of Hermes.
All the magic that determines which question should be asked next, happens here.
If you're stuck at what's what in here. First of all, I'm sorry. You can just debug this sinlge file to find out what's going on.
OooooooOh boy explaining this is going to be hard!
"""


class Question:
    """
    I've decided to treat questions as python objects insetead of strings from a json file or dictionaries.
    This way, I can add more functionality to questions in the future and tracing whats happening is much easier.
    """

    def __init__(
        self, question_id: str = None, question_text: str = None, answers: list = [], answer_condition: str = None
    ) -> None:
        """
        Each Question has four attributes (such and informative documentation. Much wow)
        Args:
            question_id (str, optional): This attribute determines what is the question's unique ID in the dialogues.json. Defaults to None.
            question_text (str, optional): Contains the natural language utterance of the question. Defaults to None.
            answers (list, optional): Contains the inform_slots that question needs in order to be answered. Defaults to [].
            answer_condition (str, optional): Is either 'and' or 'or'.If it is 'and' it means all inform_slots must be answered. O.w. only one of them is enough.
                Defaults to None.
        """
        self.id = question_id
        self.text = question_text
        self.answer = answers
        self.answer_condition = answer_condition


class Dialogue:
    """
    I'll exaplin everything in detail in each functon. Just remember that THE DIALOGUE MANAGES ITSELF.
    What should be asked next is NOT determined by the agent or the user.
    """

    def __init__(self) -> None:
        # the flow flag is deprecated. I'm not using it anywhere but somehow removing it breaks the code. Don't change it.
        self.flow = None
        # This is going to be list which contains the questions that should be asked.
        self.questions = None
        # flow changers are question that cahnge the order of questions.
        # for example if a user's response to a question is negative, some questions should not be asked.
        # if a question is in this list, one the agent reaches it, it will change the questions in self.questions according to user's answer.
        self.flow_changers = None
        # this dictionary contains all the dialogues that we have. it is better just to loead them all than to open the file everytime we're looking for something.
        self.all_dialogues = json.load(open("data/dialogues.json"))["dia_acts"]
        # determines which question we're at.
        self.question_num = 1
        # this is used to determine which json file should be opened.
        self.current_flow = None
        # I don't remember why I put this here. I'm afraid of removing it and breaking the code. Change it at your own risk.
        self.final_question = False

    def load_dialogue(self, dialogue_path: str, user_response: dict = None) -> None:
        """
        This function loads the dialogue from the json file.

        Args:
            dialogue_path (str): the name of json file which should be loaded. it is basically self.current_flow under another name.
            user_response (dict, optional): user's answer. Determines which subcategory to load. Defaults to None.
        """
        dialogue = json.load(open(f"data/problems/{dialogue_path}.json"))

        if not user_response:
            # sprcial case reserved only for init.json to load the first set of questions.
            # whatever you do, don't change this.
            quesiton_set = [dialogue["MainProblem"]] + dialogue["QuestionSet"]
            if dialogue["FlowChange"] != "None":
                self.flow_changers = dialogue["FlowChange"]

        else:
            # we look for the subcateogry that we should load in here based in function's inputs.
            sub_categories = list(dialogue["SubCategories"].keys())
            sub_category = [
                sub_category
                for sub_category in sub_categories
                if sub_category.upper() in user_response["problem"].upper()
            ][0]
            quesiton_set = dialogue["SubCategories"][sub_category]["QuestionSet"]
            # some dialogues don't have flow changers outright.
            if dialogue["SubCategories"][sub_category]["FlowChange"] != "None":
                self.flow_changers = dialogue["SubCategories"][sub_category]["FlowChange"]

        # make Question objects from all the question IDs that were loaded from the corresponfing JSON file.
        self.questions = [
            Question(
                question_id=question_id,
                question_text=self.all_dialogues[question_id]["nl"]["agent"],
                answers=self.all_dialogues[question_id]["inform_slots"],
                answer_condition=self.all_dialogues[question_id]["inform_condition"],
            )
            for question_id in quesiton_set
        ]

    def initialize_dialogue(self) -> list:
        """
        special function.
        Only used once to initilize the dialogue.
        Returns:
            list: list of questions that should be asked.
        """
        self.load_dialogue("init")
        self.current_flow = "init"
        return self.questions

    def manage_dialogue(self, question_asked: Question, user_response: dict) -> None:
        """
        The default behavior we're expecting is that a dialogue should not have any flow changers.
        But if it does, we chaeck for the conditions here and change the flow accordingly.
        Please, please, PLEASE if you're confused about this, just debug this file. I promise it'll help a lot.

        Args:
            question_asked (Question): The question which was asked.
            user_response (dict): user's answer to the question which was asked.
        """
        if question_asked.id in self.flow_changers.keys():
            try:
                flow_changer_keys = list(self.flow_changers[question_asked.id].keys())
                question_inform_key = [
                    flow_change_keyword
                    for flow_change_keyword in flow_changer_keys
                    if flow_change_keyword in list(user_response.keys())
                ][0]
            except IndexError:
                print(f"{user_response} is not a flow changer keyword found for {question_asked.id}")
                return
            except KeyError:
                print("Ho Lee sheet")

            # two flow changer structures are defined.
            if type(self.flow_changers[question_asked.id][question_inform_key]["flow"]) == list:
                if self.flow_changers[question_asked.id][question_inform_key]["action"] == "remove":

                    self.questions = [
                        question
                        for question in self.questions
                        if question.id not in self.flow_changers[question_asked.id][question_inform_key]["flow"]
                    ]
                elif self.flow_changers[question_asked.id] == "add":
                    self.questions += [
                        Question(
                            question_id=question_id,
                            question_text=self.all_dialogues[question_id]["nl"]["agent"],
                            answers=self.all_dialogues[question_id]["inform_slots"],
                            answer_condition=self.all_dialogues[question_id]["inform_condition"],
                        )
                        for question_id in self.flow_changers[question_asked.id][question_inform_key]["flow"]
                    ]

            elif type(self.flow_changers[question_asked.id][question_inform_key]["flow"]) == dict:
                if self.flow_changers[question_asked.id][question_inform_key]["action"] == "remove":
                    self.questions = [
                        question
                        for question in self.questions
                        if question.id
                        not in self.flow_changers[question_asked.id][question_inform_key]["flow"][
                            user_response[question_inform_key]
                        ]
                    ]
                elif self.flow_changers[question_asked.id][question_inform_key]["action"] == "add":
                    self.questions += [
                        Question(
                            question_id=question_id,
                            question_text=self.all_dialogues[question_id]["nl"]["agent"],
                            answers=self.all_dialogues[question_id]["inform_slots"],
                            answer_condition=self.all_dialogues[question_id]["inform_condition"],
                        )
                        for question_id in self.flow_changers[question_asked.id][question_inform_key]["flow"][
                            user_response[question_inform_key]
                        ]
                    ]

            elif type(self.flow_changers[question_asked.id][question_inform_key]["flow"]) == str:
                if self.flow_changers[question_asked.id][question_inform_key]["action"] == "change_topic":
                    self.current_flow = self.flow_changers[question_asked.id][question_inform_key]["flow"]
                    self.load_dialogue(self.current_flow, user_response)
                    return


if __name__ == "__main__":
    d = Dialogue()
    d.initialize_dialogue()
    d.load_dialogue("absence", "Unauthorized Absence")
