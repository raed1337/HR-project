from re import template
from users.real_user import RealUser
import pandas as pd
import os, json


class TestUser(RealUser):
    """
    The test user is just a real user instance.
    The difference between them is that the test user will go thorugh the responses from a csv file instead of getting them from the terminal.
    Think of it as a user simulator.
    """

    def load_test_data(self, main_problem: str, sub_problem: str) -> None:
        """
        This function loads the csv file containing the test data.

        Args:
            main_problem (str): the name of the main problem (absence, lack_of_rsepect, etc.).
            sub_problem (str): the name of the sub problem (unauthorized absence, incompetence, etc.).
        """
        test_data_file_path = os.path.join(
            os.getcwd(), "data", "test_dialogues", main_problem + "_" + sub_problem + ".csv"
        )
        test_file = pd.read_csv(test_data_file_path)
        self.dialogues = test_file["User Response"].to_list()
        self.dialogues = [dialogue for dialogue in self.dialogues if dialogue != " "]
        self.counter = 0

    def next(self, turn: int) -> dict:
        """
        Replaces RealUser's next function.
        Instead of going thorugh the responses from the terminal, it will go through the responses from a csv file.

        Args:
            turn (int): for keeping track of how many turns has it been.

        Returns:
            dict: NLU's output
        """

        self.state["turn"] = turn
        self.episode_over = False

        user_input = self.dialogues[self.counter]

        print("\033[1;34m" + "+ User: " + "\033[0m" + user_input)
        self.counter += 1

        response_action = self.generate_diaact_from_nl(user_input)

        response_action["turn"] = self.state["turn"]
        self.state["diaact"] = response_action["diaact"]
        self.state["inform_slots"] = response_action["inform_slots"]
        self.state["request_slots"] = response_action["request_slots"]

        self.action_history.append(response_action)

        return response_action

    def make_test_user_template(self, test_case_category: str, test_case_subcategory: str) -> None:
        """
        Makes an empty template to be fileed by the user.

        Args:
            test_case_category (str): same as main_problem in load_test_data.
            test_case_subcategory (str): same as sub_problem in load_test_data.
        """
        template_path = os.path.join(
            os.getcwd(), "data", "test_dialogues", test_case_category + "_" + test_case_subcategory + ".csv"
        )
        # load all dialogues to read their questions
        all_dialogues = json.load(open("./data/dialogues.json"))["dia_acts"]
        # load init dialogues to help user navigate to the desired subcategory.
        init_dialogues = json.load(open("./data/problems/init.json"))
        # load the subcategory dialogues.
        sub_category_dialogues = json.load(open(f"./data/problems/{test_case_category}.json"))
        # load the subcategory quesitons.
        sub_category_dialogues_all = sub_category_dialogues["SubCategories"][test_case_subcategory]["QuestionSet"]
        # build the list of questions.
        conv = (
            [
                "Common.RequestProblem",
                "Common.RequestProblemConfirmation",
            ]
            + [
                init_dialogues["FlowChange"]["Common.RequestProblemConfirmation"]["problem"]["flow"][
                    test_case_category
                ][0]
            ]
            + sub_category_dialogues_all
        )

        conv = [all_dialogues[dialogue]["nl"]["agent"] for dialogue in conv]
        # create the empty template.
        template_file = pd.DataFrame()
        template_file["Agent Question"] = conv
        template_file["User Response"] = " "
        template_file.to_csv(template_path, index=False)

        print("\033[1;35m" + "The template has been created in the following path: " + "\033[0m" + template_path)


if __name__ == "__main__":
    test_user = TestUser()

    test_user.make_test_user_template("absence", "Unauthorized Absence")
