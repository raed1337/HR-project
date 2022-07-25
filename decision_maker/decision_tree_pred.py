from operator import itemgetter
import pickle
import pandas as pd
import json
from users.real_user import RealUser
from nlu import nlu


class DecisionTreePred:
    def __init__(self, dataF):
        self.dataF = dataF

    def get_decision_paths(self, decision_process: dict) -> list:
        def getpaths(d):
            if not isinstance(d, dict):
                yield [d]
            else:
                yield from ([k] + w for k, v in d.items() for w in getpaths(v))

        all_decision_paths = list(getpaths(decision_process))
        paths = []
        for decision_path in all_decision_paths:
            path = []
            for element in decision_path:
                if element != "True" and element != "False" and not isinstance(element, list):
                    path.append(element)
            if path not in paths:
                paths.append(path)

        return paths

    def predictor(self, main_problem):
        if "absence" in main_problem:
            return self.predictor_absence()
        elif "work time" in main_problem:
            return self.predictor_work_time()
        elif "ovid" in main_problem:
            return self.predictor_Covid_situation()
        elif "lack respect" in main_problem:
            return self.predictor_lack_of_respect()
        elif "insubordination" in main_problem:
            return self.predictor_insubordination()
        elif "performance" in main_problem:
            return self.predictor_performance()
        else:
            return self.predictor_leave_or_holiday()

    def recomendation_process(self, dict_recomendation, dict_text):
        data = []
        for key, value in dict_recomendation.items():
            data.append(key)
            for i in range(len(value)):
                data.append(dict_text[value[i]])
        return data

    def predictor_lack_of_respect(self) -> list:
        """
        !This function is a mess. This is not the way the decision process should have been handled.
        Why was this function developed and created this way?
        Two reasons:
            - We don't have enough case data in order to train a decision tree which can differentiate between the cases with a good accuracy.
            - At the end of these conversations, a second question has beem added. This question is dependant on data extracted from the conversation AFTER the
            conversation is finished.
        In order to solve both issues, I've designed a semi-decision tree and ask the that particular question in this function.

        #############################################################################################################################
        Tips for improving this function:
            - even though this code is ugly, it is the only logical way that comes to my mind for handling our particular problem.
            So, if you want to improve this code, i suggest keeping the decision tree structure that I defined and designing a class for it.
            That way, you will be able to create your own decision trees and use them when you think SKlearn's decision tree is not good enough.
            - as you can see, I'm using some hard-coded elements in this function such as "goal" and "experience".
            I suggest adding them as configuration parameters in the JSON file so that you can have a universal function for all the decision trees.

        Returns:
            list: list of recomendation texts
        """
        results = []
        # we need the NLU to extract info out of a user's input
        NLU = nlu()
        temp_user = RealUser(nlu_model=NLU)
        # we read the semi-decision tree here. This JSON file contains both which recomendation should be made based on which answer to a particular question,
        # and the text of those recomendations.
        decision_process_all = json.load(open("decision_maker/data/disrespect.json", "r"))
        # it is better to treat the collected data as a dcitionary that a dataframe
        conversation_dict = self.dataF.to_dict(orient="list")
        # we will put all the recommendations in this list
        recoms_list = []
        # this outlines which sub-problem should be chosen from the JSON file
        problem = str(conversation_dict["problem"][0])
        # just a simplification so we don't have to type the whole shindig again
        decision_process = decision_process_all["problem"][problem]

        # goal "2" is the code for ermination goal. if the user selects this goal,
        # a recomendation should be shown to the user based on the employee's duration of service and the question for termination should be asked again.
        if str(conversation_dict["goal"][0]) == "2":
            # experience "1" is the code for more than two years of experience
            if str(conversation_dict["experience"][0]) == "1":
                print("\033[95m", decision_process_all["text_mappings"][problem]["1-4.3-1"], "\033[0m")
            else:
                print("\033[95m", decision_process_all["text_mappings"][problem]["1-4.3-2"], "\033[0m")
            # ask the dreaded question here
            print(
                "\033[91m"
                + "Following my recommendation, what do you want to do now? \n- Proceed with the progressive disciplinary sanctions \n- Terminate the employment "
                + "\033[0m"
            )
            # extract info from user's repsonse
            response = input("\033[1;34m" + "+ User: " + "\033[0m")
            final_goal = temp_user.generate_diaact_from_nl(response)["inform_slots"]["goal"]
            # load the appropriate recomendations
            if final_goal == "terminate":
                conversation_dict["final_goal_terminate"] = [1]
                conversation_dict["final_goal_correct"] = [0]
            else:
                conversation_dict["final_goal_terminate"] = [0]
                conversation_dict["final_goal_correct"] = [1]
        # if the user did not intend to terminate the employee in the first place we don't ask any quesitons
        else:
            conversation_dict["final_goal_terminate"] = [0]
            conversation_dict["final_goal_correct"] = [1]
        # fill the reconmendations according to the decision process
        for decision_key, decision_sub_key in decision_process.items():
            if any(element in list(decision_process[decision_key].keys()) for element in ["True", "False"]):
                if conversation_dict[decision_key][0] == 1:
                    recoms_list.append(decision_process[decision_key]["True"])
                if conversation_dict[decision_key][0] == 0:
                    recoms_list.append(decision_process[decision_key]["False"])
            else:
                if conversation_dict[decision_key][0] == 1:
                    for key, value in decision_sub_key.items():
                        if conversation_dict[key][0] == 1:
                            recoms_list.append(decision_process[decision_key][key]["True"])
                        if conversation_dict[key][0] == 0:
                            recoms_list.append(decision_process[decision_key][key]["False"])
        # get the actual recomendation texts
        recoms_list = [item for item in recoms_list if item != []]
        for items in sorted(recoms_list, key=itemgetter(0)):
            for item in items:
                results.append(decision_process_all["text_mappings"][problem][item])

        return results

    def predictor_insubordination(self):
        """
        !This function is a mess. This is not the way the decision process should have been handled.
        Why was this function developed and created this way?
        Two reasons:
            - We don't have enough case data in order to train a decision tree which can differentiate between the cases with a good accuracy.
            - At the end of these conversations, a second question has beem added. This question is dependant on data extracted from the conversation AFTER the
            conversation is finished.
        In order to solve both issues, I've designed a semi-decision tree and ask the that particular question in this function.

        #############################################################################################################################
        Tips for improving this function:
            - even though this code is ugly, it is the only logical way that comes to my mind for handling our particular problem.
            So, if you want to improve this code, i suggest keeping the decision tree structure that I defined and designing a class for it.
            That way, you will be able to create your own decision trees and use them when you think SKlearn's decision tree is not good enough.
            - as you can see, I'm using some hard-coded elements in this function such as "goal" and "experience".
            I suggest adding them as configuration parameters in the JSON file so that you can have a universal function for all the decision trees.

        Returns:
            list: list of recomendation texts
        """
        results = []
        # we need the NLU to extract info out of a user's input
        NLU = nlu()
        temp_user = RealUser(nlu_model=NLU)
        # we read the semi-decision tree here. This JSON file contains both which recomendation should be made based on which answer to a particular question,
        # and the text of those recomendations.
        decision_process_all = json.load(open("decision_maker/data/insubordination.json", "r"))
        # it is better to treat the collected data as a dcitionary that a dataframe
        conversation_dict = self.dataF.to_dict(orient="list")
        # we will put all the recommendations in this list
        recoms_list = []
        # this outlines which sub-problem should be chosen from the JSON file
        problem = str(conversation_dict["problem"][0])
        # just a simplification so we don't have to type the whole shindig again
        decision_process = decision_process_all["problem"][problem]
        decision_paths = self.get_decision_paths(decision_process)
        # goal "2" is the code for ermination goal. if the user selects this goal,
        # a recomendation should be shown to the user based on the employee's duration of service and the question for termination should be asked again.
        if str(conversation_dict["goal"][0]) == "2":
            # experience "1" is the code for more than two years of experience
            if str(conversation_dict["experience"][0]) == "1":
                print("\033[95m", decision_process_all["text_mappings"][problem]["1-4.3-1"], "\033[0m")
            else:
                print("\033[95m", decision_process_all["text_mappings"][problem]["1-4.3-2"], "\033[0m")
            # ask the dreaded question here
            print(
                "\033[91m"
                + "Following my recommendation, what do you want to do now? \n- Proceed with the progressive disciplinary sanctions \n- Terminate the employment "
                + "\033[0m"
            )
            # extract info from user's repsonse
            response = input("\033[1;34m" + "+ User: " + "\033[0m")
            final_goal = temp_user.generate_diaact_from_nl(response)["inform_slots"]["goal"]
            # load the appropriate recomendations
            if final_goal == "terminate":
                conversation_dict["final_goal_terminate"] = [1]
                conversation_dict["final_goal_correct"] = [0]
            else:
                conversation_dict["final_goal_terminate"] = [0]
                conversation_dict["final_goal_correct"] = [1]
        # if the user did not intend to terminate the employee in the first place we don't ask any quesitons
        else:
            conversation_dict["final_goal_terminate"] = [0]
            conversation_dict["final_goal_correct"] = [1]
        # fill the reconmendations according to the decision process
        recoms_list = []
        for decision_path in decision_paths:
            d_p = decision_process
            decision_key = ""
            while len(decision_path) > 0:
                decision_key = decision_path.pop(0)
                if int(conversation_dict[decision_key][0]):
                    d_p = d_p[decision_key]
                else:
                    recoms_list.append(d_p[decision_key]["False"])
                    break

                if "True" in d_p.keys():
                    if int(conversation_dict[decision_key][0]):
                        recoms_list.append(d_p["True"])
                    else:
                        recoms_list.append(d_p["False"])
        # get the actual recomendation texts
        recoms_list = [x for x in recoms_list if x != []]

        for items in sorted(recoms_list, key=itemgetter(0)):
            if items != []:
                for item in items:
                    results.append(decision_process_all["text_mappings"][problem][item])

        return results

    def predictor_performance(self):
        """
        !This function is a mess. This is not the way the decision process should have been handled.
        Why was this function developed and created this way?
        Two reasons:
            - We don't have enough case data in order to train a decision tree which can differentiate between the cases with a good accuracy.
            - At the end of these conversations, a second question has beem added. This question is dependant on data extracted from the conversation AFTER the
            conversation is finished.
        In order to solve both issues, I've designed a semi-decision tree and ask the that particular question in this function.

        #############################################################################################################################
        Tips for improving this function:
            - even though this code is ugly, it is the only logical way that comes to my mind for handling our particular problem.
            So, if you want to improve this code, i suggest keeping the decision tree structure that I defined and designing a class for it.
            That way, you will be able to create your own decision trees and use them when you think SKlearn's decision tree is not good enough.
            - As you can see, I'm using some hard-coded elements in this function such as "goal" and "experience".
            I suggest adding them as configuration parameters in the JSON file so that you can have a universal function for all the decision trees.

        Returns:
            list: list of recomendation texts
        """
        results = []
        # we need the NLU to extract info out of a user's input
        NLU = nlu()
        temp_user = RealUser(nlu_model=NLU)
        # we read the semi-decision tree here. This JSON file contains both which recomendation should be made based on which answer to a particular question,
        # and the text of those recomendations.
        decision_process_all = json.load(open("decision_maker/data/performance.json", "r"))
        # it is better to treat the collected data as a dcitionary that a dataframe
        conversation_dict = self.dataF.to_dict(orient="list")
        # we will put all the recommendations in this list
        recoms_list = []
        # this outlines which sub-problem should be chosen from the JSON file
        problem_num = str(conversation_dict["problem"][0])
        # just a simplification so we don't have to type the whole shindig again
        decision_process = decision_process_all["problem"][problem_num]
        # goal "2" is the code for termination goal. if the user selects this goal,
        # a recomendation should be shown to the user based on the employee's duration of service and the question for termination should be asked again.
        if str(conversation_dict["goal"][0]) == "2":
            notices = []
            notice_process = decision_process_all["notice"][problem_num]["goal"]["2"]
            for key in notice_process.keys():
                d_k = list(notice_process[key].keys())
                if d_k == ["1", "0"]:
                    if conversation_dict[key][0] == 1:
                        notices.append(notice_process[key]["1"])
                    else:
                        notices.append(notice_process[key]["0"])
                else:
                    if conversation_dict[key][0] == 1:
                        if conversation_dict["action_plan_improvement"][0] == 1:
                            notices.append(notice_process[key]["action_plan_improvement"]["1"])
                        else:
                            notices.append(notice_process[key]["action_plan_improvement"]["0"])
            # experience "1" is the code for more than two years of experience
            for notice in notices:
                if len(notice) > 1:
                    for sub_notice in notice:
                        print(decision_process_all["text_mappings"][problem_num][sub_notice] + "\n")
                else:
                    print(decision_process_all["text_mappings"][problem_num][notice[0]])

            # ask the dreaded question here
            print(
                "\033[91m"
                + "Following my recommendation, what do you want to do now? \n- Maintain employment and improve job performance \n- Terminate the employment "
                + "\033[0m"
            )
            # extract info from user's repsonse
            response = input("\033[1;34m" + "+ User: " + "\033[0m")
            final_goal = temp_user.generate_diaact_from_nl(response)["inform_slots"]["goal"]
            # load the appropriate recomendations
            if final_goal == "terminate":
                conversation_dict["final_goal_terminate"] = [1]
                conversation_dict["final_goal_correct"] = [0]
            else:
                conversation_dict["final_goal_terminate"] = [0]
                conversation_dict["final_goal_correct"] = [1]
        # if the user did not intend to terminate the employee in the first place we don't ask any quesitons
        else:
            conversation_dict["final_goal_terminate"] = [0]
            conversation_dict["final_goal_correct"] = [1]
        # fill the reconmendations according to the decision process
        if conversation_dict["final_goal_terminate"][0] == 1:
            problem = decision_process["final_goal_terminate"]
        elif conversation_dict["final_goal_correct"][0] == 1:
            problem = decision_process["final_goal_correct"]

        recoms_list = []
        result = ""
        while not isinstance(result, list):
            keys = list(problem.keys())
            if keys == ["True", "False"]:
                if conversation_dict[key][0] == 1:
                    result = problem["True"]
                    recoms_list.append(problem["True"])
                else:
                    result = problem["False"]
                    recoms_list.append(problem["False"])
            else:
                try:
                    keys.remove("False")
                except ValueError:
                    pass
                for key in keys:
                    if conversation_dict[key][0] == 1:
                        problem = problem[key]
                    else:
                        problem = problem["False"]
        # get the actual recomendation texts
        recoms_list = [x for x in recoms_list if x != []]

        for items in sorted(recoms_list, key=itemgetter(0)):
            if items != []:
                for item in items:
                    results.append(decision_process_all["text_mappings"][problem_num][item])

        return results

    def predictor_absence(self):

        with open("decision_maker/models/decision_tree_absence.pickle", "rb") as picklefile:
            model = pickle.load(picklefile)

        pred = model.predict(self.dataF)
        result = pred[0]
        return result

    def predictor_leave_or_holiday(self):

        with open("decision_maker/models/decision_tree_model_leave_holiday.pickle", "rb") as picklefile:
            model = pickle.load(picklefile)

        with open("decision_maker/data/LeaveOrHoliday.json") as json_file:
            recomendation = json.load(json_file)

        with open("decision_maker/data/LeaveOrHolidayText.json") as json_file:
            text_recomendation = json.load(json_file)

        pred = model.predict(self.dataF)

        result = pred[0]
        result = self.recomendation_process(recomendation[result]["ENG"], text_recomendation)

        return result

    def predictor_work_time(self):

        with open("decision_maker/models/decision_tree_model_work_time.pickle", "rb") as picklefile:
            model = pickle.load(picklefile)

        with open("decision_maker/data/WorkTime.json") as json_file:
            recomendation = json.load(json_file)

        with open("decision_maker/data/WorkTimeText.json") as json_file:
            text_recomendation = json.load(json_file)

        pred = model.predict(self.dataF)

        # df = pd.read_excel('deep_dialog/Decision_Maker/clabels.xlsx')
        # dict_ =df.set_index('Key').to_dict('index')
        # result = dict_[pred[0]]['value']
        result = pred[0]
        result = self.recomendation_process(recomendation[result]["ENG"], text_recomendation)

        return result

    def predictor_Covid_situation(self):

        with open("decision_maker/models/decision_tree_covid_situation.pickle", "rb") as picklefile:
            model = pickle.load(picklefile)

        with open("decision_maker/data/CovidSituation.json") as json_file:
            recomendation = json.load(json_file)

        with open("decision_maker/data/CovidSituationText.json") as json_file:
            text_recomendation = json.load(json_file)

        pred = model.predict(self.dataF)
        result = pred[0]
        result = self.recomendation_process(recomendation[result]["ENG"], text_recomendation)

        return result
