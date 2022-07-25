import pandas as pd
import json


class DataFrameMaker:
    def __init__(self, data, problem_category) -> None:
        self.data = data
        self.problem_category = problem_category
        self.tree_architecture = json.load(open("data/decision_tree/{}.json".format(self.problem_category)))

    def pre_process(self):
        mappings = self.tree_architecture["Mapping"]
        data = []
        for conversation_tuple_index, conversation_tuple in enumerate(self.data):
            agent_question = list(conversation_tuple[0].keys())[0]
            for key, value in mappings.items():
                if agent_question.split(".")[-1] in value:
                    temp = {
                        key: " ".join(conversation_tuple[1].values())
                        if len(conversation_tuple[1]) > 1
                        else list(conversation_tuple[1].values())[0]
                    }
                    data.append(temp)
        self.data = data

    def data_processor(self):
        self.pre_process()

        data_frame = {data_frame_column: 0 for data_frame_column in self.tree_architecture["DataFrameColumns"].keys()}
        data_frame = pd.DataFrame([data_frame])

        binary_columns = [
            binary_column
            for binary_column in self.tree_architecture["DataFrameColumns"].keys()
            if self.tree_architecture["DataFrameColumns"][binary_column] == "Binary"
        ]
        number_columns = [
            number_column
            for number_column in self.tree_architecture["DataFrameColumns"].keys()
            if self.tree_architecture["DataFrameColumns"][number_column] == "Number"
        ]
        keyword_columns = [
            keyword_column
            for keyword_column in self.tree_architecture["DataFrameColumns"].keys()
            if keyword_column not in binary_columns and keyword_column not in number_columns
        ]

        for conv in self.data:
            for key, value in conv.items():
                if key in binary_columns:
                    if value == "yes":
                        data_frame[key] = 1
                    else:
                        data_frame[key] = 0

                elif key in number_columns:
                    if key == "experience":
                        data_frame["experience"] = self.experience(self.convert_calender_to_number(value))
                    elif key == "indemnity":
                        data_frame["indemnity"] = self.indemnity(self.convert_calender_to_number(value))
                    elif key == "company_damage":
                        data_frame["company_damage"] = int(value)
                    else:
                        value = self.replace_numword_num(value)
                        data_frame[key] = self.convert_calender_to_number(value)

                elif key in keyword_columns:
                    possible_values = list(self.tree_architecture["DataFrameColumns"][key].keys())
                    for possible_value in possible_values:
                        if value in self.tree_architecture["KeyWords"][possible_value]:
                            data_frame[key] = [self.tree_architecture["DataFrameColumns"][key][possible_value]]
                            break

                # corner cases
                else:
                    if key == "other_sanctions":
                        for j in value.split():
                            if j in self.tree_architecture["KeyWords"]["suspension"]:
                                data_frame["other_sanctions_suspension"] = 1

                            elif j in self.tree_architecture["KeyWords"]["verbal"]:
                                data_frame["other_sanctions_verbal"] = 1

                            elif j in self.tree_architecture["KeyWords"]["written"]:
                                data_frame["other_sanctions_written"] = 1

                    elif key == "current_sanction" or key == "current_sanctions":
                        for j in value.split():
                            if j in self.tree_architecture["KeyWords"]["suspension"]:
                                data_frame["current_sanctions_suspension"] = 1

                            elif j in self.tree_architecture["KeyWords"]["verbal"]:
                                data_frame["current_sanctions_verbal"] = 1

                            elif j in self.tree_architecture["KeyWords"]["written"]:
                                data_frame["current_sanctions_written"] = 1

        return data_frame

    def convert_calender_to_number(self, s):
        numbs = [int(n) for n in s.split() if n.isdigit()]
        if "month" in s or "months" in s:
            if not numbs:
                return 30
            else:
                return numbs[0] * 30
        if "year" in s or "year" in s:
            if not numbs:
                return 365
            else:
                return numbs[0] * 365
        if "week" in s or "weeks" in s:
            if not numbs:
                return 7
            else:
                return numbs[0] * 7

        if "hour" in s or "hours" in s:
            if not numbs:
                return 60
            else:
                return numbs[0] * 60
        else:
            return numbs[0]

    def text2int(self, textnum, numwords={}):
        if not numwords:
            units = [
                "zero",
                "one",
                "two",
                "three",
                "four",
                "five",
                "six",
                "seven",
                "eight",
                "nine",
                "ten",
                "eleven",
                "twelve",
                "thirteen",
                "fourteen",
                "fifteen",
                "sixteen",
                "seventeen",
                "eighteen",
                "nineteen",
            ]

            tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]

            scales = ["hundred", "thousand", "million", "billion", "trillion"]

            numwords["and"] = (1, 0)
            for idx, word in enumerate(units):
                numwords[word] = (1, idx)
            for idx, word in enumerate(tens):
                numwords[word] = (1, idx * 10)
            for idx, word in enumerate(scales):
                numwords[word] = (10 ** (idx * 3 or 2), 0)

        ordinal_words = {"first": 1, "second": 2, "third": 3, "fifth": 5, "eighth": 8, "ninth": 9, "twelfth": 12}
        ordinal_endings = [("ieth", "y"), ("th", "")]

        textnum = textnum.replace("-", " ")

        current = result = 0
        for word in textnum.split():
            if word in ordinal_words:
                scale, increment = (1, ordinal_words[word])
            else:
                for ending, replacement in ordinal_endings:
                    if word.endswith(ending):
                        word = "%s%s" % (word[: -len(ending)], replacement)

                if word not in numwords:
                    raise Exception("Illegal word: " + word)

                scale, increment = numwords[word]

            current = current * scale + increment
            if scale > 100:
                result += current
                current = 0

        return result + current

    def replace_numword_num(self, string):
        dictionary = {
            "one": "1",
            "two": "2",
            "three": "3",
            "four": "4",
            "five": "5",
            "six": "6",
            "seven": "7",
            "eight": "8",
            "nine": "9",
            "ten": "10",
        }
        for key in dictionary.keys():
            string = string.lower().replace(key, dictionary[key])
        return string

    def indemnity(self, experience_days):

        if experience_days in range(91):
            return 0
        if experience_days in range(91, 361):
            return 1
        if experience_days in range(361, 1801):
            return 2
        if experience_days in range(1801, 3601):
            return 3
        else:
            return 4

    def experience(self, experience_days):
        if experience_days in range(720):
            return 0
        else:
            return 1


if __name__ == "__main__":
    from decision_tree_pred import DecisionTreePred

    sample = [
        [{"Common.RequestProblem": "UNK"}, {"problem": "absent"}],
        [{"Common.RequestProblemConfirmation": "UNK"}, {"problem": "absence"}],
        [{"Common.RequestConfirmProblem": "UNK"}, {"problem": "unauthorized absence"}],
        [{"Absence.RequestName": "UNK"}, {"empname": "paul stevenson"}],
        [{"Common.RequestSeniority": "UNK"}, {"time": "months"}],
        [{"Common.RequestContractType": "UNK"}, {"contract": "temporary"}],
        [{"Common.RequestWorkPosition": "UNK"}, {"position": "labourer"}],
        [{"Absence.RequestAbsenceDay": "UNK"}, {"weekday": "saturday"}],
        [{"Common.RequestIsPolicy": "UNK"}, {"negative": "no"}],
        [{"Absence.RequestIsFirstAbsence": "UNK"}, {"negative": "no"}],
        [{"Absence.RequestIsJustification": "UNK"}, {"negative": "no"}],
        [{"Absence.RequestIsPattern": "UNK"}, {"negative": "no"}],
        [{"Absence.RequestDamageScale": "UNK"}, {"number": "5"}],
        [{"Absence.RequestIsTeamImpact": "UNK"}, {"positive": "yes"}],
        [{"Common.RequestIsDisciplinaryRecord": "UNK"}, {"negative": "no"}],
        [{"Common.RequestIsPerformance": "UNK"}, {"negative": "no"}],
        [{"Absence.RequestIsPersonalMatters": "UNK"}, {"negative": "no"}],
        [{"Absence.RequestIsCompanyPastPractice": "UNK"}, {"negative": "no"}],
        [{"Absence.RequestIsLaborLawReasons": "UNK"}, {"negative": "no"}],
        [{"Absence.RequestIsReasons": "UNK"}, {"negative": "no"}],
        [{"Common.RequestGoal": "UNK"}, {"goal": "terminate"}],
    ]

    sample2 = [
        [{"Common.RequestProblem": "UNK"}, {"problem": "absent"}],
        [{"Common.RequestProblemConfirmation": "UNK"}, {"problem": "absence"}],
        [{"Common.RequestConfirmProblem": "UNK"}, {"problem": "unauthorized absence"}],
        [{"Absence.RequestName": "UNK"}, {"empname": "paul stevenson"}],
        [{"Common.RequestSeniority": "UNK"}, {"time": "months"}],
        [{"Common.RequestContractType": "UNK"}, {"contract": "temporary"}],
        [{"Common.RequestWorkPosition": "UNK"}, {"position": "labourer", "workplace": "factory"}],
        [{"Absence.RequestAbsenceDay": "UNK"}, {"weekday": "monday"}],
        [{"Common.RequestIsPolicy": "UNK"}, {"negative": "no"}],
        [{"Absence.RequestIsFirstAbsence": "UNK"}, {"negative": "no"}],
        [{"Absence.RequestIsJustification": "UNK"}, {"negative": "no"}],
        [{"Absence.RequestIsPattern": "UNK"}, {"negative": "no"}],
        [{"Absence.RequestDamageScale": "UNK"}, {"number": "5"}],
        [{"Absence.RequestIsTeamImpact": "UNK"}, {"positive": "yes"}],
        [{"Common.RequestIsDisciplinaryRecord": "UNK"}, {"negative": "no"}],
        [{"Common.RequestIsPerformance": "UNK"}, {"negative": "no"}],
        [{"Absence.RequestIsPersonalMatters": "UNK"}, {"negative": "no"}],
        [{"Absence.RequestIsCompanyPastPractice": "UNK"}, {"negative": "no"}],
        [{"Absence.RequestIsLaborLawReasons": "UNK"}, {"negative": "no"}],
        [{"Absence.RequestIsReasons": "UNK"}, {"negative": "no"}],
        [{"Common.RequestGoal": "UNK"}, {"goal": "terminate"}],
    ]
    dummy = DataFrameMaker(sample, "absence")
    x = dummy.data_processor()
    t = DecisionTreePred(x)
    print(t.predictor_absence())
    x.to_csv("l.csv", index=False)
