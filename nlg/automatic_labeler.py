import os
from itertools import chain
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
import nltk

# nltk.download("averaged_perceptron_tagger")
# nltk.download()

#! Please remember that as the training set data gets more and more diverse, the accuracy of the classifier will decrease. Use it at your own risk.
def get_data_directories():
    """
    This fucntion scans the entire folder and returns the list of files that end in '.in' and '.out'

    psuedo code:
    1. get the current directory
    2. get the list of files in the current directory
    3. for each file in the list of files
    4. if the file ends with '.in'
    5. append the file to the list of '.in' files
    6. if the file ends with '.out'
    7. append the file to the list of '.out' files
    8. return the list of '.in' files and the list of '.out' files

    Returns:
        tuple: a tuple containing two lists. one contains '.in' files and the other '.out' files
    """
    in_files = []
    out_files = []

    for root, dirs, files in os.walk("."):
        for file in files:
            if file.endswith(".in"):
                in_files.append(os.path.join(root, file))
            elif file.endswith(".out"):
                out_files.append(os.path.join(root, file))

    return in_files, out_files


def get_word_features(sentence: str, label: str) -> list:
    """
        Get's a sentence and its corresponding labels as input.
        combines them all into a word feature.
        e.g.:
            input -> 'no i did not observe the presence of a specific pattern when taking absence from the employee'
    '       output for no:
                {
                    'word': 'no',
                    'is_first': True,
                    'is_capitalized': False,
                    'is_all_caps': False,
                    'is_all_lower': True,
                    'is_numeric': False,
                    'prev_word': '',
                    'next_word': 'i',
                    'label': 'B-negative',
                }
        Args:
            sentence (str): entire sentence in string format
            label (str): corresponding labels for each label in sentence

        Returns:
            sentence_words (list): a list of word features
    """
    sentence = sentence.split()  # split every word in the sentence
    labels = label.split()  # split every label in labels
    sentence_words = []
    nltk_tag = nltk.pos_tag(sentence)
    if labels != []:
        for word_index in range(len(sentence)):
            sentence_words.append(
                {
                    "word": sentence[word_index],
                    "is_first": word_index == 0,
                    "is_last": word_index == len(sentence) - 1,
                    "is_capitalized": sentence[word_index][0].upper() == sentence[word_index][0],
                    "is_all_caps": sentence[word_index].upper() == sentence[word_index],
                    "is_all_lower": sentence[word_index].lower() == sentence[word_index],
                    "is_numeric": sentence[word_index].isdigit(),
                    "prev_word": "" if word_index == 0 else sentence[word_index - 1],
                    "next_word": "" if word_index == len(sentence) - 1 else sentence[word_index + 1],
                    "nltk_tag": nltk_tag[word_index][1],
                    "label": labels[word_index],
                }
            )
    else:
        for word_index in range(len(sentence)):
            sentence_words.append(
                {
                    "word": sentence[word_index],
                    "is_first": word_index == 0,
                    "is_last": word_index == len(sentence) - 1,
                    "is_capitalized": sentence[word_index][0].upper() == sentence[word_index][0],
                    "is_all_caps": sentence[word_index].upper() == sentence[word_index],
                    "is_all_lower": sentence[word_index].lower() == sentence[word_index],
                    "is_numeric": sentence[word_index].isdigit(),
                    "prev_word": "" if word_index == 0 else sentence[word_index - 1],
                    "next_word": "" if word_index == len(sentence) - 1 else sentence[word_index + 1],
                    "nltk_tag": nltk_tag[word_index][1],
                }
            )

    return sentence_words


def train_random_forest_classifier(all_input_data, all_label_data, cut_off_point):
    """
    Train a RandomForest tree classifier
    Args:
        all_input_data (list): a list containing all word features
        all_label_data (list): a list containing all corresponding labels for all_input_data
        cut_off_point (int): where to cutoff for separating training and test data

    Returns:
        clf (RandomTreeClassifier)
        classifier_accuracy (float): classifier accuracy on test data
    """
    try:
        clf = Pipeline(
            [("vectorizer", DictVectorizer(sparse=False)), ("classifier", RandomForestClassifier(criterion="entropy"))]
        )
        print("*" * 42)
        print("-> Starting RandomForest model training")
        clf.fit(all_input_data[:cut_off_point], all_label_data[:cut_off_point])
        print("-> RandomForest model training Finished")
        classifier_accuracy = clf.score(all_input_data[cut_off_point:], all_label_data[cut_off_point:])
        print("--> RandomForest model accuracy: ", classifier_accuracy)
        return clf, classifier_accuracy
    except Exception as e:
        print(e)


def train_decision_tree_classifier(all_input_data, all_label_data, cut_off_point):
    """
    Train a Decision tree classifier
    Args:
        all_input_data (list): a list containing all word features
        all_label_data (list): a list containing all corresponding labels for all_input_data
        cut_off_point (int): where to cutoff for separating training and test data

    Returns:
        clf (DecisionTreeClassifier)
        classifier_accuracy (float): classifier accuracy on test data
    """
    try:
        clf = Pipeline(
            [("vectorizer", DictVectorizer(sparse=False)), ("classifier", DecisionTreeClassifier(criterion="entropy"))]
        )
        print("*" * 42)
        print("-> Starting DecisionTree model training")
        clf.fit(all_input_data[:cut_off_point], all_label_data[:cut_off_point])
        print("-> DecisionTree model training Finished")
        classifier_accuracy = clf.score(all_input_data[cut_off_point:], all_label_data[cut_off_point:])
        print("--> DecisionTree model accuracy: ", classifier_accuracy)
        return clf, classifier_accuracy
    except Exception as e:
        print(e)


def classify_new_data(classifier, new_data: str) -> None:
    """
    Classifies new input data and save them into pred_results.out
    Args:
        classifier: classifier object (either RandomTree or DecisionTree)
        new_data (str): the string to be labeled
    """
    label = ""
    sentence_words = get_word_features(new_data, label)
    prediction = list(classifier.predict(sentence_words))
    print(prediction)


def get_better_classifier():
    inputs_dir, labels_dir = get_data_directories()  # find out where data is
    inputs = [open(input_data_dir, "r").readlines() for input_data_dir in inputs_dir]  # read input files
    labels = [open(label_data_dir, "r").readlines() for label_data_dir in labels_dir]  # read label files

    words = [
        [get_word_features(sentence, label) for sentence, label in zip(input_sentence, label_sentence)]
        for input_sentence, label_sentence in zip(inputs, labels)
    ]  # get word features for every word that we have

    # unwrap the nested list
    data = list(chain.from_iterable(list(chain.from_iterable(words))))
    # separate labels from the word features
    labels = [word.pop("label") for word in data]
    print(data[0])
    # set the limit for training, test cutoff point
    cut_off_point = int(0.75 * len(data))

    decision_tree_classifier, decision_tree_classifier_accuracy = train_decision_tree_classifier(
        data, labels, cut_off_point
    )
    random_tree_classifier, random_forest_classifier_accuracy = train_random_forest_classifier(
        data, labels, cut_off_point
    )

    better_classfier = (
        decision_tree_classifier
        if decision_tree_classifier_accuracy > random_forest_classifier_accuracy
        else random_tree_classifier
    )

    return better_classfier


if __name__ == "__main__":
    better_classifier = get_better_classifier()
    classify_new_data(better_classifier, "hello my employee is not performing very good and he is absent today")
    classify_new_data(better_classifier, "i want to give john a promotion")
    classify_new_data(better_classifier, "i have a problem with an employee he has been at the company for 17 years")
    classify_new_data(better_classifier, "paul has been working here for 15 months")
    classify_new_data(better_classifier, "im not seeing a pattern here")
    classify_new_data(better_classifier, "hi pam i have an employee who was absent from work without permission")
    classify_new_data(better_classifier, "lack of respect")
