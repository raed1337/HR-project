import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import pickle, os
from matplotlib import pyplot as plt


class DecisionTreeTrainer:
    def __init__(self, file):
        self.file = file

    def reader(self):
        dataset = pd.read_excel(self.file, engine="openpyxl")
        print(dataset.shape)

        return dataset

    def feature_engineering(self):

        dataset = self.reader()

        return dataset

    def model(self):
        dataset = self.feature_engineering()
        print(dataset)
        x_train = dataset.drop(["decision"], axis=1)
        print(x_train.shape)

        y_train = dataset.decision

        clf = DecisionTreeClassifier(random_state=0)
        clf = clf.fit(x_train, y_train)
        fig = plt.figure(figsize=(50, 40))
        _ = tree.plot_tree(
            decision_tree=clf, feature_names=list(dataset.columns), class_names=list(dataset.decision), filled=True
        )

        fig.savefig(
            os.path.join(os.getcwd(), "decision_maker", "decision_tree_viz") + "/decision_tree_lack_of_respect.png"
        )
        print(tree.export_text(clf))
        with open(
            os.path.join(os.getcwd(), "decision_maker", "models") + "/decision_tree_lack_of_respect.pickle", "wb"
        ) as picklefile:
            pickle.dump(clf, picklefile)


dt = DecisionTreeTrainer(
    os.path.join(os.getcwd(), "decision_maker", "training_data_archive") + "/lack_of_respect.xlsx"
)
dt.model()
