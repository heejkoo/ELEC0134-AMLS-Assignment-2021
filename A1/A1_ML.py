import os
import sys
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

dir = os.path.dirname(os.path.realpath(__file__))
main_dir = os.path.dirname(os.path.normpath(dir))
sys.path.insert(0, main_dir)

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier

from utils import data_preprocessing, set_global_determinism

# For Reproducibility
seed=42
set_global_determinism(seed)


class A1(object):
    """Task A1"""

    def __init__(self, train_set, val_set, test_set, model):

        # Task
        self.sort = 'A1'

        # Data Loader
        self.train_set = train_set
        self.train_images, self.train_labels = train_set
        self.val_set = val_set
        self.val_images, self.val_labels = val_set
        self.test_set = test_set
        self.test_images, self.test_labels = test_set

        # Training Configurations
        self.in_height = 128
        self.in_width = 128
        self.num_classes = 2

        # Path
        self.figure_path = "../results/fig/"
        self.make_directory(self.figure_path)

        # Model Selection
        self.model = model

        if self.model == 'svm':
            self.model = SVC()
            self.params = {
                "C": [0.1, 1.0, 5.0, 10.0],
                "kernel": ["linear", "rbf", "poly"]
            }

        elif self.model == 'decision':
            self.model = DecisionTreeClassifier()
            self.params = {
                "criterion": ["gini", "entropy"],
                "max_depth": [1, 2, 5, 10, 20]
            }

        elif self.model == 'knn':
            self.model = KNeighborsClassifier()
            self.params = {
                "n_neighbors": [1, 5, 10, 20],
                "weights": ['uniform', 'distance']
            }

        elif self.model == 'ada':
            self.model = AdaBoostClassifier()
            self.params = {
                "n_estimators": [1, 10, 20],
                "learning_rate": [0.1, 0.01],
                "algorithm": ['SAMME', 'SAMME.R']
            }

    def make_directory(self, path):
        """Make Directory"""
        if not os.path.exists(path):
            os.makedirs(path)

    def confusion_matrix(self):
        """Confusion Matrix"""
        conf_matrix = confusion_matrix(self.test_labels, self.pred)
        conf_matrix = pd.DataFrame(conf_matrix)

        conf_matrix = conf_matrix.rename(index={0: "Female", 1: "Male"},
                                         columns={0: "Female", 1: "Male"})

        plt.figure(num='Confusion Matrix')
        plt.title('{} {} Confusion Matrix'.format(self.sort, type(self.model).__name__), fontsize=14)

        ax = sns.heatmap(conf_matrix, fmt='d', cmap='YlGnBu', annot=True)
        ax.set_xlabel('True Class', fontsize=14)
        ax.set_ylabel('Predicted Class', fontsize=14)
        plt.savefig(os.path.join(self.figure_path, "{} {} Confusion Matrix.png".format(self.sort, type(self.model).__name__)))

    def flatten(self, images):
        """Flatten images"""
        self.images = images
        self.images = np.array(self.images, dtype=np.float32)
        self.images = self.images.reshape(self.images.shape[0], -1)
        return self.images

    def fit(self):
        """Train and Validation"""

        # Flatten Images
        self.train_images, self.val_images = self.flatten(self.train_images), self.flatten(self.val_images)

        # Train via Grid Search
        self.gridsearch = GridSearchCV(self.model, self.params)
        self.gridsearch.fit(self.train_images, self.train_labels)

        # Train Metrics
        print("Best Params : ", self.gridsearch.best_params_)
        print("Train Accuracy : {:.4f}".format(self.gridsearch.best_score_))

        # Validation and Metrics
        self.val_pred = self.gridsearch.predict(self.val_images)
        print("Validation Accuracy : {:.4f}".format(accuracy_score(self.val_labels, self.val_pred)))

        self.model_save = pickle.dumps(self.gridsearch)

    def test(self):
        """Test"""

        self.test_images = self.flatten(self.test_images)
        self.loaded_model = pickle.loads(self.model_save)
        self.pred = self.loaded_model.predict(self.test_images)

        # Print Statistics
        print("Test {}".format(str(type(self.model).__name__)))
        print("Accuracy  : {:.4f}".format(accuracy_score(self.test_labels, self.pred)))
        print("Precision : {:.4f}".format(precision_score(self.test_labels, self.pred, average="weighted")))
        print("Recall    : {:.4f}".format(recall_score(self.test_labels, self.pred, average="weighted")))
        print("F1 Score  : {:.4f}".format(f1_score(self.test_labels, self.pred, average="weighted")))

        print(classification_report(self.test_labels, self.pred, target_names=['Female', 'Male']))
        self.confusion_matrix()


if __name__ == "__main__":
    from config import *
    data_train, data_val, data_test = data_preprocessing('celeba', 'gender', landmarks=True)
    model_A1 = A1(data_train, data_val, data_test, config.model)
    acc_A1_train = model_A1.fit()
    acc_A1_test = model_A1.test()