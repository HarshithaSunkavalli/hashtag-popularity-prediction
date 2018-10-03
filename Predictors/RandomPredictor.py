from sklearn import preprocessing
from FeatureSelection.AutoEncoder import AutoEncoder
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import itertools
import random

class RandomPredictor:
    F = 25

    def __init__(self, train, test, reduce_dimensions=False):
        self.train_data = train
        self.test_data = test
        labels = self.extractLabels(self.train_data)

        if reduce_dimensions:
            train_autoencoder = AutoEncoder(self.train_data)
            self.train_data = train_autoencoder.reduce_dimensions()  # num_dimensions should be bigger than 4. else it runs for 4.

            test_autoencoder = AutoEncoder(self.test_data)
            self.test_data = test_autoencoder.reduce_dimensions()

        self.train_data["label"] = labels

    def extractLabels(self, data):
        """
        assign a label to every hashtag according to 5 popularity buckets
        """
        labels = []
        for _, row in data.iterrows():
            if row["popularity"] <= self.F:
                labels.append(0)
            elif row["popularity"] > self.F and row["popularity"] <= 2 * self.F:
                labels.append(1)
            elif row["popularity"] > 2 * self.F and row["popularity"] <= 4 * self.F:
                labels.append(2)
            elif row["popularity"] > 4 * self.F and row["popularity"] <= 8 * self.F:
                labels.append(3)
            else:
                labels.append(4)

        return labels

    def preprocess(self, data):

        columns = data.columns
        columns = columns.drop(["hashtag"])
        if "label" in columns:
            columns = columns.drop(["label"])

        # normalize data
        scaler = preprocessing.MinMaxScaler()
        data[columns] = scaler.fit_transform(data[columns].astype("float64"))

    def run(self):
        """
        Implementation of Logistic regression algorithm.
        """

        self.preprocess(self.train_data)
        labels = self.train_data["label"]

        test = self.test_data.drop(["hashtag"], axis=1)
        y_pred = [random.choice(labels) for _ in range(len(labels))]

        f1 = f1_score(labels, y_pred, average="micro")
        print("Micro-F1 score for Random Predictor: ", f1)

        label_names = np.unique(labels)
        # Compute confusion matrix
        cnf_matrix = confusion_matrix(labels, y_pred)
        np.set_printoptions(precision=2)

        # Plot non-normalized confusion matrix
        plt.figure()
        self.plot_confusion_matrix(cnf_matrix, classes=label_names,
                              title='Confusion matrix, without normalization')

        # Plot normalized confusion matrix
        # plt.figure()
        # self.plot_confusion_matrix(cnf_matrix, classes=label_names, normalize=True,
        #                       title='Normalized confusion matrix')

        plt.show()

        predictedLabels = [random.choice(labels) for _ in range(len(test))]

        return predictedLabels

    def plot_confusion_matrix(self, cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()