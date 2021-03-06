from sklearn import preprocessing
from FeatureSelection.AutoEncoder import AutoEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import numpy as np
import itertools
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks

class LR:
    F = 10 #paper uses F = 25 but there is a problem with SMOTE oversampler. label 3 has only 2 samples. min 6.

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

    def oversample(self, train, labels):
        """
            Over samples data according to SMOTE algorithm
        """
        #Oversample
        sm = SMOTE(random_state=2)
        train_res, labels_res = sm.fit_sample(train, labels)

        #clear noise points that emerged from oversampling
        tl = TomekLinks(random_state=42)
        train_res, labels_res = tl.fit_sample(train_res, labels_res)

        return train_res, labels_res

    def run(self):
        """
        Implementation of Logistic regression algorithm.
        """

        self.preprocess(self.train_data)

        train = self.train_data.drop(["hashtag", "label"], axis=1)
        labels = self.train_data["label"]

        #Oversample
        train_res, labels_res = self.oversample(train, labels)

        #Prediction model
        clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial', max_iter=1000)
        clf = clf.fit(train_res, labels_res)

        y_pred = clf.predict(train_res)

        #print statistics
        self.statistics(clf, train_res, labels_res, y_pred)

        #predict labels
        test = self.test_data.drop(["hashtag"], axis=1)
        predictedLabels = clf.predict(test)

        return predictedLabels

    def statistics(self, clf, train_res, labels_res, y_pred):
        """
        Prints micro f1 score and confusion matrix
        """

        cross_validation = True
        if cross_validation:
            f1 = cross_val_score(clf, train_res, labels_res, cv=10,
                                 scoring='f1_micro')  # list with cv=10 elements in it
            f1 = np.mean(f1)
            print("Micro-F1 score for 10-fold cross validation on Logistic Regression: ", f1)
        else:
            f1 = f1_score(labels_res, y_pred, average="micro")
            print("Micro-F1 score for Logistic Regression: ", f1)

        label_names = np.unique(labels_res)
        # Compute confusion matrix
        cnf_matrix = confusion_matrix(labels_res, y_pred)
        np.set_printoptions(precision=2)

        # Plot non-normalized confusion matrix
        plt.figure()
        self.plot_confusion_matrix(cnf_matrix, classes=label_names,
                                   title='Confusion matrix, without normalization. Micro-F1: {}'.format(f1))

        # Plot normalized confusion matrix
        # plt.figure()
        # self.plot_confusion_matrix(cnf_matrix, classes=label_names, normalize=True,
        #                       title='Normalized confusion matrix')

        plt.show()

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
        # plt.savefig("Images/ConfusionMatrices/LogisticRegression.png", format="png")