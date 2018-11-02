from sklearn import preprocessing
from FeatureSelection.AutoEncoder import AutoEncoder
from sklearn.neighbors import NearestNeighbors
from collections import Counter
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import itertools
import random
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks

class KNN:
    F = 10  # paper uses F = 25 but there is a problem with SMOTE oversampler. label 3 has only 2 samples. min 6.
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
            elif row["popularity"] > self.F and row["popularity"] <= 2*self.F:
                labels.append(1)
            elif row["popularity"] > 2*self.F and row["popularity"] <= 4*self.F:
                labels.append(2)
            elif row["popularity"] > 4*self.F and row["popularity"] <= 8*self.F:
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

    def run(self, k=2):

        self.preprocess(self.train_data)
        train = self.train_data.drop(["hashtag", "label"], axis=1)
        labels = self.train_data["label"]

        #Oversample
        train_res, labels_res = self.oversample(train, labels)

        # Prediction model
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree')
        nbrs = nbrs.fit(train_res)
        distances, indices = nbrs.kneighbors(train_res)

        y_pred = self.find_predicted_label(indices, labels_res)
        # print statistics
        self.statistics(train_res, labels_res, y_pred, k)

        # predict labels
        test = self.test_data.drop(["hashtag"], axis=1)
        distances, indices = nbrs.kneighbors(test)
        predictedLabels = self.find_predicted_label(indices, labels_res)

        return predictedLabels

    def find_predicted_label(self, indices, labels):
        l = []
        for index_list in indices:
            temp_labels = []
            for index in index_list:
                temp_labels.append(labels[index])

            counter = Counter(temp_labels)
            most_common = counter.most_common(1)

            most_common = most_common[0][
                0]  # counter returns list of tuples. take tuple and take only label without frequency
            l.append(most_common)
        return l

    def statistics(self, train_res, labels_res, y_pred, k):
        """
        Prints micro f1 score and confusion matrix
        """
        cross_validation = True
        if cross_validation:
            f1 = self.cross_validation(train_res, labels_res, k, cv=10, scoring='micro')  # list with cv=10 elements in it
            f1 = np.mean(f1)
            print("Micro-F1 score for 10-fold cross validation on K Nearest Neighbors: ", f1)
        else:
            f1 = f1_score(labels_res, y_pred, average="micro")
            print("Micro-F1 score for K Nearest Neighbors: ", f1)

        label_names = np.unique(labels_res)
        # Compute confusion matrix
        cnf_matrix = confusion_matrix(labels_res, y_pred)
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

    def cross_validation(self, train_res, labels_res, k, cv=10, scoring='micro'):
        """
            Manual cross validation.
            Split data in train and test.
            Predict labels according to nearest neighbors.
            Find f1 score.
            Shuffle data for next k-fold validation.
        """
        l = []
        for i in range(cv):
            train, test, labels_train, labels_test = train_test_split(train_res, labels_res, test_size= 1/cv, random_state=42)
            nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree')
            nbrs = nbrs.fit(train)

            distances, indices = nbrs.kneighbors(test)

            y_pred = self.find_predicted_label(indices, labels_train)
            f1 = f1_score(labels_test, y_pred, average=scoring)
            l.append(f1)

            #shuffle train data
            temp_list = list(zip(train_res, labels_res))
            random.shuffle(temp_list)
            train_res, labels_res = zip(*temp_list)

        return l

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