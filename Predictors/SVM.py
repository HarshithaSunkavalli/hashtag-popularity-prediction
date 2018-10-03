from sklearn import preprocessing
from FeatureSelection.AutoEncoder import AutoEncoder
from sklearn import svm
from sklearn.metrics import f1_score

class SVM:

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

    def run(self):
        """
        Implementation of SVM algorithm. For rbf kernel it works better without dimensionality reduction and preprocessing
        """

        #self.preprocess(self.train_data)
        train = self.train_data.drop(["hashtag", "label"], axis=1)
        labels = self.train_data["label"]

        test = self.test_data.drop(["hashtag"], axis=1)

        clf = svm.SVC(kernel="rbf", gamma="scale")
        clf = clf.fit(train, labels)

        y_pred = clf.predict(train)

        f1 = f1_score(labels, y_pred, average="micro")
        print("Micro-F1 score for SVM: ", f1)

        predictedLabels = clf.predict(test)

        return predictedLabels
