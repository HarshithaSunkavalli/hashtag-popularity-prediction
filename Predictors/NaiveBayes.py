from sklearn import preprocessing
from FeatureSelection.AutoEncoder import AutoEncoder

class NaiveBayes:

    def __init__(self, train, test, reduce_dimensions=False):
        self.train_data = train
        self.test_data = test
        #labels = self.extractLabels()

        if reduce_dimensions:
            train_autoencoder = AutoEncoder(self.train_data)
            self.train_data = train_autoencoder.reduce_dimensions()  # num_dimensions should be bigger than 4. else it runs for 4.

            test_autoencoder = AutoEncoder(self.test_data)
            self.test_data = test_autoencoder.reduce_dimensions()

        #self.train_data["label"] = labels



    def preprocess(self, data):

        columns = data.columns
        columns = columns.drop(["hashtag"])
        if "label" in columns:
            columns = columns.drop(["label"])

        # normalize data
        scaler = preprocessing.MinMaxScaler()
        data[columns] = scaler.fit_transform(data[columns])