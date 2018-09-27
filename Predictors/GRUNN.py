from FeatureSelection.AutoEncoder import AutoEncoder
from sklearn import preprocessing
import tensorflow as tf
import numpy as np

class GRUNN:

    def __init__(self, train, test, reduce_dimensions=False):

        self.train_data = train
        self.test_data = test

        self.train_data = self.train_data.sort_values(by='popularity', ascending=False) # test data is also sorted, so in this way i can find them and discriminate them
        self.train_data = self.train_data[self.test_data.shape[0]:] # keep for train only the data that isnt contained in test
        self.train_data.reset_index(drop=True, inplace=True)

        labels = self.extractLabels()

        if reduce_dimensions:
            train_autoencoder = AutoEncoder(self.train_data)
            self.train_data = train_autoencoder.reduce_dimensions()  # num_dimensions should be bigger than 4. else it runs for 4.

            test_autoencoder = AutoEncoder(self.test_data)
            self.test_data = test_autoencoder.reduce_dimensions()

        self.train_data["label"] = labels

    def extractLabels(self):
        threshold = 10000
        labels = [0 if row["popularity"] < threshold else 1 for _, row in self.train_data.iterrows() ]
        return labels

    def preprocess(self, data):

        columns = data.columns
        columns = columns.drop(["hashtag"])
        if "label" in columns:
            columns = columns.drop(["label"])

        # normalize data
        scaler = preprocessing.MinMaxScaler()
        data[columns] = scaler.fit_transform(data[columns])


    def train(self):
        self.preprocess(self.train_data)



