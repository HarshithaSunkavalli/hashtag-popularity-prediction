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
        threshold = 100
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

        num_inputs = 1
        num_neurons = 100
        num_outputs = 1
        learning_rate = 0.0001
        epochs = 2000
        batch_size = 1

        X = tf.placeholder(tf.float32, shape=(None, num_inputs))
        y = tf.placeholder(tf.float32, shape=(None, num_outputs))


        #RNN CELL LAYER
        cell = tf.contrib.rnn.BasicRNNCell(num_units=num_neurons, activation=tf.nn.relu)
        cell = tf.contrib.rnn.OutputProjectionWrapper(cell, output_size=num_outputs)

        outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

        loss = tf.reduce_mean(tf.square(outputs -y))

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train = optimizer.minimize(loss)

        #create dataset
        BATCH_SIZE = 10
        self.timeData = self.train_data["lifespan"].values
        self.timeDataLabels = self.train_data["label"].values
        self.train_dataset = tf.data.Dataset.from_tensor_slices((self.timeData, self.timeDataLabels))
        self.train_dataset = self.train_dataset.shuffle(buffer_size=10000) # randomly shuffle data
        self.train_dataset = self.train_dataset.batch(BATCH_SIZE) # batch data
        self.train_dataset = self.train_dataset.repeat() # repeat indefinitely. control mannualy with epochs



        return self.train_data["label"]



