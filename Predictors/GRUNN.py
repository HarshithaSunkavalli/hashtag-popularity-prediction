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
        threshold = 20
        labels = [0 if row["popularity"] < threshold else 1 for _, row in self.train_data.iterrows() ]
        return labels

    def preprocess(self, data):

        columns = data.columns
        columns = columns.drop(["hashtag"])
        if "label" in columns:
            columns = columns.drop(["label"])

        # normalize data
        scaler = preprocessing.MinMaxScaler()
        data[columns] = scaler.fit_transform(data[columns].astype("float64"))

    def next_batch(self, batch_size):

        #fetch random batch
        batch_data = []
        batch_labels = []
        for _ in range(batch_size):
            index = np.random.randint(0, len(self.timeData))
            batch_data.append(self.timeData[index])
            batch_labels.append(self.timeDataLabels[index])

        batch_data = np.array(batch_data)
        batch_labels = np.array(batch_labels)

        batch_data = np.reshape(batch_data, (batch_size, 1))
        batch_labels = np.reshape(batch_labels, (batch_size, 1))

        dummy = batch_data.reshape(-1, batch_size, 1).shape

        return batch_data.reshape(-1, batch_size, 1), batch_labels

    def train(self):

        num_inputs = 1
        num_neurons = 100
        num_outputs = 1
        learning_rate = 0.0001
        epochs = 2000
        BATCH_SIZE = 50

        X = tf.placeholder(tf.float32, shape=(None, None, num_inputs))
        y = tf.placeholder(tf.float32, shape=(None, num_outputs))


        #RNN CELL LAYER
        cell = tf.contrib.rnn.GRUCell(num_units=num_neurons, activation=tf.nn.relu)
        cell = tf.contrib.rnn.OutputProjectionWrapper(cell, output_size=num_outputs)

        outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
        outputs = tf.nn.softmax(outputs)

        loss = tf.reduce_mean(tf.square(outputs -y))

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train = optimizer.minimize(loss)

        #create dataset

        self.timeData = self.train_data["lifespan"].values
        self.timeDataLabels = self.train_data["label"].values

        init = tf.global_variables_initializer()

        # SESSION
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.85)  # for a GPU Bug
        saver = tf.train.Saver()

        with tf.Session(config= tf.ConfigProto(gpu_options=gpu_options)) as sess:
            sess.run(init)

            for epoch in range(epochs):
                X_batch, y_batch = self.next_batch(BATCH_SIZE)

                sess.run(train, feed_dict={X: X_batch, y: y_batch})

                if epoch % 100 == 0:
                    mse = loss.eval(feed_dict={X: X_batch, y: y_batch})
                    print(epoch, "\tMSE", mse)

            saver.save(sess, "./rnn_time_series_model")

        return self.train_data["label"]




