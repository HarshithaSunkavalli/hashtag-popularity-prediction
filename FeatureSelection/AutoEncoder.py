import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

class AutoEncoder:

    def __init__(self, csv="features.csv"):
        self.data = pd.read_csv(csv)
        self.hashtags = self.data["hashtag"].values

    def reduce_dimensions(self, num_dimensions=4):
        """
        Dimension will contain lifespan and created_at without modification,
        Giving num_dimensions = 4 it means that the returned dimensions will be lifespan, created_at and 2 that emerge from dimensionality reduction
        :param num_dimensions: the number of dimensions the feature vector will be reduced to. Default 2
        :return: the reduced feature vector
        """
        if num_dimensions <= 4:
            num_dimensions = 4

        features= self.sanitize_features()

        values = features.values
        #scale data
        scaler = MinMaxScaler()
        scaled_values = scaler.fit_transform(values)


        num_inputs = scaled_values.shape[1]
        num_hidden = num_dimensions - 2 #lifespan and created_at will be included at the end
        num_outputs = num_inputs

        learning_rate = 0.01
        num_steps = 1000

        X = tf.placeholder(tf.float32, shape=[None, num_inputs])
        hidden = fully_connected(X, num_hidden, activation_fn=None) # no activation function because we are implementing a linear autoencoder
        outputs = fully_connected(hidden, num_outputs, activation_fn=None)

        loss = tf.reduce_mean(tf.square(outputs - X))
        optimizer = tf.train.AdamOptimizer(learning_rate)

        train = optimizer.minimize(loss)

        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)

            temp = []
            for iteration in range(num_steps):
                sess.run(train, feed_dict={X: scaled_values})
                if iteration%100 == 0:
                    temp.append(loss.eval(feed_dict={X: scaled_values}))

            output = hidden.eval(feed_dict={X: scaled_values})

        # import matplotlib.pyplot as plt
        # import matplotlib as mpl
        # mpl.style.use("seaborn")
        # plt.plot(temp[:], 'go-')
        # plt.title("{} - {} Linear PCA AutoEncoder".format(len(self.data.columns) -1, num_dimensions))
        # plt.xlabel("100-iteration step")
        # plt.ylabel("Loss")
        # plt.show()

        result = []
        for i, list in enumerate(output):
            list = list.tolist()
            list.append(self.data.loc[i, "created_at"])
            list.append(self.data.loc[i, "lifespan"])
            result.append(list)

        return result

    def sanitize_features(self):
        values = self.data.drop(["hashtag","created_at","lifespan"], axis=1)
        return  values

