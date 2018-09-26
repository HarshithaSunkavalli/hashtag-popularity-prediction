from scipy.spatial.distance import squareform, pdist
from sklearn import preprocessing
from FeatureSelection.AutoEncoder import AutoEncoder

class DBScan:

    def __init__(self, users, eps, MinPts, reduce_dimensions=False):
        if reduce_dimensions:
            autoencoder = AutoEncoder(users)
            users = autoencoder.reduce_dimensions()  # num_dimensions should be bigger than 4. else it runs for 4.

        self.userLength = users.shape[0]
        self.eps = eps
        self.MinPts = MinPts
        self.DistanceMatrix = self.__calcDistanceMatrix(users)

    def run(self):
        """
        MyDBSCAN takes a dataset `users` (a list of vectors), a threshold distance
        `eps`, and a required number of points `MinPts`.

        It will return a list of cluster labels. The label -1 means noise, and then
        the clusters are numbered starting from 1.
        """

        # This list will hold the final cluster assignment for each user in users.
        # There are two reserved values:
        #    -1 - Indicates a noise user
        #     0 - Means the user hasn't been considered yet.
        # Initially all labels are 0.
        labels = [0] * self.userLength

        # id is the ID of the current cluster.
        id = 0

        # This outer loop is just responsible for picking new seed points--a user
        # from which to grow a new cluster.
        # Once a valid seed user is found, a new cluster is created, and the
        # cluster growth is all handled by the 'expandCluster' routine.

        # For each user user in users...
        # ('user' is the index of the user, rather than the user itself.)
        for user in range(self.userLength):

            #pick only unvisited nodes
            if not (labels[user] == 0):
                continue

            # Find neighbors
            NeighborPts = self.__regionQuery(user)

            # If the number is below MinPts, this user is noise.
            if len(NeighborPts) < self.MinPts:
                labels[user] = -1
            # Otherwise, if there are at least MinPts nearby, use this user as the
            # seed for a new cluster.
            else:
                id += 1
                self.__growCluster(labels, user, NeighborPts, id)

        return labels, id

    def __growCluster(self, labels, user, NeighborPts, id):
        """
        Grow a new cluster with label `id` from the seed user `user`.

        This function searches through the dataset to find all points that belong
        to this new cluster. When this function returns, cluster `id` is complete.

        Parameters:
          `labels` - List storing the cluster labels for all dataset points
          `user`      - Index of the seed user for this new cluster
          `NeighborPts` - All of the neighbor of `user`
          `id`      - The label for this new cluster.
        """
        # Assign the cluster label to the seed user.
        labels[user] = id

        i = 0
        while i < len(NeighborPts):
            # Get the next user from the queue.
            neighbor = NeighborPts[i]

            # If neighbor was labelled NOISE make it leaf node
            if labels[neighbor] == -1:
                labels[neighbor] = id

            # Otherwise, if neighbor isn't already claimed, claim it as part of id.
            elif labels[neighbor] == 0:
                # Add neighbor to cluster id (Assign cluster label id).
                labels[neighbor] = id

                # Find all the neighbor of neighbor
                neighbors = self.__regionQuery(neighbor)

                setExpandedNeighbors = set(neighbors)
                setNeighbors = set(NeighborPts)

                if len(neighbors) >= self.MinPts:
                    NeighborPts = list(setNeighbors.union(setExpandedNeighbors))  # if they werent sets this would be an infinite loop

            i += 1

    def __regionQuery(self, user):
        """
        Find all users within distance `eps` of user `user`.
        """
        neighbors = []

        # For each user in the dataset
        for Pn in range(self.userLength):

            if self.DistanceMatrix[user, Pn] < self.eps:
                neighbors.append(Pn)

        return neighbors

    def __calcDistanceMatrix(self, users):
        """
        Use euclidean distance implementation
        :param users: users pandas dataframe
        :return: distance matrix
        """
        users = users.drop(["hashtag"], axis=1)

        #normalize data
        scaler = preprocessing.MinMaxScaler()
        users[users.columns] = scaler.fit_transform(users[users.columns])

        l = []
        for _, row in users.iterrows():
            l.append(row.tolist())

        dist = squareform(pdist(l, 'euclidean'))

        return dist