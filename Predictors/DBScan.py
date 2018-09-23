
class Predictors:

    def __init__(self, D, eps, MinPts, DistanceMatrix):
        self.D = D
        self.eps = eps
        self.MinPts = MinPts
        self.DistanceMatrix = DistanceMatrix

    def run(self):
        """
        Cluster the dataset `D` using the DBSCAN algorithm.

        MyDBSCAN takes a dataset `D` (a list of vectors), a threshold distance
        `eps`, and a required number of points `MinPts`.

        It will return a list of cluster labels. The label -1 means noise, and then
        the clusters are numbered starting from 1.
        """

        # This list will hold the final cluster assignment for each point in D.
        # There are two reserved values:
        #    -1 - Indicates a noise point
        #     0 - Means the point hasn't been considered yet.
        # Initially all labels are 0.
        labels = [0] * len(self.D)

        # id is the ID of the current cluster.
        id = 0

        # This outer loop is just responsible for picking new seed points--a point
        # from which to grow a new cluster.
        # Once a valid seed point is found, a new cluster is created, and the
        # cluster growth is all handled by the 'expandCluster' routine.

        # For each point point in the Dataset D...
        # ('point' is the index of the datapoint, rather than the datapoint itself.)
        for point in range(0, len(self.D)):

            #pick only unvisited nodes
            if not (labels[point] == 0):
                continue

            # Find neighbors
            NeighborPts = self.regionQuery(self.D, point, self.eps, self.DistanceMatrix)

            # If the number is below MinPts, this point is noise.
            if len(NeighborPts) < self.MinPts:
                labels[point] = -1
            # Otherwise, if there are at least MinPts nearby, use this point as the
            # seed for a new cluster.
            else:
                id += 1
                self.growCluster(labels, point, NeighborPts, id)

        # All data has been clustered!
        return labels, id

    def growCluster(self, labels, point, NeighborPts, id):
        """
        Grow a new cluster with label `id` from the seed point `point`.

        This function searches through the dataset to find all points that belong
        to this new cluster. When this function returns, cluster `id` is complete.

        Parameters:
          `labels` - List storing the cluster labels for all dataset points
          `point`      - Index of the seed point for this new cluster
          `NeighborPts` - All of the neighbor of `point`
          `id`      - The label for this new cluster.
        """
        # Assign the cluster label to the seed point.
        labels[point] = id

        i = 0
        while i < len(NeighborPts):
            # Get the next point from the queue.
            neighbor = NeighborPts[i]

            # If neighbor was labelled NOISE make it leaf node
            if labels[neighbor] == -1:
                labels[neighbor] = id

            # Otherwise, if neighbor isn't already claimed, claim it as part of id.
            elif labels[neighbor] == 0:
                # Add neighbor to cluster id (Assign cluster label id).
                labels[neighbor] = id

                # Find all the neighbor of neighbor
                neighbors = self.regionQuery(neighbor)

                setExpandedNeighbors = set(neighbors)
                setNeighbors = set(NeighborPts)

                if len(neighbors) >= self.MinPts:
                    NeighborPts = list(setNeighbors.union(setExpandedNeighbors))  # if they werent sets this would be an infinite loop

            i += 1

    def regionQuery(self, point):
        """
        Find all points in dataset `D` within distance `eps` of point `point`.
        """
        neighbors = []

        # For each point in the dataset...
        for Pn in range(0, len(self.D)):
            if self.DistanceMatrix[point, Pn] < self.eps:
                neighbors.append(Pn)

        return neighbors