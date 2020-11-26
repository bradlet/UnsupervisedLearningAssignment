# Bradley Thompson
# Programming Assignment #3
# CS 545 - Anthony Rhodes

import os
import numpy as np
import matplotlib.pyplot as pl

K_CLUSTERS = 3
ITERATIONS = 10
FUZZIFIER = 1.1


#################################
# Parent learner - For EM Learners
#################################

class Learner:

    def __init__(self, data):
        self.centroids = self.random_centroid_init(data)
        self.clusters = create_array_of_empty_lists(K_CLUSTERS)

    # Returns an array of indices for n centroids at random from data_table, where n = K_CLUSTERS
    @staticmethod
    def random_centroid_init(data):
        random_indices = np.random.randint(0, len(data), K_CLUSTERS)
        centroids = list()
        for i in range(0, len(random_indices)):
            centroids.append(data[random_indices[i]])
        return centroids

    # The Expectation Step
    def assignment(self, data):
        raise NotImplementedError("Children of class Learner must implement the assignment method.")

    # The Maximization Step
    # Find the new mean point for each cluster and set that to the centroids
    def update(self):
        for i in range(0, K_CLUSTERS):
            self.centroids[i] = mean_point(self.clusters[i])

    # Calculate the SSE for a single cluster
    def calculate_cluster_error(self, cluster_number):
        cluster_squared_error = 0
        for point in self.clusters[cluster_number]:
            cluster_squared_error += np.square(distance(point, self.centroids[cluster_number]))
        return cluster_squared_error

    # Accumulate SSE for all clusters
    def calculate_error(self):
        sum_squared_error = 0
        for i in range(0, K_CLUSTERS):
            sum_squared_error += self.calculate_cluster_error(i)
        return sum_squared_error

    # Learn on Data n times, where n = ITERATIONS (global)
    def run(self, data):
        for i in range(0, ITERATIONS):
            self.assignment(data)
            self.update()


#################################
# K Mean Child Learner
#################################

class KMeansLearner(Learner):

    def __init__(self, data):
        super().__init__(data)

    # For each point in data, find which centroid is closest - argmin for distance()
    # Assign point to that centroids set.
    def assignment(self, data):
        for point in data:
            min_distance = np.inf
            min_cluster = 0
            for i in range(0, K_CLUSTERS):
                current_distance = distance(self.centroids[i], point)
                if current_distance < min_distance:
                    min_distance = current_distance
                    min_cluster = i
            self.clusters[min_cluster].append(point)

    # Run the assignment and update steps n times, where n = ITERATIONS
    #   For each step, record the sum squared error and centroids for this step.
    def run(self, data):
        run_history = list()
        for i in range(0, ITERATIONS):
            self.assignment(data)
            self.update()
            # Create a tuple w/ (error, centroids that produced that error on the dataset), add that to history
            run_history.append((self.calculate_error(), self.centroids, self.clusters))
            self.clusters = create_array_of_empty_lists(K_CLUSTERS)
        return run_history


#################################
# Fuzzy c-means Child Learner
#################################

class FuzzyCLearner(Learner):

    def __init__(self, data):
        self.centroids = self.random_centroid_init(data)
        self.weights = np.random.random((len(data), K_CLUSTERS))

    def assignment(self, data):
        for i in range(0, len(data)):
            for j in range(0, K_CLUSTERS):
                self.weights[i][j] = self.weight_update(data, i, j)

    def weight_update(self, data, datapoint_num, centroid_num):
        denominator = 0
        for centroid in self.centroids:
            denominator += \
                (distance(data[datapoint_num], self.centroids[centroid_num]) /
                 distance(data[datapoint_num], centroid)) ** (2/(FUZZIFIER-1))
        return 1 / denominator

    def update(self):
        for i in range(0, len(self.centroids)):
            self.centroids[i] = self.centroid_update(i)

    def centroid_update(self, centroid_num):
        numerator = np.dot(np.transpose(self.weights)[centroid_num] ** FUZZIFIER, dataset)
        denominator = sum(np.transpose(self.weights)[centroid_num] ** FUZZIFIER)
        return numerator / denominator

    def run(self, data):
        run_history = list()
        for i in range(0, ITERATIONS):
            self.assignment(data)
            self.update()
            run_history.append((self.centroids, self.weights))
        return run_history


############################################
# End of class implementations
# Kept in single file for grading purposes.
############################################

def create_array_of_empty_lists(size):
    array = np.empty(size, type(list))
    for i in range(0, size):
        array[i] = list()
    return array


# Returns euclidean distance between two points
def distance(point1, point2):
    diff = np.square(point2 - point1)
    if sum(diff) <= 0.0:
        return 0.00001
    return np.sqrt(sum(diff))


# Returns the point representing the mean
def mean_point(list_of_points):
    sum_x = sum_y = 0
    for point in list_of_points:
        sum_x += point[0]
        sum_y += point[1]
    return sum_x / len(list_of_points), sum_y / len(list_of_points)


def load_data(name):
    return np.loadtxt(fname=name, dtype=np.dtype(np.float), usecols=range(2))


def create_graph(run_instance):
    clusters = run_instance[2]  # array of K_CLUSTER lists, each list contains points in the cluster
    for cluster in clusters:
        # Take transpose so that 0th indices are x's, 1st are y's
        point_array = np.transpose(np.array(cluster))
        pl.scatter(point_array[0], point_array[1])
    pl.show()


if __name__ == '__main__':
    os.chdir('./dataset')
    dataset = load_data('545_cluster_dataset.txt')

    # learner = KMeansLearner(dataset)
    # runs = learner.run(dataset)
    #
    # print("SSE's: ")
    # for run in runs:
    #     print(run[0])
    #     create_graph(run)

    fuzzy_learner = FuzzyCLearner(dataset)
    fuzzy_learner.run(dataset)
