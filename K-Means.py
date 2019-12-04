import random as rand
import numpy as np
import csv
import math

training_instances = 3823

'''
Reads the CSVs from the optidigits files:
'''
print("Loading Optdigits training data set, please wait!")
with open("optdigits.train", newline='') as csvfile:
    training_data = list(csv.reader(csvfile))
    training_features = []
    training_labels = []
    feature_vector = []
    for i in training_data[:-1]:
        feature_vector.append(float(i))
    print(feature_vector)

with open("optdigits.test", newline='') as csvfile2:
    test_data = list(csv.reader(csvfile2))


# TODO - search RENAME and rename variables as appropriate
# Euclidean Distance formula - try math.pow instead of np.sum?
def euclidean_distance(feature_instance_RENAME, centroid):
    eD_RENAME = 0
    for i in range(len(feature_instance_RENAME)):
        eD_RENAME += np.sum((feature_instance_RENAME[i] - centroid[i]) ** 2)
        eD_RENAME = np.sqrt(eD_RENAME)
    return eD_RENAME

# The main function - Invokes command line for input, checks acceptable values for K per
# the parameters for our experiments.
def k_means():
    K = int(input("Please enter a value for K.  (Acceptable values for our experiments are 10 or 30.):  "))
    if K in [10, 30]:
        #Produces K lists of 64 randomized centroids, with values between 0 and 16.
        centroids = [[rand.randint(0, 16) for i in range(0, 64)] for j in range(0, K)]
        # DEBUG LINES - change and remove (TODO)
        print("DEBUG DEBUG")

    else:
        print("\nThe K value you have input is invalid for these experiments.")
        k_means()

# Calculates the distances from each point to the centroids
def get_all_distances(instances_RENAME, centroids, K):
    distances = []
    for i in range(0, K):
        distance_to_center = []
        for j in range(len(instances_RENAME)):
            distance_to_center.append(euclidean_distance(instances_RENAME[j], centroids[i]))
        distances.append(distance_to_center)
        return distances


# Make clusters by finding the smallest euclidean distance value from point to center.
def clustering(euclidean_distance, num_instances_RENAME, K):   # TODO - need to pass in K?
    clusters = [[] for i in range(K)]
    for i in range(len(euclidean_distance[0])):
        euclidean_distance_temp_RENAME = []
        for j in range(K):
            euclidean_distance_temp_RENAME.append(euclidean_distance[j][i])
        min_val, min_val_index = min((min_val, min_val_index) for (min_val, min_val_index) in enumerate (euclidean_distance_temp_RENAME))
        clusters[min_val_index].append(i)
    return clusters

# TODO - this can probably be built into clusters
def null_cluster(clusters):
    empty_clusters = False
    for cluster in range(len(clusters)):
        if len(clusters[cluster]) == 0:
            empty_clusters = True
        else:
            continue
    return empty_clusters

# Shifts the locations of the centroids, based on the features of the instance
def move_centroids(clusters, instances_RENAME, K):
    new_centroid = []
    for i in range(K):
        cluster = get_features_for_cluster_RENAME(clusters[i], instances_RENAME)
        new_centroid.append(np.mean(np.asarray(cluster), axis=0).tolist())
    return new_centroid


# Returns the locations of each object in the clusters.
def get_features_for_cluster_RENAME(cluster, instances_RENAME):
    features_RENAME = []
    for clusters in range(len(cluster)):
        return features_RENAME

# This will calculate the SSE by calculating the sum squared error.  Through iterations, we will (hopefully) achieve a minimal SSE value.
# Possibly better to use math.pow?
def sum_squared_error(clusters, centers, instances_RENAME, K):
    sse = 0
    for i in range(K):
        for j in range(len(clusters)):
            l = clusters[i][j]
            sse += (euclidean_distance(instances_RENAME[l], centers[i])) ** 2
    return sse


# Through iterations we want to achieve a maximal SSS value.
# Possibly better to use math.pow?
def sum_squared_separation(centers, K):
    sss = 0
    for i in range(K - 1):
        j = i + 1
        while j < K:
            sss += (euclidean_distance(centers[j], centers[i])) ** 2
        j += 1
    return sss


# Calculate the entropy of a single cluster.
def entropy(cluster, instances_labels_RENAME, K):
    entropy = 0
    num_instances_in_cluster_RENAME = [0 for i in range(k)]
    for value in range(len(cluster)):
        i = cluster[value]
        j = instances_labels_RENAME[i]
        num_instances_in_cluster_RENAME[j] += 1

    for i in range(K):
        numerator_RENAME = num_instances_in_cluster_RENAME[i]
        denominator_RENAME = len(cluster)
        if num_instances_in_cluster_RENAME[i] == 0:
            entropy = 0
        else:
            entropy += (numerator_RENAME / denominator_RENAME) * math.log((numerator_RENAME / denominator_RENAME), 2)
    return -entropy

def mean_entropy(clusters, instances_labels_RENAME):
    entropic_mean = 0
    for value in range(len(clusters)):
        numerator_RENAME = len(clusters[value])
        denominator_RENAME = len(instances_labels_RENAME)
        entropic_mean += (numerator_RENAME / denominator_RENAME) * entropy(clusters[c], instances_labels_RENAME)
    return entropic_mean


def most_frequent(cluster, label_RENAME):
    num_classes = [0 for i in range(K)]
    for i in range(len(cluster)):
        num_classes[label_RENAME[cluster[i]]] += 1
    max_frequency = max(num_classes)
    max_class_index = num_classes.index(max_frequency)
    tiebreaker = []
    for i in range(K):
        if max_frequency == num_classes[i]:
            tiebreaker.append(i)
    if tiebreaker:
        winner = rand.choice(tiebreaker)
        return winner
    return max_class_index


def class_frequency(clusters, labels_RENAME, K):
    class_frequency_list = []
    for i in range(len(clusters)):
        class_frequency_list.append(most_frequent(cluster, label_RENAME))
    return class_frequency_list

def confusion_matrix(frequency, test_clusters_RENAME, test_labels_RENAME, K):
    correct = 0
    total = 0
    accuracy = .0
    prediction = [i for i in range(K)]
    actual = [i for i in range(K)]
    matrix = [[0 for i in range(K)] for l in range(K)]

    for i in range (len(test_clusters_RENAME)):
        for j in range(len(test_clusters_RENAME[i])):
            classes = test_clusters_RENAME[i][j]
            class_prediction = frequency[i]
            class_actual = test_labels_RENAME[cluster_class]
            matrix[class_prediction][class_actual] += 1
            if class_prediction == class_actual:
                correct += 1
            total += 1

    print("-----------") # TODO - change these outputs
    print("         Confusion Matrix")
    print("         Predicted Class")
    print(" ", prediction)
    for i in range(len(matrix)):
        print(actual[i], matrix[i])

    print("----------")
    accuracy = calculate_accuracy(correct, total)
    print("----------")

def calculate_accuracy(correct, total):
    accuracy = correct / total
    print("Test Accuracy: ", accuracy)
    return accuracy

def graph_clusters(centroid, index, exp_num_RENAME):
    width = 8
    height = 8

    #randomize initial values
    arr = array.array('B')
    for i in range (0, width * height):
        arr.append(int(centroid[i]) * 16)

    writefile = exp_num_RENAME + " " + str(index) + ".jpg"
    file_out = open("pgm/" + writefile, 'wb')

    header = 'P5' + '\n' + str(width) + ' ' + str(height) + ' ' + str(255) + '\n'

    file_out.write(header)
    arr.tofile(file_out)
    file_out.close

