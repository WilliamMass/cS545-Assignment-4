import numpy as np
import random as rand
from collections import Counter
from operator import itemgetter

class HashableList(object): # hashable list code borrowed from http://blog.frank-mich.com/python-counters-on-objects/

    def __init__(self, val):
        self.val = val

    def __hash__(self):
        return hash(str(self.val))

    def __repr__(self):
            # Bonus: define this method to get clean output
        return str(self.val)

    def __eq__(self, other):
        return str(self.val) == str(other.val)


def main():
    K = int(input("Please enter the number of clusters (K) you would like to calculate.  Values of either 10 or 30 are acceptable."))
    if K == 10 or K == 30:
        training_file = "optdigits.train"
        test_file = "optdigits.test"

        training_matrix = readfile(training_file)
        i, j = training_matrix.shape
        test_matrix = readfile(test_file)
        l, m = test_matrix.shape

        max_sse = []
        max_sss = []
        max_entropy = []
        centroids = []
        for i in range(0, 1):
            sse, sss, entropy, centers = k_means(training_matrix, K)
            max_sse.append(sse)
            max_sss.append(sss)
            max_entropy.append(entropy)
            centroids.append(centroids)

        test_points = np.argmin(max_sse)
        test_centers = centroids[test_points]
        test_clusters = calculate_partitions(test_matrix, test_centers)
        print("Test: Sum Squared Separation: ", sum_squared_separation(test_centers))
        print("Test: Entropy: ", mean_entropy(test_clusters, K, i))
        print("Test: Sum Squared Error: ", sum_squared_error(test_centers, test_clusters))
        frequency(test_clusters, test_centers, l)
        write_file(test_centers)
    else:
        print("Improper K value specified; please enter either 10 or 30")
        main()

def readfile(file):
    feature_vector =[]
    with open(file, 'r') as f:
        line = f.read().splitlines()
    for l in line:
        splitline = l.split(",")
        for element in splitline:
            feature_vector.append(element)
    col = int(len(feature_vector)/65)
    matrix = np.zeros(col*65).reshape(col, 65)
    i, j = matrix.shape
    for l in range (0, i):
        for m in range (0, j):
            matrix[l][m] = int(feature_vector.pop(0))
    return matrix

def centroid(K):
    centroids = [[rand.randint(0, 16) for i in range(0, 64)] for j in range(0, K)]
    return centroids

def recalculate_centroids(clusters):
    new_centroids = []
    for i in range(0, len(clusters)):
        feature_vector = []
        new_vector = []
        if(len(clusters[i]) > 0 ):
            for j in clusters[i]:
                remove_label = j[:-1]
                feature_vector.append(list(remove_label))
            new_centroids.append(list(np.mean(feature_vector, axis=0, dtype=np.int64)))
        else:
            for j in range(0, 64):
                new_vector.append(rand.uniform(0, 17))
            new_centroids.append(new_vector)
    return new_centroids

def k_means(matrix, K):
    sse_data, sss_data, entropy_data, centroid_data = [[] for i in range(4)]
    i, j = matrix.shape
    centroids = centroid(K)
    for n in range (0, 5):
        clusters = calculate_partitions(matrix, centroids)
        sss = sum_squared_separation(centroids)
        ent_mean = mean_entropy(clusters, K, i)
        sse = sum_squared_error(centroids, clusters)
        entropy_data.append(ent_mean)
        sse_data.append(sse)
        sss_data.append(sss)
        centroid_data.append(clusters)
        centroids = recalculate_centroids(clusters)
    sse, center, sss, entropy = minimum_comparison(sss_data, sss_data, centroid_data, entropy_data)
    return sss, center, sss, entropy

def sum_squared_error(centroids, clusters):
    error = 0
    for i in range(0, len(centroids)):
        feature = clusters[i]
        for j in feature:
            error += np.square(np.asarray(i[:-1]) - np.asarray(centroids[i])).sum(axis=0)
    return error

def sum_squared_separation(clusters):
    sum = 0
    for i in range(0, len(clusters)):
        cluster = clusters.pop(i)
        for j in clusters:
            sum += np.square(np.asarray(cluster) - np.asarray(j)).sum(axis=0)
        clusters.insert(i, cluster)
    return sum

def euclidean_distance(feature, centroid_list):
    distances = [];
    features = feature[:-1]
    for i in range(0, len(centroid_list)):
        centroid = centroid_list.pop(i)
        sum = 0
        for i in range(0, len(centroid_list)):
            if len(centroid) != 0:
                sum += np.square(centroid[i] - features[i])
            else:
                break
        distance = np.sqrt(sum)
        distances.append(distance)
        centroid_list.insert(i, centroid)
    return distances

def minimum_comparison(sse, cen, sss, ent):
    index = np.argmin(sse)
    return sse[index], cen[index], sss[index], ent[index]

def count_bag(bag):
    counter = 0
    for x, y in bag:
        counter += y
    return counter

def mean_entropy(clusters, K, size_of_data):
    classes, cluster_size, coefficient = [[] for i in range(3)]
    entropy, log, class_count = 0, 0, 0
    for each in range(0, len(clusters)):
        clusters = clusters.pop(each)
        cluster_size.append(len(clusters))
        for i in clusters:
            classes.append(i[:-1])
        class_count = Counter(classes)
        unique_items = class_count.items()
        cluster_total = count_bag(unique_items)
        for x, y in unique_items:
            probability = y/cluster_total
            log += probability * np.log2(probability)
        coefficient.append(-log)
        log = 0
        clusters.insert(each, clusters)
    for i in range(0, len(clusters)):
        entropy += (cluster_size[i]/(size_of_data))*coefficient[i]
    return entropy

def frequency(clusters, centroids, quantity):
    most_frequent = []
    accuracy = 0
    for each in range(0, len(clusters)):
        count = 0
        class_list, frequency = []
        cluster = clusters.pop(i)
        print("Cluster: ", i)
        for i in cluster:
            class_list.append(int(i(-1)))
        class_frequency = Counter(class_list)
        unique_objects = class_frequency.items()
        for i, j in unique_objects:
            print("Object: ", i, " -- Frequency: ", j)
            count += j
            frequency.append((i, j))
        if frequency:
            frequent = max(frequency, key = itemgetter(1))[1]
            accuracy += frequent
            item = max(frequency, key=itemgetter(1))[1]
            print("The most frequent item is ", frequent, item, count)
            most_frequent.append(item)
        else:
            print("Cluster ", each, "is empty")
            most_frequent.append(None)
        clusters.insert(each, cluster)



def calculate_partitions(matrix, centers):
    i, j = matrix.shape
    num_centers = len(centers)
    clusters = [[] for i in range(num_centers)]
    for row in range(0, i):
        distances = []
        distances = euclidean_distance(matrix[row], centers)
        index = np.argmin(distances)
        clusters[index].append(list(matrix[row]))
    return clusters

def write_file(lists):
    with open('Users\Will\PycharmProjects\Assignment 4\k-Means.pgm', 'w') as file:
        for i in range(0, len(lists)):
            value = lists.pop(i)
            for j in range(0, 64):
                num = value.pop(i)
                int_num = int(num)
                file.write(str(int_num) + " ")
                value.insert(i, int_num)
            file.write('\n')
            lists.insert(i, value)

main()