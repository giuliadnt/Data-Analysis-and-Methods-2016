from __future__ import division
import numpy as np
from apply_pca import apply_pca
import matplotlib.pyplot as plt

plt.style.use('ggplot')

iris = np.loadtxt('Irisdata.txt')
np.random.seed(0)
seeds = np.random.choice(range(len(iris)), 3, replace=False)


def kmeans(data, seed_indices):

    """
    :param data: dataset in a d x N matric
    :param seedIndices: vector of indices for k seed observation
    :return: cluster centers in a numpy array; assigned label for each datapoint (numpy array
    """
    centroids = data[seed_indices, :]
    C = np.zeros(len(data))
    new_centroids = np.zeros_like(centroids)
    threshold = 100
    delta = 0

    while True:
        for i, obj in enumerate(data):
            C[i] = np.argmin(map(sum, (obj - centroids) ** 2))
        for j in xrange(len(centroids)):
            index = (C == j)
            new_centroids[j, :] = sum(data[index]) / sum(index)

        centroids = new_centroids
        delta += 1
        if delta > threshold:
            break

    return centroids, C


# PLOT OF CLUSTERS AND CENTROIDS
clusters = apply_pca(iris)
labels = kmeans(apply_pca(iris), seeds)[1]
centers = kmeans(apply_pca(iris), seeds)[0]
plt.scatter(clusters[np.where(labels == 0)][:, 0], clusters[np.where(labels == 0)][:, 1], marker='.', color='r')
plt.scatter(centers[0][0], centers[0][1], s=50, color='r')
plt.scatter(clusters[np.where(labels == 1)][:, 0], clusters[np.where(labels == 1)][:, 1], marker='.', color='g')
plt.scatter(centers[1][0], centers[1][1], s=50, color='g')
plt.scatter(clusters[np.where(labels == 2)][:, 0], clusters[np.where(labels == 2)][:, 1], marker='.', color='b')
plt.scatter(centers[2][0], centers[2][1], s=50, color='b')
plt.title("Kmeans clusters on IRIS dataset")
plt.show()