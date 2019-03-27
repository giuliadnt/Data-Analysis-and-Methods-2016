from __future__ import division
from sklearn.cross_validation import KFold
import numpy as np
from knn import knn
from center_and_normalize import cent_and_norm

np.random.seed(0)
# remove label column from train and test dataset
train = np.genfromtxt('covtype_train.csv', delimiter=',')
test = np.genfromtxt('covtype_test.csv', delimiter=',')
train_data = train[:, :-1]
test_data = test[:, :-1]
# store labels
train_labels = train[:, -1]
test_labels = test[:, -1]

train_norm, test_norm = cent_and_norm(train_data[:, 0:9], test_data[:, 0:9])
binary_cols_train = train_data[:, 10:]
binary_cols_test = test_data[:, 10:]
train_x = np.column_stack((train_norm, binary_cols_train))
rand_vec = np.random.permutation(len(train_labels))


def accuracy(test_y, predictions):
    correct = 0
    for x in range(len(test_y)):
        if test_y[x] == predictions[x]:
            correct += 1
    return (correct/float(len(test_y))) * 100.0


def cv(train, trainlabels, rand_perm):

    """
    :param train N by d array
    :param trainlabels 1d array of labels
    :param rand_perm vector of randomized indices for dataset
    :return optimal k, matrix of accuracy over each k
    """

    error_matrix = np.zeros((5, 13))
    k_list = range(0, 11)[1::2]
    kf = KFold(len(train), n_folds=5)
    f = 0
    for train_idx, test_idx in kf:
        id_tr, id_tst = rand_perm[train_idx], rand_perm[test_idx]
        train_x, test_x = train[id_tr], train[id_tst]
        train_y, test_y = trainlabels[id_tr], trainlabels[id_tst]
        for i, k in enumerate(k_list):
            error_matrix[f, i] = accuracy(test_y, (knn(train_x, test_x, train_y, k)[1]))
        f += 1

    means = error_matrix.mean(axis=0)
    idx = np.argmax(means)
    k = k_list[idx]

    return k, error_matrix


# OPTIMAL K
print cv(train_x, train_labels, rand_vec)[0]
