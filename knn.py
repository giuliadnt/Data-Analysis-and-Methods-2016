from __future__ import division
import numpy as np
from scipy.spatial import distance
from center_and_normalize import cent_and_norm

#remove label column from train and test dataset
train = np.genfromtxt('covtype_train.csv', delimiter=',')
test = np.genfromtxt('covtype_test.csv', delimiter=',')
train_data = train[:,:-1]
test_data = test[:,:-1]
#store labels
train_labels = train[:,-1]
test_labels = test[:,-1]

train_norm, test_norm = cent_and_norm(train_data[:,0:9], test_data[:,0:9])
binary_cols_train = train_data[:,10:]
binary_cols_test = test_data[:,10:]
train_x = np.column_stack((train_norm, binary_cols_train))
test_x = np.column_stack((test_norm, binary_cols_test))

def knn(train, test, trainlabels, k):
    '''
    :param train  N by d array
    :param test   M by d array
    :param trainlabels 1d array of binary classes
    :param k number of neighbours
    :return matrix of computed distances (numpy nd array), nd array of predictions
    '''
    predictions = np.zeros(len(test))
    dist_matrix = np.zeros((len(test), len(train)))
    for i, test_vector in enumerate(test):
        for j, train_vector in enumerate(train):
            dist_matrix[i,j] = distance.euclidean(test_vector, train_vector)

    for test_idx in range(len(test)):
        distances_testinst_vs_train = dist_matrix[test_idx, :]
        indices_trainingset_ordered = np.argsort(distances_testinst_vs_train)[:k]
        classes = np.asarray(trainlabels)[indices_trainingset_ordered]
        counts = np.bincount(classes.astype(int))
        most_common_class = np.argmax(counts)
        predictions[test_idx] =  most_common_class
    return dist_matrix, predictions

def accuracy(test_y, predictions):
    correct = 0
    for x in range(len(test_y)):
        if test_y[x] == predictions[x]:
           correct += 1
    return (correct/float(len(test_y))) * 100.0
# list of possible k
k_list = [1, 3, 5, 7, 9]

for k in k_list:
    #training on train testing on test
    y_pred_k = knn(train_x, test_x, train_labels, k)[1]
    #accuracy on test
    print accuracy(test_labels, y_pred_k)
    #training on train testing on train
    y_pred_k_2 = knn(train_x, train_x, train_labels, k)[1]
    #accuracy on train
    print accuracy_score(train_labels, y_pred_k_2)