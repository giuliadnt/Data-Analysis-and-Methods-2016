from __future__ import division
import numpy as np


def cent_and_norm(train, test):

    """
    param: N by d numpy array
    param: M by d numpy array
    return: normalized train, normalized test as numpy nd arrays
    """

    rows, cols = train.shape
    train_2 = np.zeros((rows, cols))
    rows2, cols2 = test.shape
    test_2 = np.zeros((rows2, cols2))
    for col in xrange(cols):
        means = np.mean(train[:,col])
        std = np.std(train[:,col])
        for row in xrange(rows):
            train_2[row, col] = (train[row, col] - means)/std
            # np.mean(train[:, col]))/np.std(train[:, col])
            for r in range(rows2):
                test_2[r, col] = (test[r, col] - means)/std

    return train_2, test_2
