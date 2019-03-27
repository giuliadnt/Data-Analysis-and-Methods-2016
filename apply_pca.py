from __future__ import division
import numpy as np


def apply_pca(data):

    """
    :param data N by d array
    :return centered data
    """

    centered = np.array([i-(data.mean(axis=0)) for i in data])
    evals, evecs = np.linalg.eig(np.cov(centered.T))
    evals, evecs = np.real(evals), np.real(evecs)
    sorted_evals, sorted_evecs = evals[np.argsort(evals)[::-1]], evecs[:, np.argsort(evals)[::-1]]
    evecs_matrix = sorted_evecs[:, :2]
    return np.dot(centered, evecs_matrix)
