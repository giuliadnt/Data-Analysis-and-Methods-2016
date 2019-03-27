from __future__ import division
import numpy as np
from center_and_normalize import cent_and_norm

train_x_orig = np.loadtxt('redwine_train.txt')
train_y = np.loadtxt('redwine_trainlabels.txt')
test_x_orig = np.loadtxt('redwine_test.txt')
test_y = np.loadtxt('redwine_testlabels.txt')

train_x, test_x = cent_and_norm(train_x_orig, test_x_orig)


def logistic(inp):
    return 1.0 / (1.0 + np.exp(-inp))


def logistic_insample(X, y, w):
    # N, num_feat = X.shape
    E = 0
    for n in xrange(X.shape[0]):
        E += (1 / X.shape[0]) * np.log(1 / logistic(y[n] * np.dot(w, X[n, :])))

    return E


def logistic_gradient(X, y, w):
    # N, _ = X.shape
    g = 0 * w
    for n in xrange(X.shape[0]):
        g += (1 / X.shape[0]) * (-y[n] * X[n, :] * logistic(-y[n] * np.dot(w, X[n, :])))
    return g


def log_reg(Xorig, y, max_iter, grad_thr):

    """
    :param Xorig: dataset, d by N data matrix of input values
    :param y: target labels binary values: 1; -1
    :param max_iter: maximum number of iteration
    :param grad_thr: threshold for the gradient descent
    :return: vector of weigths, vector of values computed by the logistic gradient function
    """

    # X is a d by N data matrix of input values
    num_pts, num_feat = Xorig.shape
    onevec = np.ones((num_pts, 1))
    X = np.concatenate((onevec, Xorig), axis=1)
    dplus1 = num_feat + 1

    # y is a N by 1 matrix of target values -1 and 1
    y = np.array((y - .5) * 2)

    # Initialize learning rate for gradient descent
    learningrate = 0.1

    # Initialize weights at time step 0
    w = 0.1 * np.random.randn(dplus1)

    # Compute value of logistic log likelihood
    value = logistic_insample(X, y, w)

    num_iter = 0
    convergence = 0

    # Keep track of function values
    E_in = []

    while convergence == 0:
        num_iter = num_iter + 1

        # Compute gradient at current w
        g = logistic_gradient(X, y, w)

        # Set direction to move and take a step
        w_new = w - learningrate * g  #

        # Check for improvement
        # Compute in-sample error for new w
        cur_value = logistic_insample(X, y, w_new)
        if cur_value < value:
            w = w_new
            value = cur_value
            E_in.append(value)
            learningrate *= 1.1
        else:
            learningrate *= 0.9

        # Determine whether we have converged: Is gradient norm below
        # threshold, and have we reached max_iter?

        g_norm = np.linalg.norm(g)
        if g_norm < grad_thr:
            convergence = 1

        elif num_iter > max_iter:
            convergence = 1

    return w, E_in


def log_pred(x_orig, w):
    ones = np.ones((len(x_orig), 1))
    x_orig_1 = np.concatenate((ones, x_orig), axis=1)
    P = logistic(np.dot(x_orig_1, np.transpose(w)))
    pred_classes = np.where(P >= 0.5, 1, 0)
    return P, pred_classes


def logreg(train_data, train_labels, test_data):
    # train on training
    # get parameters
    w, e = log_reg(train_data, train_labels, 20000, 0.0001)
    # predict on test --> XORIG = test
    pred_classes = log_pred(test_data, w)[1]

    return pred_classes, w


def accuracy(test_y, predictions):
    correct = 0
    for x in range(len(test_y)):
        if test_y[x] == predictions[x]:
            correct += 1
    return (correct / float(len(test_y))) * 100.0


# PREDICTION ON TEST SET
pred = logreg(train_x, train_y, test_x)[0]
print accuracy(test_y, pred)

# PREDICTIONS ON TRAIN SET
pred_2 = logreg(train_x, train_y, train_x)[0]
print accuracy(train_y, pred)
