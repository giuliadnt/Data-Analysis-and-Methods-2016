from __future__ import division
from scipy.stats import t
import numpy as np
import math

train_x = np.loadtxt('redwine_train.txt')
train_y = np.loadtxt('redwine_trainlabels.txt')

def hyptest(X, Y):
    '''
    :param X: dataset, d by N data matrix of input values
    :param Y: target labels binary values: 0, 1
    :return True if null hypothesis is accepted, False if it is rejected
    '''
    #H0 --> muX - muY = 0
    alpha = 0.05 #significance level
    t_val_num = np.mean(X) - np.mean(Y)
    t_val_den = math.sqrt((np.var(X)/len(X))+(np.var(Y)/len(Y)))
    t_val = t_val_num/t_val_den
    v_num = ((np.var(X)/len(X))+(np.var(Y)/len(Y)))**2
    v_den = ((np.var(X))**2/((len(X)**2)*(len(X)-1)))+((np.var(Y))**2/((len(Y)**2)*(len(Y)-1)))
    v = v_num/v_den #v degrees of freedom
    p_t0 = 2 * t.cdf(-abs(t_val), math.floor(v)) # ???
    print t_val, v, p_t0
    if p_t0 < alpha:
        return True #we should reject the null hypotesis
    else:
        return False #we cannot reject the null hypothesis

print hyptest(train_x, train_y)