import numpy as np
import math

#read data
def read_data(fn):
    #list comprehension to parse the data
    f = [i.strip().split() for i in open(fn).readlines()]
    # convert all items in float and split each instance in two parts:
    # data values and class of the object
    data = []
    for i in f:
        tup = (float(i[0]), float(i[1]))
        label = str(i[2])
        #append both tuple and label to data list
        data.append((tup, label))
    return data

#euclidean
def euclidean(inst1, inst2):
    result = sum((np.subtract(inst1, inst2))**2)
    dist = np.sqrt(result)
    return dist

#get nearest neighbor:
#takes a set (train) and a data point (test_instance)
def knn(train, test_inst, k):
    #create an empty list to store the results
    distances =[]
    #get the instances from the training set
    for x in train:
        #compute euclidean distance among each instance of the training set and the test instance
        distance = euclidean(x[0], test_inst)
        #append to list both distance and instance of training set
        distances.append((distance, x))
    #sort the distances by the first element
    sorted(distances, key=lambda x: x[0])
    #get the first k elements (nearest neighbors)
    k_nn = distances[:k]

    return k_nn
