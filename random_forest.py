from __future__ import division
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

# read the raw data
train = np.genfromtxt('covtype_train.csv', delimiter=',')
test = np.genfromtxt('covtype_test.csv', delimiter=',')
# remove labels column from train set
train_data = train[:, :-1]
# remove labels column from test set
test_data = test[:, :-1]
# train labels
train_labels = train[:, -1]
# test labels
test_labels = test[:, -1]

# train random forest
print ('training random forest')
clf = RandomForestClassifier(n_estimators=100, random_state=42, max_features=None)
clf.fit(train_data, train_labels)

# training error
pred_train = clf.predict(train_data)
score_train = accuracy_score(train_labels, pred_train)
print ('accuracy on training set: %1.3f' % score_train)

# test error
pred_test = clf.predict(test_data)
score_test = accuracy_score(test_labels, pred_test)
print ('accuracy on test set:     %1.3f' % score_test)
