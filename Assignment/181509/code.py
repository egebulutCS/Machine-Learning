# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

#importing training csv file
data = pd.read_csv('training.csv')

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

#splitting file into predictions and features
y = data['prediction']
x = data.drop(['prediction'], axis=1)

#splitting the extracted information into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=27)

#creating the classifier with the following parameters
clf = MLPClassifier(hidden_layer_sizes=(700,700,700), max_iter=1500, alpha=0.01, solver='sgd', random_state=21, tol=0.000000001)

#training the classifier with the previously extracted sets
clf.fit(x_train, y_train)
clf.fit(x_test, y_test)
#predicting one of the testing sets
y_pred = clf.predict(x_test)

#importing testing csv file
testing = pd.read_csv('testing.csv')
#running predictions on the actual testing file
test_pred = clf.predict(testing)

print(test_pred)
#exporting the results of the predcition of testing file to a new csv file called results.
pd.DataFrame(test_pred).to_csv("results.csv")

#analysis of the testing set with multiple tools
import matplotlib.pyplot as plt
#plotting the loss curve on graph
loss_values = clf.loss_curve_
plt.plot(loss_values)
plt.show()

#constructing confusion matrix and classification report on a single set of training and testing set
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

#calculating accuracy, precision and recall on a single set of training and testing set
from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))