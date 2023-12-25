# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("training.csv")

data.shape
data.head()

x = data.drop('prediction', axis=1)
x = x.drop('ID', axis=1)
y = data['prediction']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 109)

from sklearn.svm import SVC
svcclassifier = SVC(kernel='linear')
svcclassifier.fit(x_train, y_train)
svcclassifier.score(x_train, y_train)

y_pred = svcclassifier.predict(x_test)

testing = pd.read_csv("testing.csv")
testing = testing.drop('GIST', axis=1)
testing_predict = svcclassifier.predict(testing)

#print(testing_predict)
print(testing)

from sklearn.metrics import classification_report, confusion_matrix
#print(confusion_matrix(y_test,y_pred))
#print(classification_report(y_test,y_pred))

#print(confusion_matrix(testing,testing_predict))
#print(classification_report(testing,testing_predict))

from sklearn import metrics

#print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
#print("Precision:",metrics.precision_score(y_test, y_pred))
#print("Recall:",metrics.recall_score(y_test, y_pred))

#print("Accuracy:",metrics.accuracy_score(testing,testing_predict))
#print("Precision:",metrics.precision_score(testing,testing_predict))
#print("Recall:",metrics.recall_score(testing,testing_predict))