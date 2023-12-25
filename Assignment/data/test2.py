# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("training.csv")

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

x = data.drop('prediction', axis=1)
#x = x.drop('ID', axis=1)
y = data['prediction']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=27)

from sklearn.svm import SVC
svcclassifier = SVC(kernel='linear')
svcclassifier.fit(x_train, y_train)
y_pred = svcclassifier.predict(x_test)

testing = pd.read_csv("testing.csv")
#testing = testing.drop('GIST', axis=1)
testing_pred = svcclassifier.predict(testing)

print(testing_pred)

pd.DataFrame(testing_pred).to_csv("svc_results.csv")