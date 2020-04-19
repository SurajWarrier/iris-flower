# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 17:43:29 2020

@author: ASUS
"""
# importing the packages
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# reading the dataset
df = pd.read_csv('iris.csv', header = None)
X = df.iloc[:, :-1].values
y = df.iloc[:, 4].values

# splitting the data into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# training the model 
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 20, random_state = 0)
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)

# evaluating the model
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))


