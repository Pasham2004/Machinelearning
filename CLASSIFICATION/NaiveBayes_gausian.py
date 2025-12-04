import numpy as np 	#Array		

import matplotlib.pyplot as plt		

import pandas as pd	

dataset = pd.read_csv(r"C:\Users\vaish\Downloads\logit classification.csv")

X = dataset.iloc[:, [2,3]].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

#model accuracy

from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred)
print(ac)

#classifications report

from sklearn.metrics import classification_report
cr = classification_report(y_test, y_pred)
print(cr)

#training score

bias = classifier.score(X_train, y_train)
print(bias)

#testing score

variance = classifier.score(X_test, y_test)
print(variance)

#----------------------------FUTURE DATASET-----------------------------

dataset1 = pd.read_csv(r"C:\Users\vaish\OneDrive\Documents\Desktop\SEPTEMBER DS\2.LOGISTIC REGRESSION CODE\2.LOGISTIC REGRESSION CODE\final1.csv")

d2 = dataset1.copy()

dataset1 = dataset1.iloc[:,[3,4]].values

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
M = sc.fit_transform(dataset1)

y_pred1 = pd.DataFrame()

d2['y_pred1'] = classifier.predict(M)

d2.to_csv('finalGaussian.csv')

import os
os.getcwd()














