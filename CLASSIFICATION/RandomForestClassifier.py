import numpy as np 	#Array		

import matplotlib.pyplot as plt		

import pandas as pd	

dataset = pd.read_csv(r"C:\Users\vaish\Downloads\logit classification.csv")

X = dataset.iloc[:, [2,3]].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

from sklearn.ensemble import  RandomForestClassifier
classifier = RandomForestClassifier(max_depth=4,n_estimators=100, random_state=0)
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
