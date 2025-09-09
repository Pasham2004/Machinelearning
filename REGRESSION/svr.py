import numpy as np 	#Array		

import matplotlib.pyplot as plt		

import pandas as pd	

dataset = pd.read_csv(r"C:\Users\vaish\Downloads\emp_sal.csv")

X = dataset.iloc[: , 1:2].values
y = dataset.iloc[: , 2].values


#lets build regression build
from sklearn.svm import SVR
svr_regressor = SVR()
svr_regressor.fit(X, y)


svr_model_pred = SVR(kernel='poly', degree = 5, gamma='scale',C=10.0)
svr_regressor.fit(X,y)


svr_model_pred = svr_regressor.predict([[6.5]])
print(svr_model_pred)

from sklearn.neighbors import KNeighborsRegressor
knn_reg_model=KNeighborsRegressor(n_neighbors=7,weights='distance')
knn_reg_model.fit(X,y)

knn_reg_pred = knn_reg_model.predict([[6.5]])
print(knn_reg_pred)



4-- 182500
5 -- 175248
6 -- 196521

#DecisionTree
from sklearn.tree import DecisionTreeRegressor
dtr_reg_model = DecisionTreeRegressor(criterion='absolute_error',max_depth = 10 ,splitter = 'random')
dtr_reg_model.fit(X,y)

dtr_reg_pred = dtr_reg_model.predict([[6.5]])
print(dtr_reg_pred)


#random forest
from sklearn.ensemble import RandomForestRegressor
rfr_reg_model = RandomForestRegressor(n_estimators=7, random_state=0)
rfr_reg_model.fit(X,y)

rfr_reg_pred = rfr_reg_model.predict([[6.5]])
print(rfr_reg_pred)
