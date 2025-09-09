import numpy as np	

import matplotlib.pyplot as plt		

import pandas as pd	

dataset = pd.read_csv(r"C:\Users\vaish\Downloads\Investment.csv")

X = dataset.iloc[:, :-1]
y = dataset.iloc[:, 4]

X = pd.get_dummies(X,dtype=int)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2, random_state=0)

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

m = regressor.coef_
print(m)


c = regressor.intercept_
print(c)

#X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1)

X = np.append(arr=np.full((50,1),42467).astype(int), values=X, axis=1)

#statsmodel.api  api-- application program interface(agent)

import statsmodels.api as sm

X_opt = X[:,[0,1,2,3,4,5]]
#OrdinaryLeastSquares
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()  #exog==feedin feedout or garbagein garbageout
regressor_OLS.summary()

import statsmodels.api as sm

X_opt = X[:,[0,1,2,3,5]]
#OrdinaryLeastSquares
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

import statsmodels.api as sm

X_opt = X[:,[0,1,2,3]]
#OrdinaryLeastSquares
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

import statsmodels.api as sm

X_opt = X[:,[0,1,3,]]
#OrdinaryLeastSquares
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

import statsmodels.api as sm

X_opt = X[:,[0,1,]]
#OrdinaryLeastSquares
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

import pickle
filename = 'Multilinear_regression_model.pkl'
with open (filename,'wb') as file:
    pickle.dump(regressor, file)
print("Model has been pickled and saved as Multilinear_regression_model.pkl")

import os 
os.getcwd


#IV QUES:- what is p-value where used p-value in your existing
#p-value also used stock grow, kite
#p-value 0.05 helps to reject null hypothesis
   #ml based 0.05 we eliminate the attribute for business explanation
   #this concept in ml is called bacward elimination this elImination comes under recursive feature elimination(RFE). RFE hasb2btwo types 1.backward elimination - pvalue(0.05)<model generate significant value 2.forward elimination
   
   #hoe to eliminate features in ml model:-1.business understanding 2.based on p value 3.based on pca 4.decision tree













