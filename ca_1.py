#group 2

import pandas as pd
import numpy as np

np.random.seed(0)

data = pd.read_csv("crimedata.csv")

X = data.iloc[:,5:127]
Y = data.iloc[:,127]

print(X.shape)
print(Y.shape)

X_ = np.array(X.replace({'?':0})).T
X_ = X_.astype(float)
Y_ = np.array(Y).reshape(Y.shape[0],1)
lambda_ = 0.0
I = np.identity(122)
p1 = np.matmul(X_,X_.T)
p2 = lambda_*I
inv = np.linalg.inv(p1+p2)
close_form_solution = np.matmul(np.matmul(inv,X_),Y_)
print(close_form_solution)


###############using iteretive models for smaller data set################
from sklearn import linear_model
reg = linear_model.Ridge(alpha=0)
reg.fit(X_.T, Y_)
print(reg.coef_)

###############Previous model for latge data set##############

data = pd.read_csv("household_power_consumption.txt",";")

print(data.shape)
print(data.head(3))
print(data.tail(3))

X = data.iloc[:,2:6]
Y = data.iloc[:,7]

print(X.shape)
print(Y.shape)

# X['Date'] = pd.to_datetime(X['Date'].values).days
# first_day = pd.to_datetime(X['Date'].values).min()
# X['Date'] = X['Date'] - first_day
# X['Time'] = pd.to_datetime(X['Time'].values, format='%H:%M:%S').seconds
# print(X['Date'].iloc[0], X['Time'].iloc[0])

X_ = np.array(X.replace({'?':0})).T
X_ = X_.astype(float)
Y_ = np.array(Y.replace({'?':0})).reshape(Y.shape[0],1).astype(float)
lambda_ = 0.1
I = np.identity(4)
p1 = np.matmul(X_,X_.T)
p2 = lambda_*I
inv = np.linalg.inv(p1+p2)
close_form_solution = np.matmul(np.matmul(inv,X_),Y_)
print(close_form_solution)

###############using iteretive models for smaller data set################
from sklearn import linear_model
reg = linear_model.Ridge(alpha=0)
reg.fit(X_.T, Y_)
print(reg.coef_)


