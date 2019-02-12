#Group 2

import numpy as np
import math
def SVR(x,y, w, alpha, num_iters, lambda_, epsilon):
    T = 100
    K = math.floor(num_iters/T)
    Z = np.matmul(x,np.diagflat(y))

    N = x.shape[1]

    for k in range(K):
        wz = np.matmul(w.T , Z)
        diag = np.diagflat(1/(1+np.exp(-1*wz))-np.ones((1,N)))
        Ga_ = np.matmul(Z , diag)
        ga_ = (1/N) * np.matmul(Ga_ , np.ones((N,1)))
        for t in range(T):
            r = int(np.random.random(1) * N)
            col = Z[:,r]
            col = col.reshape((col.shape[0],1))
            g = np.matmul(col , (1/(1+np.exp(-1 * np.matmul(w.T , col)))-1))
            Ga_col = Ga_[:,r]
            Ga_col = Ga_col.reshape(Ga_col.shape[0],1)
            w = w - alpha * (g - Ga_col + ga_ + 2 * lambda_ * w)
    return w


def cost(x,y,w,lambda_):
    N = x.shape[1]
    value = 0
    for i in range(N):
        y_ = y[i].reshape(y[i].shape[0],1)
        x_ = x[:,i].reshape(x[:,i].shape[0],1)
        value += np.log(1+np.exp(-1*np.matmul(np.matmul(y_,w.T),x_)))
    return value/N + lambda_ * (np.linalg.norm(w) ** 2)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import time
# from sklearn.linear_model import RidgeClassifier

np.random.seed(0)

data = pd.read_csv("household_power_consumption.txt",";")

print(data.shape)
print(data.head(3))
print(data.tail(3))

X = pd.DataFrame(data.iloc[:,2:6], columns=["Global_active_power","Global_reactive_power","Voltage","Global_intensity"])
Y = pd.DataFrame(data.iloc[:,7], columns=["Sub_metering_2"])

print(X.head())

X = X.replace({'?':0})
Y = Y.replace({'?':0})

###########!!!!!We need to set the timestamp as the index of data set and use it in next steps.
###########!!!!!We can also use index to sort data set before training.

X_features = X.columns
Y_features = Y.columns
XY = pd.concat([X[X_features], Y[Y_features]], axis=1)

# Split XY into training set and test set of equal size
train, test = train_test_split(XY, test_size = 0.005)
# Sort the train and test sets after index (which became unsorted through sampling)
train = train.sort_index(axis=0)
test = test.sort_index(axis=0)

# Extract X,Y components from test and train sets
X_train = train[X_features].astype(float); X_test = test[X_features].astype(float)
Y_train = train[Y_features].astype(float); Y_test = test[Y_features].astype(float)

print(X_train.shape,X_test.shape, Y_train.shape,Y_test.shape)

w = np.zeros((X_train.shape[1],1))
alpha = 0.1
num_iters = 1000
lambda_ = 0.1
epsilon = 0.0001
y = np.array(Y_train.iloc[0:6000])
x = np.array(X_train.iloc[0:6000,:])
svre = SVR(x.T,y,w, alpha, num_iters, lambda_, epsilon)
print(svre)

err = cost(x.T,y,svre,0.1)
print(err)
