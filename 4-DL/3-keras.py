

from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.utils import np_utils
from keras import backend as K
import numpy as np
import time
import pandas as pd
from sklearn import metrics


d_train = pd.read_csv("train-1m.csv")
d_test = pd.read_csv("test.csv")
d_train_test = d_train.append(d_test)

d_train_test["DepTime"] = d_train_test["DepTime"]/2500
d_train_test["Distance"] = np.log10(d_train_test["Distance"])/4


vars_categ = ["Month","DayofMonth","DayOfWeek","UniqueCarrier", "Origin", "Dest"]
vars_num = ["DepTime","Distance"]
def get_dummies(d, col):
    dd = pd.get_dummies(d.ix[:, col])
    dd.columns = [col + "_%s" % c for c in dd.columns]
    return(dd)
X_train_test_categ = pd.concat([get_dummies(d_train_test, col) for col in vars_categ], axis = 1)
X_train_test = pd.concat([X_train_test_categ, d_train_test.ix[:,vars_num]], axis = 1)
y_train_test = np.where(d_train_test["dep_delayed_15min"]=="Y", 1, 0)

X_train = X_train_test[0:d_train.shape[0]]
y_train = y_train_test[0:d_train.shape[0]]
X_test = X_train_test[d_train.shape[0]:]
y_test = y_train_test[d_train.shape[0]:]


X_train = X_train.as_matrix()
X_test = X_test.as_matrix()

y_train = np_utils.to_categorical(y_train, 2)


model = Sequential()
model.add(Dense(200, activation = 'relu', input_dim = X_train.shape[1]))
model.add(Dense(200, activation = 'relu'))
model.add(Dense(2, activation = 'softmax'))
sgd = SGD(lr = 0.01, momentum = 0.9)
model.compile(loss = 'categorical_crossentropy', optimizer = sgd, metrics = ['accuracy'])


start = time.time()
model.fit(X_train, y_train, batch_size = 128, nb_epoch = 1)
end = time.time()
print('Train time:', end - start, 'sec')


phat = model.predict_proba(X_test)[:,1]
metrics.roc_auc_score(y_test, phat)


## on Tensorflow:

## GPU:
# Train time: 34.6609380245 sec
# 0.71491495195154053

## CPU 4 cores 
### export CUDA_VISIBLE_DEVICES=""
# Train time: 58.4619350433 sec


## on Theano:

## GPU
### export KERAS_BACKEND=theano
### export THEANO_FLAGS='cuda.root=/usr/local/cuda-7.5,device=gpu,floatX=float32,lib.cnmem=0.9'
## Train time: 23.2013888359 sec

## CPU - uses 1 core
### export KERAS_BACKEND=theano
# Train time: 68.615885973 sec

