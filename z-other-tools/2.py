import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

d_train = pd.read_csv("train-1m.csv")
d_test = pd.read_csv("test.csv")
d_train_test = d_train.append(d_test)

vars_categ = ["Month","DayofMonth","DayOfWeek","UniqueCarrier", "Origin", "Dest"]
vars_num = ["DepTime","Distance"]
def get_dummies(d, col):
    dd = pd.get_dummies(d.ix[:, col])
    dd.columns = [col + "_%s" % c for c in dd.columns]
    return(dd)
%time X_train_test_categ = pd.concat([get_dummies(d_train_test, col) for col in vars_categ], axis = 1)
X_train_test = pd.concat([X_train_test_categ, d_train_test.ix[:,vars_num]], axis = 1)
y_train_test = np.where(d_train_test["dep_delayed_15min"]=="Y", 1, 0)

X_train = X_train_test[0:d_train.shape[0]]
y_train = y_train_test[0:d_train.shape[0]]
X_test = X_train_test[d_train.shape[0]:]
y_test = y_train_test[d_train.shape[0]:]


md = RandomForestClassifier(n_estimators = 100, n_jobs = -1)
%time md.fit(X_train, y_train)

%time phat = md.predict_proba(X_test)[:,1]
metrics.roc_auc_score(y_test, phat)
