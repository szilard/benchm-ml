import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics

d_train = pd.read_csv("train-0.1m.csv")
d_test = pd.read_csv("test.csv")
d_train_test = d_train.append(d_test)


vars_cat = ["Month","DayofMonth","DayOfWeek","UniqueCarrier", "Origin", "Dest"]
vars_num = ["DepTime","Distance"]
def my_get_dummies(d, col):
    dd = pd.get_dummies(d.ix[:, col])
    dd.columns = [col + "_%s" % c for c in dd.columns]
    return(dd)
%time X_train_test_cat = pd.concat([my_get_dummies(d_train_test, col) for col in vars_cat], axis = 1)
X_train_test = pd.concat([X_train_test_cat, d_train_test.ix[:,vars_num]], axis = 1)
y_train_test = np.where(d_train_test["dep_delayed_15min"]=="Y", 1, 0)

X_train = X_train_test[0:d_train.shape[0]]
y_train = y_train_test[0:d_train.shape[0]]
X_test = X_train_test[d_train.shape[0]:]
y_test = y_train_test[d_train.shape[0]:]


md = GradientBoostingClassifier(n_estimators = 1000, 
         learning_rate = 0.01, max_depth = 16, subsample = 0.5)
%time md.fit(X_train, y_train)


%time phat = md.predict_proba(X_test)[:,1]
metrics.roc_auc_score(y_test, phat)


