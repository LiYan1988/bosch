# -*- coding: utf-8 -*-
"""
Created on Tue Nov 08 19:43:28 2016

@author: lyaa
"""


from boschStart2 import *

mcc_score = metrics.make_scorer(mcc_sklearn, greater_is_better=True)

# test score function
#ground_truth = np.array([[0],[1],[1],[0]])
#predictions  = np.array([0, 1, 1, 0])
#clf = dummy.DummyClassifier(strategy='most_frequent', random_state=0)
#clf = clf.fit(ground_truth, predictions)
#print(mcc_score(clf,ground_truth, predictions) )

x_train = read_data('train_numeric_feature200_set1.pkl')
x_train.fillna(-999)
y_train = np.array(x_train.Response.values)
x_train.drop(['Response'], axis=1, inplace=True)
#x_train = x_train.iloc[range(10000)]
#y_train = y_train[:10000]

clf = xgb.XGBClassifier(max_depth=15, objective='binary:logistic', 
                        learning_rate=0.01, colsample_bytree=0.4,
                        min_child_weight=1, n_estimators=40, subsample=0.8,
                        reg_alpha=2, reg_lambda=1)

prior = 1.*y_train.sum()/len(y_train)
clf.base_score = prior

n_train = 1183747
n_test = 1183748

np.random.seed(0)

params = {}
params['subsample'] = [0.6, 0.8]
params['learning_rate'] = [0.005, 0.01, 0.002]
params['n_estimators'] = [50]
params['max_depth'] = [13, 15, 17, 19, 21]
params['colsample_bytree'] = [0.2, 0.3, 0.4]
params['reg_alpha'] = [2, 3]

gridcv = model_selection.RandomizedSearchCV(clf, params, scoring='roc_auc', 
    n_iter=20, verbose=10, cv=5)
gridcv.fit(x_train, y_train)