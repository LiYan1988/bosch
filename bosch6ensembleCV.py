# -*- coding: utf-8 -*-
"""
Created on Wed Nov 09 14:57:27 2016

@author: lyaa
"""

from boschStart2 import *

#%% load data
x_train = read_data('train_numeric_feature200_set1.pkl')
x_train.fillna(-999, inplace=True)
y_train = np.array(x_train.Response.values)
response = x_train[['Id', 'Response']].copy()
x_train.drop(['Response'], axis=1, inplace=True)
x_test = read_data('test_numeric_feature200_set1.pkl')
x_test.fillna(-999, inplace=True)

#%%
#x_train = x_train.iloc[np.arange(10000)]
#y_train = y_train[np.arange(10000)]
#x_test = x_test.iloc[np.arange(10000, 20000)]

#%% split data into meta and bag sets
clf = xgb.XGBClassifier(max_depth=15, objective='binary:logistic', 
                        learning_rate=0.01, colsample_bytree=0.4,
                        min_child_weight=1, n_estimators=40, subsample=0.8,
                        reg_alpha=2, reg_lambda=1)

n_run = 2
n_cv = 2
random_state = 0
results = cv_bag(clf, x_train, y_train, x_test, n_run, n_cv, random_state, 
           verbose=True)
y_train_pred_list, y_test_pred_list, best_ntree_limit = results

for i in range(n_run):
    best_proba, best_mcc, _ = eval_mcc(y_train, y_train_pred_list[i], True)