# -*- coding: utf-8 -*-
"""
Created on Mon Nov 07 19:28:56 2016

@author: lyaa
"""

from boschStart2 import *

#train_id = pd.read_csv('input/train_numeric.csv', usecols=['Id', 'Response'])
#train_id.to_csv('train_id_response.csv', index_label=False)
train_id_response = pd.read_csv('train_id_response.csv')

clf = xgb.XGBClassifier(max_depth=6, objective='binary:logistic', 
                        learning_rate=0.005, colsample_bytree=0.1,
                        min_child_weight=1, n_estimators=69, subsample=1,
                        reg_alpha=3, reg_lambda=0)

n_sample = 110000
n_train = n_sample-10000
x_train = pd.read_csv('input/train_numeric.csv', nrows=n_sample)
y_train = np.array(x_train.Response.values)
x_train.drop('Response', axis=1, inplace=True)
x_train0 = x_train.iloc[np.arange(n_train)]
y_train0 = y_train[np.arange(n_train)]
x_train1 = x_train.iloc[np.arange(n_train, n_sample)]
y_train1 = y_train[np.arange(n_train, n_sample)]
#                 
prior = 1.*y_train0.sum()/len(y_train0)
clf.base_score = prior
#clf.fit(x_train0, y_train0, eval_set=[(x_train0, y_train0), (x_train1, y_train1)], 
#        eval_metric='auc', early_stopping_rounds=10, verbose=True)
#y_pred = clf.predict_proba(x_train1)[:, 1]
#
#best_proba, best_mcc, _ = eval_mcc(y_train1, y_pred, True)
#

#cv = model_selection.cross_val_score(clf, x_train0, y_train0, scoring='roc_auc',
#    cv=10, verbose=10, fit_params={'eval_set':[(x_train0, y_train0), 
#    (x_train1, y_train1)], 'eval_metric':'auc', 'early_stopping_rounds':10, 
#    'verbose':True})

params = {}
params['max_depth'] = [5, 6, 7]
params['learning_rate'] = [0.005, 0.008, 0.01]
params['colsample_bytree'] = [0.1, 0.2, 0.3]
params['min_child_weight'] = [1, 1.5]
params['reg_alpha'] = [2, 3]
params['reg_lambda'] = [0, 1]
gs = model_selection.RandomizedSearchCV(clf, params, n_iter=36, 
    scoring='roc_auc', fit_params={'eval_set':[(x_train0, y_train0), 
    (x_train1, y_train1)], 'eval_metric':'auc', 'early_stopping_rounds':10, 
    'verbose':True}, cv=10, verbose=10, random_state=0)
gs.fit(x_train0, y_train0)