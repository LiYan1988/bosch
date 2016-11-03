# -*- coding: utf-8 -*-
"""
Created on Wed Nov 02 23:48:27 2016

@author: lyaa
"""

from boschStart2 import *

train_set = pd.read_csv('train_date_feats.csv')
test_set = pd.read_csv('test_date_feats.csv')
y = pd.read_csv('input/train_numeric.csv', usecols=['Response'])

prior = np.array(np.sum(y) / (1.*len(y)))[0]

clfxgb = xgb.XGBClassifier(objective='binary:logistic', silent=False, 
        seed=0, nthread=-1, gamma=1, subsample=0.8, learning_rate=0.1,
        n_estimators=150, max_depth=6, base_score=prior)
#clfxgb.fit(train_set, y)

#y_test_pred = clfxgb.predict_proba(test_set)[:, 1]
#y_train_pred = clfxgb.predict_proba(train_set)[:, 1]
#
#best_proba, best_mcc, _ = eval_mcc(y.values, y_train_pred, True)
#y_test_pred = (y_test_pred >= best_proba).astype(int)
#
#save_submission(y_test_pred, 'test_submission.csv')

x_train = train_set.values
y_train = y.values
y_train.shape = (y_train.shape[0],)
mcc_scores = []
kf = model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
for train_index, valid_index in kf.split(x_train, y_train):
    clfxgb.fit(x_train[train_index], y_train[train_index])
    y_valid_pred = clfxgb.predict_proba(x_train[valid_index])[:,1]
    best_proba, best_mcc, _ = eval_mcc(y_train[valid_index], y_valid_pred, True)
    mcc_scores.append(best_mcc)
    print best_mcc
