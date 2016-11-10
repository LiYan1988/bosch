# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 20:25:08 2016

@author: lyaa
"""

from boschStart2 import *

meta_estimator_rf = ensemble.RandomForestClassifier(n_estimators=40, 
    max_depth=7, n_jobs=-1)
meta_estimator_ext = ensemble.ExtraTreesClassifier(n_estimators=40, 
    max_depth=7, n_jobs=-1)
meta_estimator_ada = ensemble.AdaBoostClassifier(base_estimator=
    tree.DecisionTreeClassifier(max_depth=10), n_estimators=40)
#meta_estimator_gb = naive_bayes.GaussianNB(priors=[.9942, .0058])
meta_estimator_xgb = xgb.XGBClassifier(max_depth=7, 
    objective='binary:logistic', learning_rate=0.01, 
    colsample_bytree=0.4, min_child_weight=1, n_estimators=40, 
    subsample=0.6, reg_alpha=2, reg_lambda=1)
meta_estimators = {'rf':meta_estimator_rf, 'ext':meta_estimator_ext, 
                   'ada':meta_estimator_ada, 'xgb':meta_estimator_xgb}

clf = xgb.XGBClassifier(max_depth=11, objective='binary:logistic', 
                        learning_rate=0.01, colsample_bytree=0.4,
                        min_child_weight=1, n_estimators=90, 
                        subsample=0.6, reg_alpha=2, reg_lambda=1)

y_train_pred_list = []
y_test_pred_list = []
n_rep = 30
n_cv = 5
np.random.seed(0)
for i in range(n_rep):
    y_train_pred, y_test_pred, y_train, test_id = \
        bagging2(meta_estimators, clf, n_cv, 
                 np.random.randint(10000))
    y_train_pred_list.append(y_train_pred)
    y_test_pred_list.append(y_test_pred)
    
y_train_pred = np.mean(y_train_pred_list, axis=0)
y_test_pred = np.mean(y_test_pred_list, axis=0)

best_proba, best_mcc, _ = eval_mcc(y_train, y_train_pred, True)
y_test_hat = (y_test_pred>=best_proba).astype(int)

save_submission(y_test_hat, 'ensembleCVsubmission4.csv', test_id)

