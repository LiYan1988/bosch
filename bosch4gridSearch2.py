# -*- coding: utf-8 -*-
"""
Created on Fri Nov 04 21:11:58 2016

@author: lyaa
"""

from boschStart2 import *

n_train = 1183747
n_test = 1183748

x_train = read_data('x_train_partcols.pkl')
y_train = read_data('y_train.pkl')
x_test = read_data('x_test_partcols.pkl')

posidx = np.array(read_data('posidx.pkl'))
negidx = np.array(read_data('negidx.pkl'))

np.random.seed(0)
negidx_tmp = np.random.choice(negidx, len(posidx)*19, False)
negidx_tmp = np.concatenate((negidx_tmp, posidx))
negidx_tmp = np.sort(negidx_tmp)
x_train_tmp = x_train.iloc[negidx_tmp]
y_train_tmp = y_train.iloc[negidx_tmp]
prior = np.sum(y_train_tmp)/(1.0*len(y_train_tmp))
prior = prior.iloc[0]
y_train_tmp = y_train_tmp.values.ravel()
#
clfxgb = xgb.XGBClassifier(objective='binary:logistic', silent=False, 
        seed=0, nthread=-1, gamma=1, subsample=0.6, learning_rate=0.1,
        n_estimators=150, max_depth=12)

params = {}
#params['subsample'] = [0.5, 0.6, 0.7]
params['learning_rate'] = [0.05]
params['n_estimators'] = [400, 800, 1200, 1600, 2000]
params['max_depth'] = [16, 20, 24, 28, 32]

gridcv = model_selection.GridSearchCV(clfxgb, params, scoring='roc_auc', 
    verbose=10)
gridcv.fit(x_train_tmp, y_train_tmp)