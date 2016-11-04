# -*- coding: utf-8 -*-
"""
Created on Thu Nov 03 21:17:53 2016

@author: lyaa
"""

from boschStart2 import *



x_train = read_data('x_train_partcols.pkl')
y_train = read_data('y_train.pkl')
x_test = read_data('x_test_partcols.pkl')
prior = np.sum(y_train)/(1.0*len(y_train))
prior = prior.iloc[0]

posidx = read_data('posidx.pkl')
negidx = read_data('negidx.pkl')

clfxgb = xgb.XGBClassifier(objective='binary:logistic', silent=False, 
        seed=0, nthread=-1, gamma=1, subsample=0.8, learning_rate=0.1,
        n_estimators=2, max_depth=6, base_score=prior)
clfxgb.fit(x_train, y_train)

