# -*- coding: utf-8 -*-
"""
Created on Thu Nov 03 21:17:53 2016

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

#np.random.seed(0)
#negidx_tmp = np.random.choice(negidx, len(posidx)*9, False)
#negidx_tmp = np.concatenate((negidx_tmp, posidx))
#negidx_tmp = np.sort(negidx_tmp)
#x_train_tmp = x_train.iloc[negidx_tmp]
#y_train_tmp = y_train.iloc[negidx_tmp]
#prior = np.sum(y_train_tmp)/(1.0*len(y_train_tmp))
#prior = prior.iloc[0]
#y_train_tmp = y_train_tmp.values.ravel()
#
clfxgb = xgb.XGBClassifier(objective='binary:logistic', silent=False, 
        seed=0, nthread=-1, gamma=1, subsample=0.6, learning_rate=0.1,
        n_estimators=50, max_depth=12)
#clfxgb.fit(x_train_tmp, y_train_tmp)
##y_pred = clfxgb.predict_proba(x_train.iloc[:1000])
#y_pred = []
#n_start = 0
#chunksize = 100000
#n_chunks = int(np.ceil(1.*n_test/chunksize))
#for i in range(n_chunks):
#    print('Predicting {} to {}'.format(n_start, min(n_test, n_start+chunksize)))
#    a = x_test.iloc[np.arange(n_start, min(n_test, n_start+chunksize))]
#    y_pred.append(clfxgb.predict_proba(a)[:,1])
#    n_start += chunksize
#    
#y_pred = np.concatenate(y_pred)

n_runs = 200
p_ratio = 0.05
random_state=0
mcs = MCSClassifier(clfxgb, n_runs, p_ratio, random_state)
#x_test_tmp = x_test.iloc[np.arange(1, 100)]
#mcs.one_run_(x_train, y_train, x_test_tmp)
y_pred = mcs.multi_run(x_train, y_train, x_test)
save_submission(y_pred, 'mcs_submission.csv', None)