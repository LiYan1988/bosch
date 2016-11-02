# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 16:04:29 2016

@author: konrad
"""


import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier

if __name__ == '__main__':

    xtrain = pd.read_csv('../input/train_numeric.csv')
    xtrain.fillna(value = -99, inplace = True)
    
    id_train = xtrain['Id']
    ytrain = xtrain['Response']
    xtrain.drop(['Id', 'Response'], axis = 1, inplace = True)
    
    clf = ExtraTreesClassifier(n_estimators = 50, n_jobs = -1,
                                   min_samples_leaf = 10, verbose = 1)
    clf.fit(xtrain, ytrain)                                   
    
    xtest = pd.read_csv('../input/test_numeric.csv')
    xtest.fillna(value = -99, inplace = True)
    
    id_test = xtest['Id']
    xtest.drop(['Id'], axis = 1, inplace = True)
    
    pred = clf.predict_proba(xtest)
    
    xsub = pd.read_csv('../input/sample_submission.csv')
    xsub['Response'] = pred[:,1]
    xsub.to_csv('etrees_quickie.csv', index = False)