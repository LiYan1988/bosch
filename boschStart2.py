# -*- coding: utf-8 -*-
"""
Created on Wed Nov 02 22:38:50 2016

@author: lyaa
"""

import numpy as np
import pandas as pd
import gc
import matplotlib.pyplot as plt

from sklearn import model_selection
from sklearn.metrics import matthews_corrcoef
import xgboost as xgb
#from numba import jit

#@jit
def mcc(tp, tn, fp, fn):
    sup = tp * tn - fp * fn
    inf = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if inf==0:
        return 0
    else:
        return sup / np.sqrt(inf)

#@jit
def eval_mcc(y_true, y_prob, show=False):
    idx = np.argsort(y_prob)
    y_true_sort = y_true[idx]
    n = y_true.shape[0]
    nump = 1.0 * np.sum(y_true) # number of positive
    numn = n - nump # number of negative
    tp = nump
    tn = 0.0
    fp = numn
    fn = 0.0
    best_mcc = 0.0
    best_id = -1
    prev_proba = -1
    best_proba = -1
    mccs = np.zeros(n)
    for i in range(n):
        # all items with idx < i are predicted negative while others are predicted positive
        # only evaluate mcc when probability changes
        proba = y_prob[idx[i]]
        if proba != prev_proba:
            prev_proba = proba
            new_mcc = mcc(tp, tn, fp, fn)
            if new_mcc >= best_mcc:
                best_mcc = new_mcc
                best_id = i
                best_proba = proba
        mccs[i] = new_mcc
        if y_true_sort[i] == 1:
            tp -= 1.0
            fn += 1.0
        else:
            fp -= 1.0
            tn += 1.0
    if show:
        y_pred = (y_prob >= best_proba).astype(int)
        score = matthews_corrcoef(y_true, y_pred)
        print(score, best_mcc)
        plt.plot(mccs)
        return best_proba, best_mcc, y_pred
    else:
        return best_mcc
        
def save_submission(y_pred, file_name, test_id):
    df = pd.DataFrame(data=y_pred, columns=['Response'])
    if test_id is None:
        test_id = pd.read_csv('input/test_numeric.csv', usecols=['Id'])
    df.index = list(test_id.values.ravel())
    df.index.name = 'Id'
    df.to_csv(file_name, index=True)
    
def create_date_features():
    # sample columns
    train_date_part = pd.read_csv('input/train_date.csv', nrows=10000)
    date_cols = train_date_part.drop('Id', axis=1).count().reset_index().\
        sort_values(by=0, ascending=False)
    date_cols['station'] = date_cols['index'].apply(lambda s: s.split('_')[1])
    date_cols = date_cols.drop_duplicates('station', keep='first')['index'].\
        tolist()
    del train_date_part
    # train start date
    train_start_date = pd.read_csv('input/train_date.csv', 
        usecols=['Id']+date_cols)
    train_start_date['start_date'] = train_start_date[date_cols].min(axis=1)
    train_start_date.drop(date_cols, axis=1, inplace=True)
    n_train = train_start_date.shape[0]
    # test start date
    test_start_date = pd.read_csv('input/test_date.csv', 
        usecols=['Id']+date_cols)
    test_start_date['start_date'] = test_start_date[date_cols].min(axis=1)
    test_start_date.drop(date_cols, axis=1, inplace=True)
    n_test = test_start_date.shape[0]
    # concate train and test
    start_date = pd.concat([train_start_date, test_start_date])
    del train_start_date, test_start_date
    gc.collect()
    
    train_id = pd.read_csv('input/train_numeric.csv', usecols=['Id'])
    test_id = pd.read_csv('input/test_numeric.csv', usecols=['Id'])
    train_id = train_id.merge(start_date, on='Id')
    test_id = test_id.merge(start_date, on='Id')
    train_test_id = pd.concat((train_id, test_id)).reset_index(drop=True).\
        reset_index(drop=True)
        
    train_test_id['stDiff1'] = train_test_id['start_date'].diff().\
        fillna(9999999).astype(float)
    train_test_id['stDiff2'] = train_test_id['start_date'].iloc[::-1].diff().\
        fillna(9999999).astype(float)
    train_test_id['idDiff1'] = train_test_id['Id'].diff().fillna(9999999).astype(int)
    train_test_id['idDiff2'] = train_test_id['Id'].iloc[::-1].diff().\
        fillna(9999999).astype(int)
    train_test_id = train_test_id.sort_values(by=['start_date', 'Id'], 
        ascending=True)
    train_test_id['idDiff3'] = train_test_id['Id'].diff().fillna(9999999).astype(int)
    train_test_id['idDiff4'] = train_test_id['Id'].iloc[::-1].diff().\
        fillna(9999999).astype(int)
    train_test_id['idDiff5'] = 1+2*(train_test_id['idDiff3']>1)+\
        1*(train_test_id['idDiff4']<-1)
    train_test_id['stDiff3'] = train_test_id['start_date'].diff().\
        fillna(9999999).astype(float)
    train_test_id['stDiff4'] = train_test_id['start_date'].iloc[::-1].diff().\
        fillna(9999999).astype(float)
    train_test_id.drop(['start_date'], axis=1, inplace=True)
    
    train_set = train_id.merge(train_test_id, on='Id')
    test_set = test_id.merge(train_test_id, on='Id')
    
    train_set.to_csv('train_date_feats.csv')
    test_set.to_csv('test_date_feats.csv')

if __name__=='__main__':
    print('main')
    