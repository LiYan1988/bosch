# -*- coding: utf-8 -*-
"""
Created on Wed Nov 02 22:38:50 2016

@author: lyaa
"""

import numpy as np
import pandas as pd
import gc

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