# -*- coding: utf-8 -*-
"""
Created on Wed Nov 02 21:18:08 2016

@author: lyaa
"""

import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#import seaborn as sns
import gc
#sns.set_style('whitegrid')

import xgboost as xgb

STATIONS = ['S32', 'S33', 'S34']
train_date_part = pd.read_csv('input/train_date.csv', nrows=10000)
date_cols = train_date_part.drop('Id', axis=1).count().reset_index().sort_values(by=0, ascending=False)
date_cols['station'] = date_cols['index'].apply(lambda s: s.split('_')[1])
date_cols = date_cols[date_cols['station'].isin(STATIONS)]
date_cols = date_cols.drop_duplicates('station', keep='first')['index'].tolist()
print(date_cols)
train_date = pd.read_csv('input/train_date.csv', usecols=['Id'] + date_cols)
print(train_date.columns)
train_date.columns = ['Id'] + STATIONS
for station in STATIONS:
    train_date[station] = 1 * (train_date[station] >= 0)
response = pd.read_csv('input/train_numeric.csv', usecols=['Id', 'Response'])
print(response.shape)
train = response.merge(train_date, how='left', on='Id')
# print(train.count())
train.head(3)

train['cnt'] = 1
failure_rate = train.groupby(STATIONS).sum()[['Response', 'cnt']]
failure_rate['failure_rate'] = failure_rate['Response'] / failure_rate['cnt']
failure_rate = failure_rate[failure_rate['cnt'] > 1000]  # remove 
failure_rate.head(20)

train_date_part = pd.read_csv('input/train_date.csv', nrows=10000)
date_cols = train_date_part.drop('Id', axis=1).count().reset_index().sort_values(by=0, ascending=False)
date_cols['station'] = date_cols['index'].apply(lambda s: s.split('_')[1])
date_cols = date_cols.drop_duplicates('station', keep='first')['index'].tolist()
# Train start dates
train_start_date = pd.read_csv('input/train_date.csv', usecols=['Id'] + date_cols)
train_start_date['start_date'] = train_start_date[date_cols].min(axis=1)
train_start_date = train_start_date.drop(date_cols, axis=1)
print(train_start_date.shape)
# Test start dates
test_start_date = pd.read_csv('input/test_date.csv', usecols=['Id'] + date_cols)
test_start_date['start_date'] = test_start_date[date_cols].min(axis=1)
test_start_date = test_start_date.drop(date_cols, axis=1)
print(test_start_date.shape)
start_date = pd.concat([train_start_date, test_start_date])
print(start_date.shape)
del train_start_date, test_start_date
gc.collect()
start_date.head()

train_id = pd.read_csv('input/train_numeric.csv', usecols=['Id'])
test_id = pd.read_csv('input/test_numeric.csv', usecols=['Id'])
train_id = train_id.merge(start_date, on='Id')
test_id = test_id.merge(start_date, on='Id')
train_test_id = pd.concat((train_id, test_id)).reset_index(drop=True).reset_index(drop=False)
train_test_id = train_test_id.sort_values(by=['start_date', 'Id'], ascending=True)
train_test_id['IdDiff1'] = train_test_id['Id'].diff().fillna(9999999).astype(int)
train_test_id['IdDiff2'] = train_test_id['Id'].iloc[::-1].diff().fillna(9999999).astype(int)
train_test_id['Magic'] = 1 + 2 * (train_test_id['IdDiff1'] > 1) + 1 * (train_test_id['IdDiff2'] < -1)

train_with_magic = train.merge(train_test_id[['Id', 'Magic']], on='Id')
train_with_magic.head()

