# -*- coding: utf-8 -*-
"""
Created on Fri Nov 04 23:24:20 2016

@author: lyaa
"""

from boschStart2 import *
#
#x_train = pd.read_csv('input/train_categorical.csv', usecols=[1], dtype=np.str)
#response = pd.read_csv('input/train_numeric.csv', usecols=['Response'], 
#                       dtype=np.uint8)

#x_train['Response'] = response.values

cols = pd.read_csv('input/train_categorical.csv', nrows=1)
cols= list(cols.columns)
cols.remove('Id')
Response = pd.read_csv('input/train_numeric.csv', usecols=['Response'], 
                       dtype=np.float16)
feat_cat_cnt = []
#feat_cat_frq = []
for i, c in enumerate(cols):
    print('Processing column {}, {}'.format(i, c))
    feat_cat = pd.read_csv('input/train_categorical.csv', usecols=[c], 
                           dtype=np.str)
    feat_cat['Response'] = Response.values
    feat_cat_cnt.append(feat_cat.groupby(c)['Response'].count().reset_index())
#    feat_cat_frq.append(feat_cat.groupby(c)['Response'].mean())
#    if i > 2:
#        break
