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
feat_cat_stat = []
for i, c in enumerate(cols):
    feat_cat = pd.read_csv('input/train_categorical.csv', usecols=[c], 
                           dtype=np.str)
    feat_cat['Response'] = Response.values
    cnt = feat_cat.groupby(c)['Response'].count().reset_index()
    cnt.columns = [c, 'Count']
    pos = feat_cat.groupby(c)['Response'].sum().reset_index()
    cnt['Sum'] = pos['Response']
    cnt['Frequency'] = cnt['Sum'].div(cnt['Count'], axis=0)
    feat_cat_stat.append(cnt)
    print('Processed column {}, {}, {} events, {} positives'.format(i, c, 
          feat_cat_stat[-1].Count.sum(), feat_cat_stat[-1].Sum.sum()))
#    if i > 2:
#        break
