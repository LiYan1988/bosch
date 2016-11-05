# -*- coding: utf-8 -*-
"""
Created on Sat Nov 05 08:20:58 2016

@author: lyaa
"""

from boschStart2 import *
#
#x_train = pd.read_csv('input/train_categorical.csv', usecols=[1], dtype=np.str)
#response = pd.read_csv('input/train_numeric.csv', usecols=['Response'], 
#                       dtype=np.uint8)

#x_train['Response'] = response.values

cols = pd.read_csv('input/train_numeric.csv', nrows=1)
cols= list(cols.columns)
cols.remove('Id')
cols.remove('Response')
Response = pd.read_csv('input/train_numeric.csv', usecols=['Response'], 
                       dtype=np.float16)
feat_num_stat = []
for i, c in enumerate(cols):
    feat_num = pd.read_csv('input/train_numeric.csv', usecols=[c], 
                           dtype=np.float16)
    feat_num['Response'] = Response.values
    cnt = feat_num.groupby(c)['Response'].count().reset_index()
    cnt.columns = [c, 'Count']
    frq = feat_num.groupby(c)['Response'].sum().reset_index()
    cnt['Sum'] = frq.Response
    cnt['Frequency'] = cnt['Sum'].div(cnt['Count'], axis=0)
    feat_num_stat.append(cnt)
    print('Processed column {}, {}, {} events, {} positives'.format(i, c, 
          feat_num_stat[-1].Count.sum(), feat_num_stat[-1].Sum.sum()))
#    plt.plot(feat_num_stat[-1][c], feat_num_stat[-1].Frequency)
#    if i > 2:
#        break