# -*- coding: utf-8 -*-
"""
Created on Sat Nov 05 08:20:58 2016

@author: lyaa
"""

from boschStart2 import *

#cols = pd.read_csv('input/train_numeric.csv', nrows=1)
#cols= list(cols.columns)
#cols.remove('Id')
#cols.remove('Response')
#Response = pd.read_csv('input/train_numeric.csv', usecols=['Response'], 
#                       dtype=np.float16)
#feat_num_stat = []
#for i, c in enumerate(cols):
#    feat_num = pd.read_csv('input/train_numeric.csv', usecols=[c], 
#                           dtype=np.float16)
#    feat_num['Response'] = Response.values
#    cnt = feat_num.groupby(c)['Response'].count().reset_index()
#    cnt.columns = [c, 'Count']
#    frq = feat_num.groupby(c)['Response'].sum().reset_index()
#    cnt['Sum'] = frq.Response
#    cnt['Frequency'] = cnt['Sum'].div(cnt['Count'], axis=0)
#    feat_num_stat.append(cnt)
#    print('Processed column {}, {}, {} events, {} positives'.format(i, c, 
#          feat_num_stat[-1].Count.sum(), feat_num_stat[-1].Sum.sum()))

# save data
# save_data(feat_num_stat, 'feat_num_stat.pkl')

# load data
feat_num_stat = read_data('feat_num_stat.pkl')

# mean frequency of nonzero frequencies
nz_freq = [(i, feat_num_stat[i].columns[0], 
            1.*feat_num_stat[i].loc[feat_num_stat[i].Frequency>0].\
            Sum.sum()/\
            feat_num_stat[i].loc[feat_num_stat[i].Frequency>0].Count.sum(), 
            int(feat_num_stat[i].loc[feat_num_stat[i].Frequency>0].Sum.sum()))
            for i in range(len(feat_num_stat))]
nz_freq = pd.DataFrame(nz_freq)
nz_freq.columns = ['index', 'feature', 'nz_freq', 'count']
nz_freq['count'].astype(int)
nz_freq_good = nz_freq.loc[(nz_freq['count']>50)&(nz_freq['nz_freq']>0.02)].\
                           sort_values(by=['nz_freq', 'count'], 
                                       ascending=False)
nz_freq_big = nz_freq.loc[(nz_freq['count']>1000)&(nz_freq['nz_freq']>0.0055)].\
                           sort_values(by=['count', 'nz_freq'], 
                                       ascending=False)
cols = pd.merge(nz_freq_big, nz_freq_good, how='outer', 
                on=list(nz_freq_big.columns))

save_data(cols, 'numeric_feature_set1.pkl')

#nz_min = [(i, feat_num_stat[i].columns[0], 
#           feat_num_stat[i].loc[feat_num_stat[i].Frequency>0].Frequency.min(),
#           int(feat_num_stat[i].loc[feat_num_stat[i].Frequency>0].Sum.sum()))
#            for i in range(len(feat_num_stat))]
#nz_min = pd.DataFrame(nz_min)
#nz_min.columns = ['index', 'feature', 'nz_min', 'count']
#nz_min['count'].astype(int)
#nz_min_good = nz_min.loc[(nz_min['count']>20)&(nz_min['nz_min']>0.05)].\
#                           sort_values(by=['nz_min', 'count'], 
#                                       ascending=False)
#nz_min_big = nz_min.loc[(nz_min['count']>1000)&(nz_min['nz_min']>0.006)].\
#                           sort_values(by=['count', 'nz_min'], 
#                                       ascending=False)

#nz_max = [(i, feat_num_stat[i].columns[0], 
#           feat_num_stat[i].loc[feat_num_stat[i].Frequency>0].Frequency.max(),
#            int(feat_num_stat[i].loc[feat_num_stat[i].Frequency>0].Sum.sum()))
#            for i in range(len(feat_num_stat))]
#nz_max = pd.DataFrame(nz_max)
#nz_max.columns = ['index', 'feature', 'nz_max', 'count']
#nz_max['count'].astype(int)

# experiment naive bayes
#x_train = pd.read_csv('input/train_numeric.csv', nrows=10000, 
#                      usecols=['Id', 'Response'].extend(list(cols.feature)))
#y_train = x_train.Response
#x_train.drop(Response, axis=1, inplace=True)