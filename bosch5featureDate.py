# -*- coding: utf-8 -*-
"""
Created on Sat Nov 05 14:06:46 2016

@author: lyaa
"""

from boschStart2 import *

#%% Load date
#chunksize = 100000
#n_train = 1183747
#n_test = 1183748
#
#train_date = pd.read_csv('input/train_date.csv', usecols=['Id'], 
#                         dtype=np.uint32)
#test_date = pd.read_csv('input/test_date.csv', usecols=['Id'],
#                        dtype=np.uint32)
#cols = pd.read_csv('input/train_date.csv', nrows=10)
#cols = list(cols.columns)
#cols.remove('Id')
#
#types = {'Id':np.uint32}
#types.update({i:np.float16 for i in cols})
#
#train_date['min_date'] = -1
#train_date['max_date'] = -1
#test_date['min_date'] = -1
#test_date['max_date'] = -1
#
#nrows = 0
#train_date_list = []
#for i, tr in enumerate(pd.read_csv('input/train_date.csv', 
#                                   chunksize=chunksize, dtype=types)):
#    print('processing train date, row {} to {}'.\
#          format(tr.Id.iloc[0], tr.Id.iloc[-1]))
#    train_date.loc[train_date.Id.isin(tr.Id), 'min_date'] = \
#                   tr[cols].min(axis=1).values
#    train_date.loc[train_date.Id.isin(tr.Id), 'max_date'] = \
#                   tr[cols].max(axis=1).values
#    train_date_list.append(tr.to_sparse())
#    del tr
#    gc.collect()
#
#train_date_list = pd.concat(train_date_list)
#save_data(train_date_list, 'train_date_all.pkl')
#
#nrows = 0
#test_date_list = []
#for i, te in enumerate(pd.read_csv('input/test_date.csv', 
#                                   chunksize=chunksize, dtype=types)):
#    print('processing test date, row {} to {}'.\
#          format(te.Id.iloc[0], te.Id.iloc[-1]))
#    test_date.loc[test_date.Id.isin(te.Id), 'min_date'] = \
#                  te[cols].min(axis=1).values
#    test_date.loc[test_date.Id.isin(te.Id), 'max_date'] = \
#                  te[cols].max(axis=1).values
#    test_date_list.append(te.to_sparse())
#    del te
#    gc.collect()
#
#test_date_list = pd.concat(test_date_list)
#save_data(test_date_list, 'test_date_all.pkl')
    
#train_date.to_csv('train_date_pure.csv')
#test_date.to_csv('test_date_pure.csv')

#%% Process date
#train_date = pd.read_csv('train_date_all.csv')
#train_date.drop('Unnamed: 0', axis=1, inplace=True)
#test_date = pd.read_csv('test_date_all.csv')
#test_date.drop('Unnamed: 0', axis=1, inplace=True)
#
#Response = pd.read_csv('input/train_numeric.csv', usecols=['Id', 'Response'])
#train_date['Response'] = Response.Response
#test_date['Response'] = 0
#train_test = pd.concat([train_date, test_date])
##del train_date, test_date
##gc.collect()
#
#train_test['duration'] = train_test.max_date-train_test.min_date
#
#train_test['date_diff1'] = train_test['min_date'].diff().\
#    fillna(99999999).astype(float).values
#train_test['date_diff2'] = train_test['min_date'].iloc[::-1].diff().\
#    fillna(99999999).astype(float).values
#train_test['id_diff1'] = train_test['Id'].diff().\
#    fillna(99999999).astype(int).values
#train_test['id_diff2'] = train_test['Id'].iloc[::-1].diff().\
#    fillna(99999999).astype(int).values
#train_test= train_test.sort_values(by=['min_date', 'Id'], ascending=True)
#train_test['id_diff3'] = train_test['Id'].diff().fillna(99999999).\
#    astype(int).values
#train_test['id_diff4'] = train_test['Id'].iloc[::-1].diff().\
#    fillna(99999999).astype(int).values
#train_test['id_diff5'] = 1+2*(train_test['id_diff3']>1)+\
#    1*(train_test['id_diff4']<-1).values
#train_test['date_diff3'] = train_test['min_date'].diff().\
#    fillna(99999999).astype(float).values
#train_test['date_diff4'] = train_test['min_date'].iloc[::-1].diff().\
#    fillna(99999999).astype(float).values
#
#train_test.to_csv('train_test_date.csv', index_label=False)

#%% consective Id have more positives
#pos = Response.Id[Response.Response==1].values
#pos = train_test.loc[train_test.Id.isin(pos)].copy()
#pos = pos.sort_values(by='Id')
#pos['diff'] = pos['Id'].diff().fillna(999999999).astype(int).values
#a = pos['diff'].value_counts().sort_index()

#%% consective start time
#pos = Response.Id[Response.Response==1].values
#pos = train_test.loc[train_test.Id.isin(pos)].copy()
#pos = pos.sort_values(by='min_date')
#pos['diff'] = pos['min_date'].diff().fillna(999999999).astype(float).\
#    round(2).values
#a = pos['diff'].value_counts().sort_index()
#plt.plot(np.log(a.index.values), a.values)

#%% consective end time
#pos = Response.Id[Response.Response==1].values
#pos = train_test.loc[train_test.Id.isin(pos)].copy()
#pos = pos.sort_values(by='max_date')
#pos['diff'] = pos['max_date'].diff().fillna(999999999).astype(float).\
#    round(2).values
#a = pos['diff'].value_counts().sort_index()
#plt.plot(np.log(a.index.values), a.values)

#%% consective end time
#pos = Response.Id[Response.Response==1].values
#pos = train_test.loc[train_test.Id.isin(pos)].copy()
#pos = pos.sort_values(by='duration')
#pos['diff'] = pos['duration'].diff().fillna(999999999).astype(float).\
#    round(2).values
#a = pos['diff'].value_counts().sort_index()
#plt.plot(np.log(a.index.values), a.values)

#%% 
#train_test = pd.read_csv('train_test_date.csv')
#train_id = pd.read_csv('train_date_pure.csv', usecols=['Id']).Id.values
## the samples in training set 2
#np.random.seed(0)
#train_id_part = np.random.choice(train_id, len(train_id)/2)
#
#train_test.loc[train_test.Id.isin(train_id_part), 'Response'] = 0
#train_test.sort_values(by='Id', ascending=True, inplace=True)
#train_test['id_window'] = train_test.Response.rolling(window=10, 
#    min_periods=1, center=True).sum()
#
#bin_interval = 10
#n_bins = int(1718.48/bin_interval)
#train_test['min_date_binned'] = \
#    pd.cut(train_test['min_date'], bins=n_bins, labels=np.arange(n_bins))
#binned = train_test.groupby('min_date_binned')['Response'].sum().fillna(0).\
#    to_frame()
#train_test = pd.merge(train_test, binned, how='left', 
#                      left_on='min_date_binned',
#                      right_index=True, suffixes=('', '_min_date_windowed'))
#train_test.drop('min_date_binned', axis=1, inplace=True)
#cols = train_test.columns
#cols[-1] = ''

#%%
#train_id = pd.read_csv('train_date_pure.csv', usecols=['Id']).Id.values
## the samples in training set 2
#np.random.seed(0)
#train_id_part = np.random.choice(train_id, len(train_id)/2)
#df = create_bag_features(train_id_part, id_window_sizes=[3, 5, 7, 9],
#                        time_window_sizes=[0.01, 0.05, 0.1, 0.5])

#%% Process train_date_all and test_date_all
train_date = pd.read_csv('train_date_pure.csv')
test_date = pd.read_csv('test_date_pure.csv')
train_id = train_date.Id.copy()
test_id = test_date.Id.copy()
train_test = pd.concat([train_date, test_date])
del train_date, test_date
gc.collect()

#cols = list(train_test.columns)
#cols.remove('Id')
#train_test['StartTime'] = train_test.min(axis=1).values
train_test.drop('Unnamed: 0', axis=1, inplace=True)
train_test = train_test.reset_index(drop=True).reset_index(drop=False)
train_test['magic1'] = train_test['Id'].diff().fillna(9999999).astype(int)
train_test['magic2'] = train_test['Id'].iloc[range(len(train_id)-1, -1, -1)].\
    diff().fillna(9999999).astype(int)
train_test['tDiff1'] = train_test['min_date'].diff().fillna(9999999).astype(float)
train_test['tDiff2'] = train_test['min_date'].iloc[range(len(train_id)-1, -1, -1)].\
    diff().fillna(9999999).astype(float)
train_test['tDiff3'] = train_test['max_date'].diff().fillna(9999999).astype(float)
train_test['tDiff4'] = train_test['max_date'].iloc[range(len(train_id)-1, -1, -1)].\
    diff().fillna(9999999).astype(float)
train_test = train_test.sort_values(by=['min_date', 'Id'], ascending=True)
train_test['magic3'] = train_test['Id'].diff().fillna(9999999).astype(int)
train_test['magic4'] = train_test['Id'].iloc[range(len(train_id)-1, -1, -1)].\
    diff().fillna(9999999).astype(int)
train_test['magic5'] = 1+2*(train_test['magic3']>1)+1*(train_test['magic4']<-1)
train_test['tDiff5'] = train_test['min_date'].diff().fillna(9999999).astype(float)
train_test['tDiff6'] = train_test['min_date'].iloc[range(len(train_id)-1, -1, -1)].\
    diff().fillna(9999999).astype(float)
train_test['tDiff7'] = train_test['max_date'].diff().fillna(9999999).astype(float)
train_test['tDiff8'] = train_test['max_date'].iloc[range(len(train_id)-1, -1, -1)].\
    diff().fillna(9999999).astype(float)

train_test['duration'] = train_test['max_date']-train_test['min_date']


train_test = train_test.sort_values(by=['index']).drop(['index'], axis=1)
train = train_test.iloc[range(len(train_id)), :]
train_id = pd.read_csv('train_id_response.csv')
train = pd.merge(train, train_id, how='left', left_on='Id', right_on='Id')

axes_lims = [-10, 10]
twoplot(train, 'magic5', axes_lims)

#test = pd.merge(test, train_test, how='left', left_on='Id', right_on='Id')
train_test.to_csv('train_test_date2.csv', index_label=False)