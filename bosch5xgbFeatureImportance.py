# -*- coding: utf-8 -*-
"""
Created on Mon Nov 07 21:12:45 2016

@author: lyaa
"""

from boschStart2 import *

train_id_response = pd.read_csv('train_id_response.csv')

clf = xgb.XGBClassifier(max_depth=5, objective='binary:logistic', 
                        learning_rate=0.005, colsample_bytree=0.2,
                        min_child_weight=1.5, n_estimators=15, subsample=1,
                        reg_alpha=2, reg_lambda=1)

#%%
#feature_importance = []
#CHUNKSIZE = 100000
#for i, chunk in enumerate(pd.read_csv('input/train_numeric.csv', 
#                                      chunksize=CHUNKSIZE, 
#                                      dtype=np.float16)):
#    print('processing chunk {}'.format(i))
#    y_train = np.array(chunk['Response'].values)
#    chunk.drop(['Id', 'Response'], axis=1, inplace=True)
#    prior = 1.*y_train.sum()/len(y_train)
#    clf.fit(chunk, y_train)
#    feature_importance.append(clf.feature_importances_)
#    
#cols = list(chunk.columns)
#fi_mean = np.array(feature_importance).mean(axis=0)
#cols_good1 = [cols[i] for i in list(np.nonzero(fi_mean)[0].astype(int))]
#
#cols_out = np.setdiff1d(cols, cols_good1)
#save_data(cols_good1, 'selected_features.pkl')

#%%
#a = pd.read_csv('input/train_numeric.csv', nrows=2)
#cols_good1 = read_data('selected_features.pkl')
#cols_out = list(np.setdiff1d(list(a.columns), cols_good1))
#del a
#gc.collect()
#
#feature_importance = []
#CHUNKSIZE = 100000
#for i, chunk in enumerate(pd.read_csv('input/train_numeric.csv', 
#                                      chunksize=CHUNKSIZE, 
#                                      dtype=np.float16, 
#                                      usecols=cols_out)):
#    print('processing chunk {}'.format(i))
#    y_train = np.array(chunk['Response'].values)
#    chunk.drop(['Id', 'Response'], axis=1, inplace=True)
#    prior = 1.*y_train.sum()/len(y_train)
#    clf.fit(chunk, y_train)
#    feature_importance.append(clf.feature_importances_)
#    
#fi_mean = np.array(feature_importance).mean(axis=0)
#cols_good1 = [cols_out[i] for i in list(np.nonzero(fi_mean)[0].astype(int))]
#              
#save_data(cols_good2, 'selected_features2.pkl')

#%%
cols_good1 = read_data('selected_features.pkl')
cols_good2 = read_data('selected_features2.pkl')
feat_num_stat = read_data('feat_num_stat.pkl')
feat_num_stat= {i.columns[0]: i for i in feat_num_stat}
cols_selected = {i:feat_num_stat[i] for i in cols_good1}
cols_selected.update({i:feat_num_stat[i] for i in cols_good2})
cols_selected_sum = {i:cols_selected[i].Sum.sum() for i in cols_selected.keys()}
cols_selected_count = {i:cols_selected[i].Count.sum() for i in cols_selected.keys()}
                     
cols_good1.extend(cols_good2)
save_data(cols_good1, 'selected_features200.pkl')
                     