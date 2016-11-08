# -*- coding: utf-8 -*-
"""
Created on Tue Nov 08 08:41:15 2016

@author: lyaa
"""

from boschStart2 import *

#%% load data
#numeric_columns = read_data('selected_features200.pkl')
#types = {f: np.float16 for f in numeric_columns}
#numeric_columns.append('Id')
#types['Id'] = np.uint32
#numeric_columns.append('Response')
#types['Response'] = np.uint8
#
#train_numeric = []
#numeric_chunks = pd.read_csv('input/train_numeric.csv', 
#    chunksize=100000, dtype=types, usecols=numeric_columns)
#for i, dchunk in enumerate(numeric_chunks):
#    train_numeric.append(dchunk.to_sparse())
#    print(i, train_numeric[-1].memory_usage().sum(), 
#          dchunk.memory_usage().sum())
#    del dchunk
#    gc.collect()
#train_numeric = pd.concat(train_numeric)
#
#numeric_columns.remove('Response')
#del types['Response']
#test_numeric = []
#numeric_chunks = pd.read_csv('input/test_numeric.csv', 
#    chunksize=100000, dtype=types, usecols=numeric_columns)
#for i, dchunk in enumerate(numeric_chunks):
#    test_numeric.append(dchunk.to_sparse())
#    print(i, test_numeric[-1].memory_usage().sum(), 
#          dchunk.memory_usage().sum())
#    del dchunk
#    gc.collect()
#test_numeric = pd.concat(test_numeric)
#
#save_data(train_numeric, 'train_numeric_feature200_set.pkl')
#save_data(test_numeric, 'test_numeric_feature200_set.pkl')

#%% 
x_train = read_data('train_numeric_feature200_set.pkl')
x_train.fillna(-999)
y_train = np.array(x_train.Response.values)
x_train.drop(['Response'], axis=1, inplace=True)
x_test = read_data('test_numeric_feature200_set.pkl')

clf = xgb.XGBClassifier(max_depth=6, objective='binary:logistic', 
                        learning_rate=0.005, colsample_bytree=0.1,
                        min_child_weight=1, n_estimators=69, subsample=1,
                        reg_alpha=3, reg_lambda=0)
prior = 1.*y_train.sum()/len(y_train)

x_train0 = x_train.iloc[::2]
y_train0 = y_train.iloc[::2]
x_train1 = x_train.iloc[1::2]
y_train1 = y_train.iloc[1::2]
clf.fit(x_train0, y_train0, eval_set=[(x_train1, y_train1)], eval_metric='auc',
                                      early_stopping_rounds=10, verbose=True)