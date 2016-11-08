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

#%% Try xgboost, it's okay to put half of the data into memory
#x_train = read_data('train_numeric_feature200_set.pkl')
#x_train.fillna(-999)
#y_train = np.array(x_train.Response.values)
#x_train.drop(['Response'], axis=1, inplace=True)
#x_test = read_data('test_numeric_feature200_set.pkl')
#
#clf = xgb.XGBClassifier(max_depth=7, objective='binary:logistic', 
#                        learning_rate=0.005, colsample_bytree=0.4,
#                        min_child_weight=1, n_estimators=69, subsample=1,
#                        reg_alpha=2, reg_lambda=1)
##
#x_train0 = x_train.iloc[np.arange(0, x_train.shape[0], 2)]
#y_train0 = y_train[np.arange(0, x_train.shape[0], 2)]
#x_train1 = x_train.iloc[np.arange(1, x_train.shape[0], 2)]
#y_train1 = y_train[np.arange(1, x_train.shape[0], 2)]
#del x_train, y_train
#gc.collect()
#
#prior = 1.*y_train0.sum()/len(y_train0)
#clf.base_score = prior
#clf.fit(x_train0, y_train0, eval_set=[(x_train1, y_train1)], eval_metric='auc',
#                                      early_stopping_rounds=10, verbose=True)
#y_train1_pred = clf.predict_proba(x_train1)
#best_proba, best_mcc, _ = eval_mcc(y_train1, y_train1_pred, True)

#%% merge with date
#x_train = read_data('train_numeric_feature200_set.pkl')
#x_test = read_data('test_numeric_feature200_set.pkl')
#train_test_date = pd.read_csv('train_test_date2.csv')
##train_test_date.drop(['Response'], axis=1, inplace=True)
#x_train = pd.merge(x_train, train_test_date, how='left', left_on='Id', right_on='Id')
#x_test = pd.merge(x_test, train_test_date, how='left', left_on='Id', right_on='Id')
#save_data(x_train, 'train_numeric_feature200_set1.pkl')
#save_data(x_test, 'test_numeric_feature200_set1.pkl')
#
#%% try to train xgboost with date features

#x_train = read_data('train_numeric_feature200_set1.pkl')
#x_train.fillna(-999)
#y_train = np.array(x_train.Response.values)
#x_train.drop(['Response'], axis=1, inplace=True)
##x_test = read_data('test_numeric_feature200_set1.pkl')
#
#clf = xgb.XGBClassifier(max_depth=15, objective='binary:logistic', 
#                        learning_rate=0.01, colsample_bytree=0.4,
#                        min_child_weight=1, n_estimators=690, subsample=0.8,
#                        reg_alpha=2, reg_lambda=1)
##
#x_train0 = x_train.iloc[np.arange(0, x_train.shape[0], 2)]
#y_train0 = y_train[np.arange(0, x_train.shape[0], 2)]
#x_train1 = x_train.iloc[np.arange(1, x_train.shape[0], 2)]
#y_train1 = y_train[np.arange(1, x_train.shape[0], 2)]
#del x_train, y_train
#gc.collect()
#
#prior = 1.*y_train0.sum()/len(y_train0)
#clf.base_score = prior
#clf.fit(x_train0, y_train0, eval_set=[(x_train0, y_train0), 
#                                      (x_train1, y_train1)], 
#                                      eval_metric=mcc_eval,
#                                      early_stopping_rounds=10, 
#                                      verbose=True)
#y_train1_pred = clf.predict_proba(x_train1)[:, 1]
#best_proba, best_mcc, _ = eval_mcc(y_train1, y_train1_pred, True)

# train xgboost on the other subset of train
#del clf
#gc.collect()
#
#clf = xgb.XGBClassifier(max_depth=15, objective='binary:logistic', 
#                        learning_rate=0.01, colsample_bytree=0.4,
#                        min_child_weight=1, n_estimators=690, subsample=0.8,
#                        reg_alpha=2, reg_lambda=1)
#
#prior = 1.*y_train1.sum()/len(y_train1)
#clf.base_score = prior
#clf.fit(x_train1, y_train1, eval_set=[(x_train1, y_train1), 
#                                      (x_train0, y_train0)], 
#                                      eval_metric=mcc_eval,
#                                      early_stopping_rounds=10, 
#                                      verbose=True)
#y_train0_pred = clf.predict_proba(x_train0)[:, 1]
#best_proba, best_mcc, _ = eval_mcc(y_train0, y_train0_pred, True)

#%%
x_train = read_data('train_numeric_feature200_set1.pkl')
x_train.fillna(-999)
y_train = np.array(x_train.Response.values)
x_train.drop(['Response'], axis=1, inplace=True)

clf = xgb.XGBClassifier(max_depth=15, objective='binary:logistic', 
                        learning_rate=0.01, colsample_bytree=0.4,
                        min_child_weight=1, n_estimators=40, subsample=0.8,
                        reg_alpha=2, reg_lambda=1)

prior = 1.*y_train.sum()/len(y_train)
clf.base_score = prior
clf.fit(x_train, y_train, eval_set=[(x_train, y_train)], 
                                      eval_metric=mcc_eval,
                                      early_stopping_rounds=10, 
                                      verbose=True)
y_train_pred = clf.predict_proba(x_train)[:, 1]
best_proba, best_mcc, _ = eval_mcc(y_train, y_train_pred, True)

del x_train
gc.collect()

x_test = read_data('test_numeric_feature200_set1.pkl')
y_test_pred = clf.predict_proba(x_test)
y_test_pred = (y_test_pred>=best_proba).astype(int)
test_id = list(x_test.Id.values.ravel())
save_submission(y_test_pred, 'numericFeatureSubmission1.csv', test_id)

