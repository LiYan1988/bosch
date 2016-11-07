# -*- coding: utf-8 -*-
"""
Created on Sun Nov 06 19:54:30 2016

@author: lyaa
"""

from boschStart2 import *

#%% prepare data
#numeric_columns = read_data('numeric_feature_set1.pkl')
#numeric_columns = list(numeric_columns.feature)
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
#save_data(train_numeric, 'train_numeric_feature_set1.pkl')
#save_data(test_numeric, 'test_numeric_feature_set1.pkl')

#%%
#train_numeric = read_data('train_numeric_feature_set1.pkl')
#test_numeric = read_data('test_numeric_feature_set1.pkl')
#test_numeric['Response'] = 0
#
#train_test = pd.concat([train_numeric, test_numeric])
##del train_numeric, test_numeric
##gc.collect()
#
#cols = list(train_test.columns)
#cols.remove('Id')
#cols.remove('Response')
#
#for i, c in enumerate(cols):
#    print('processing {}, {}'.format(i, c))
#    n_bins = min(51, len(train_test[c].value_counts()))
#    bins = np.linspace(train_test[c].min(), train_test[c].max(), n_bins)
#    labels = np.arange(n_bins-1)
#    train_test[c] = pd.cut(train_test[c], bins, labels=labels)
#
#train_id = np.array(train_numeric.Id.values)
#test_id = np.array(test_numeric.Id.values)
#train_numeric = train_test.loc[train_test.Id.isin(train_id)]
#test_numeric = train_test.loc[train_test.Id.isin(test_id)]
#save_data(train_numeric, 'train_numeric_feature_set1.pkl')
#save_data(test_numeric, 'test_numeric_feature_set1.pkl')

#%% 
#train_numeric = read_data('train_numeric_feature_set1.pkl')
#test_numeric = read_data('test_numeric_feature_set1.pkl')
#
#train_id = np.array(train_numeric.Id.values)
#test_id = np.array(test_numeric.Id.values)
## the samples in training set 2
#np.random.seed(0)
#train_id_meta = np.random.choice(train_id, len(train_id)/2, replace=False)
#train_id_oob = np.setdiff1d(train_id, train_id_meta)
#train_meta = train_numeric[train_numeric.Id.isin(train_id_meta)].copy()
##train_oob = train_numeric[train_numeric.Id.isin(train_id_oob)].copy()
#
#cols = list(train_numeric.columns)
#cols.remove('Id')
#c = cols[0]
#n_bins = min(51, len(train_numeric[c].value_counts()))
##n_bins = len(train_numeric[c].value_counts())
#bins = np.linspace(train_numeric[c].min(), train_numeric[c].max(), n_bins)
#labels = np.arange(n_bins-1).astype(np.uint8)
#train_numeric[c] = pd.cut(train_numeric[c], bins, labels=labels)
#likelihood = train_numeric.groupby(c)['Response'].mean().reset_index()
##likelihood.rename(c+'_likelihood')
##counts = train_numeric.groupby(c)['Response'].apply(lambda x: len(x)).reset_index()
##plt.plot(likelihood[c], likelihood.Response)
##plt.plot(counts[c], counts.Response)
##pd.merge(train_oob, likelihood, how='left', left_on=

#%%
#train_numeric = read_data('train_numeric_feature_set1.pkl')
#x_test = read_data('test_numeric_feature_set1.pkl')
#x_test.fillna(-999, inplace=True)
#
#train_id = np.array(train_numeric.Id.values)
##test_id = np.array(test_numeric.Id.values)
## the samples in training set 2
#np.random.seed(0)
#train_id_meta = np.random.choice(train_id, len(train_id)/2, replace=False)
#train_id_oob = np.setdiff1d(train_id, train_id_meta)
#x_train_meta = train_numeric[train_numeric.Id.isin(train_id_meta)].copy()
#x_train_oob = train_numeric[train_numeric.Id.isin(train_id_oob)].copy()
#y_train_meta = x_train_meta['Response']
#x_train_meta.drop('Response', axis=1, inplace=True)
#y_train_oob = x_train_oob['Response']
#x_train_oob.drop('Response', axis=1, inplace=True)
#x_train_meta.fillna(-999, inplace=True)
#x_train_oob.fillna(-999, inplace=True)
##
#clfmeta = ensemble.RandomForestClassifier(n_estimators=100, max_depth=10,
#                                          random_state=0, verbose=10, 
#                                          n_jobs=-1)
#clfmeta.fit(x_train_meta, y_train_meta)
#y_pred = clfmeta.predict_proba(x_train_oob)[:, 1]
#x_train_oob['meta_feature'] = y_pred
#y_pred = clfmeta.predict_proba(x_test)[:, 1]
#x_test['meta_feature'] = y_pred
#
#del clfmeta, x_train_meta, train_numeric
#gc.collect()
#
#y_train_oob = np.array(y_train_oob.values)
##x_train_oob = x_train_oob.as_matrix()
##x_test = x_test.as_matrix()
#prior = y_train_oob.sum()/(1.*len(y_train_oob))
#clf = xgb.XGBClassifier(max_depth=7, objective='binary:logistic', 
#                        learning_rate=0.021, colsample_bytree=0.82,
#                        min_child_weight=3, base_score=prior, 
#                        n_estimators=69)
#clf.fit(x_train_oob, y_train_oob, eval_set=[(x_train_oob, y_train_oob)], 
#        eval_metric='auc', verbose=10, early_stopping_rounds=10)
#y_test_pred = clf.predict_proba(x_test)[:, 1]
##save_data(y_test_pred, 'y_test_pred_tmp.pkl')
#y_train_oob_pred = clf.predict_proba(x_train_oob)[:, 1]
#best_proba, best_mcc, _ = eval_mcc(y_train_oob, y_train_oob_pred, True)
#y_test_pred = (y_test_pred>=best_mcc).astype(np.uint8)
#save_submission(y_test_pred, 'numericFeatureSubmission.csv', None)

#%%
#train_numeric = read_data('train_numeric_feature_set1.pkl')
#train_id = np.array(train_numeric.Id.values)
#np.random.seed(0)
#train_meta_id = np.random.choice(train_id, len(train_id)/2, replace=False)
#train_work_id = np.setdiff1d(train_id, train_meta_id)
#del train_numeric
#gc.collect()
#
#estimator_meta = ensemble.RandomForestClassifier(n_estimators=8, 
#                                                 max_depth=10,
#                                                 random_state=0, 
#                                                 verbose=10, 
#                                                 n_jobs=-1)
#train_work_new_feature, test_new_feature, y_train_work = \
#    create_numeric_meta_features(train_meta_id, estimator_meta)
#    
##%%
#train_numeric = read_data('train_numeric_feature_set1.pkl')
#train_id = np.array(train_numeric.Id.values)
#np.random.seed(0)
#train_meta_id = np.random.choice(train_id, len(train_id)/2, replace=False)
#del train_numeric
#
#x_train_work_date, x_test_date, _ = \
#    create_date_merta_features(train_meta_id, id_window_sizes=[3], 
#                               time_window_sizes=[0.5])
#
##%%
#train_numeric = read_data('train_numeric_feature_set1.pkl')
#x_train_work = train_numeric[train_numeric.Id.isin(train_work_id)].copy()
#y_train_work = x_train_work['Response']
#x_train_work.drop('Response', axis=1, inplace=True)
#x_train_work.fillna(-999, inplace=True)
#del train_numeric
#gc.collect()
#
#x_train_work['new_numeric_feature'] = train_work_new_feature
#x_train_work = pd.merge(x_train_work, x_train_work_date, how='left', 
#                        left_on='Id', right_on='Id')
#
#x_test = read_data('test_numeric_feature_set1.pkl')
#x_test['new_numeric_feature'] = test_new_feature
#x_test = pd.merge(x_test, x_test_date, how='left', 
#                        left_on='Id', right_on='Id')

#%%
#train_numeric = read_data('train_numeric_feature_set1.pkl')
#train_id = np.array(train_numeric.Id.values)
#np.random.seed(0)
#train_meta_id = np.random.choice(train_id, len(train_id)/2, replace=False)
#train_work_id = np.setdiff1d(train_id, train_meta_id)
#estimator1 = ensemble.RandomForestClassifier(n_estimators=8, 
#                                                 max_depth=10,
#                                                 random_state=0, 
#                                                 verbose=10, 
#                                                 n_jobs=-1)
#estimator2 = ensemble.RandomForestClassifier(n_estimators=8, 
#                                                 max_depth=10,
#                                                 random_state=1, 
#                                                 verbose=10, 
#                                                 n_jobs=-1)
#meta_estimators = [estimator1, estimator2]
#x_train_work1, y_train_work1, x_test1 = \
#    create_train_test_sets(train_meta_id, meta_estimators, 
#                           id_window_sizes=[3, 5, 7, 9], 
#                           time_window_sizes=[0.01, 0.05, 0.1, 0.5])
#
#clf = xgb.XGBClassifier(max_depth=7, objective='binary:logistic', 
#                        learning_rate=0.021, colsample_bytree=0.82,
#                        min_child_weight=3, base_score=prior, 
#                        n_estimators=69)
#clf.fit(x_train_work1, y_train_work1)
#y_train_work1_pred = clf.predict_proba(x_train_work1)
#del x_train_work1
#y_test_pred1 = clf.predict_proba(x_test1)

#%%
train_numeric = read_data('train_numeric_feature_set1.pkl')
train_id = np.array(train_numeric.Id.values)
np.random.seed(0)
train_meta_id = np.random.choice(train_id, len(train_id)/2, replace=False)
train_work_id = np.setdiff1d(train_id, train_meta_id)

estimator1 = ensemble.RandomForestClassifier(n_estimators=8, 
                                                 max_depth=1,
                                                 random_state=0, 
                                                 verbose=10, 
                                                 n_jobs=-1)
estimator2 = ensemble.RandomForestClassifier(n_estimators=8, 
                                                 max_depth=1,
                                                 random_state=1, 
                                                 verbose=10, 
                                                 n_jobs=-1)
meta_estimators = [estimator1, estimator2]
clf = xgb.XGBClassifier(max_depth=3, objective='binary:logistic', 
                        learning_rate=0.021, colsample_bytree=0.82,
                        min_child_weight=3, n_estimators=1)
    
y_train_pred, y_test_pred = \
    meta_predict(clf, meta_estimators, id_window_sizes=[5], 
                 time_window_sizes=[0.5], random_state=0)