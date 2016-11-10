# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 21:24:12 2016

@author: lyaa
"""

from boschStart2 import *

#%% load data
#x_train = read_data('train_numeric_feature200_set1.pkl')
#x_train.fillna(-999, inplace=True)
#y_train = np.array(x_train.Response.values)
#response = x_train[['Id', 'Response']].copy()
#x_train.drop(['Response'], axis=1, inplace=True)
#x_test = read_data('test_numeric_feature200_set1.pkl')
#x_test.fillna(-999, inplace=True)

#x_train = x_train.iloc[np.arange(100000)]
#y_train = y_train[np.arange(100000)]
#x_test = x_test.iloc[np.arange(100000, 200000)]
#x_train = x_train.iloc[:, range(20)]
#x_test = x_test.iloc[:, range(20)]

#%% split data into meta and bag sets
#np.random.seed(0)
#train_id = np.array(x_train.Id.values)
#
#y_test_pred = np.zeros((x_test.shape[0],))
#y_train_pred = y_train.copy()
#y_train_pred = y_train_pred.astype(np.float16)
#y_train_pred.fill(0.)
#
#n_cv = 5
#kf = model_selection.StratifiedKFold(n_splits=n_cv, shuffle=True, random_state=2)
#
#meta_estimator_rf = ensemble.RandomForestClassifier(n_estimators=100, max_depth=7, n_jobs=-1)
#meta_estimator_ext = ensemble.ExtraTreesClassifier(n_estimators=100, max_depth=7, n_jobs=-1)
#meta_estimators = {'rf1':meta_estimator_rf, 'ext1':meta_estimator_ext}
#
#clf = xgb.XGBClassifier(max_depth=11, objective='binary:logistic', 
#                        learning_rate=0.01, colsample_bytree=0.4,
#                        min_child_weight=1, n_estimators=90, subsample=0.6,
#                        reg_alpha=2, reg_lambda=1)
#
#for bag_id, meta_id in kf.split(x_train, y_train):
#    x_train_meta = x_train.iloc[meta_id].copy()
#    y_train_meta = response.iloc[meta_id].copy()
#    y_train_meta = np.array(y_train_meta.Response.values)
#    x_train_meta['Response'] = y_train_meta
#    
#    x_train_bag = x_train.iloc[bag_id].copy()
#    y_train_bag = response.iloc[bag_id].copy()
#    y_train_bag = np.array(y_train_bag.Response.values)
#    
#    x_test_copy = x_test.copy()
#    
#    cols = list(x_train_meta.columns)
#    cols.remove('Id')
#    cols.remove('Response')
#    
#    for i, c in enumerate(cols):
#        print 'column {}, {}'.format(i, c)
#        b = x_train_meta.groupby(c)['Response'].mean().reset_index()
#        b.columns = [c, c+'_b']
#        x_train_bag = pd.merge(x_train_bag, b, how='left', left_on=c, right_on=c)
#        x_train_bag.drop(c, axis=1, inplace=True)
#        x_test_copy = pd.merge(x_test_copy, b, how='left', left_on=c, right_on=c)
#        x_test_copy.drop(c, axis=1, inplace=True)
#    
#    print('Training outer classifier...')
#    prior = 1.*y_train_bag.sum()/len(y_train_bag)
#    clf.base_score = prior
#    clf.fit(x_train_bag, y_train_bag)
#    print('Predicting on bag set')
#    y_train_pred[bag_id] += clf.predict_proba(x_train_bag)[:, 1]
#    del x_train_bag
#    gc.collect()
#    
#    print('Predicting on test set')
#    y_test_pred += clf.predict_proba(x_test_copy)[:, 1]
#    del x_test_copy
#    gc.collect()
#    
#
#y_train_pred /= 1.*(n_cv-1)
#y_test_pred /= 1.*n_cv
#
#best_proba, best_mcc, _ = eval_mcc(y_train, y_train_pred, True)
#y_test_hat = (y_test_pred>=best_proba).astype(int)
##
#test_id = list(x_test.Id.values.ravel())
#save_submission(y_test_hat, 'extremeBayesCVsubmission1.csv', test_id)

#%%
clf = xgb.XGBClassifier(max_depth=11, objective='binary:logistic', 
                        learning_rate=0.01, colsample_bytree=0.4,
                        min_child_weight=1, n_estimators=90, subsample=0.6,
                        reg_alpha=2, reg_lambda=1)
y_train_pred_list = []
y_test_pred_list = []
n_rep = 10
n_cv = 2
np.random.seed(0)
for i in range(n_rep):
    y_train_pred, y_test_pred, y_train, test_id = \
        extremeBayesCV(clf, n_cv, np.random.randint(10000))
    y_train_pred_list.append(y_train_pred)
    y_test_pred_list.append(y_test_pred)
    
y_train_pred = np.mean(y_train_pred_list, axis=0)
y_test_pred = np.mean(y_test_pred_list, axis=0)

best_proba, best_mcc, _ = eval_mcc(y_train, y_train_pred, True)
y_test_hat = (y_test_pred>=best_proba).astype(int)

best_proba, best_mcc, _ = eval_mcc(y_train, y_train_pred, True)
y_test_hat = (y_test_pred>=best_proba).astype(int)
save_submission(y_test_hat, 'extremeBayesCVsubmission1.csv', test_id)
save_data((y_train_pred, y_test_pred), 'extremeBayesCV_train_test1.pkl')