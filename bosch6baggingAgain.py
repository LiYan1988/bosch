# -*- coding: utf-8 -*-
"""
Created on Tue Nov 08 22:10:07 2016

@author: lyaa
"""


from boschStart2 import *

#%% load data
x_train = read_data('train_numeric_feature200_set1.pkl')
x_train.fillna(-999, inplace=True)
y_train = np.array(x_train.Response.values)
response = x_train[['Id', 'Response']].copy()
x_train.drop(['Response'], axis=1, inplace=True)
x_test = read_data('test_numeric_feature200_set1.pkl')
x_test.fillna(-999, inplace=True)

#x_train = x_train.iloc[np.arange(100000)]
#y_train = y_train[np.arange(100000)]
#x_test = x_test.iloc[np.arange(100000, 200000)]

#%% split data into meta and bag sets
np.random.seed(0)
train_id = np.array(x_train.Id.values)

y_test_pred = np.zeros((x_test.shape[0],))
y_train_pred = y_train.copy()
y_train_pred = y_train_pred.astype(np.float16)
y_train_pred.fill(0.)

n_cv = 5
kf = model_selection.StratifiedKFold(n_splits=n_cv, shuffle=True, random_state=2)

meta_estimator_rf = ensemble.RandomForestClassifier(n_estimators=10, max_depth=7, n_jobs=-1)
meta_estimator_ext = ensemble.ExtraTreesClassifier(n_estimators=10, max_depth=7, n_jobs=-1)
meta_estimator_gb = naive_bayes.GaussianNB(priors=[.9942, .0058])
#meta_estimator_knn = neighbors.KNeighborsClassifier(n_neighbors=16)
meta_estimators = {'rf1':meta_estimator_rf, 'ext1':meta_estimator_ext, 
                   'gb1':meta_estimator_gb}
#meta_estimators = {}

clf = xgb.XGBClassifier(max_depth=15, objective='binary:logistic', 
                        learning_rate=0.01, colsample_bytree=0.4,
                        min_child_weight=1, n_estimators=40, subsample=0.8,
                        reg_alpha=2, reg_lambda=1)

for bag_id, meta_id in kf.split(x_train, y_train):
    x_train_meta = x_train.iloc[meta_id].copy()
    y_train_meta = response.iloc[meta_id].copy()
    y_train_meta = np.array(y_train_meta.Response.values)
    
    x_train_bag = x_train.iloc[bag_id].copy()
    y_train_bag = response.iloc[bag_id].copy()
    y_train_bag = np.array(y_train_bag.Response.values)
    
    train_new_features = []
    test_new_features = []
    names = []
    for name, estimator in meta_estimators.iteritems():
        print('Estimator {}'.format(name))
        names.append(name)
        print('Fitting...')
        estimator.fit(x_train_meta, y_train_meta)
        print('Predicting on bag set')
        u = estimator.predict_proba(x_train_bag)[:, 1]
        train_new_features.append(u)
        print('Predicting on test set')
        u = estimator.predict_proba(x_test)[:, 1]
        test_new_features.append(u)
        del estimator
        gc.collect()
        
    for i, name in enumerate(names):
        x_train_bag[name] = train_new_features[i]
        x_test[name] = test_new_features[i]
        
    del x_train_meta, y_train_meta, names, train_new_features, test_new_features
    gc.collect()
        
#    u = meta_estimator_rf.predict_proba(x_train_bag)[:, 1]
#    x_train_bag['rf1'] = u
#    u = meta_estimator_rf.predict_proba(x_test)[:, 1]
#    x_test['rf1'] = u
    
    print('Training outer classifier...')
    prior = 1.*y_train_bag.sum()/len(y_train_bag)
    clf.base_score = prior
    clf.fit(x_train_bag, y_train_bag)
    print('Predicting on bag set')
    y_train_pred[bag_id] += clf.predict_proba(x_train_bag)[:, 1]

    del x_train_bag
    gc.collect()
    
    print('Predicting on test set')
    y_test_pred += clf.predict_proba(x_test)[:, 1]
    x_test.drop(meta_estimators.keys(), axis=1, inplace=True)
    

y_train_pred /= 1.*(n_cv-1)
y_test_pred /= 1.*n_cv

best_proba, best_mcc, _ = eval_mcc(y_train, y_train_pred, True)
y_test_hat = (y_test_pred>=best_proba).astype(int)

test_id = list(x_test.Id.values.ravel())
#del x_train, x_test
#gc.collect()
save_submission(y_test_hat, 'ensembleCVsubmission2.csv', test_id)
