# -*- coding: utf-8 -*-
"""
Created on Sun Nov 06 18:34:39 2016

@author: lyaa
"""

from boschStart2 import *

train_id = pd.read_csv('train_date_pure.csv', usecols=['Id']).Id.values
test_id = pd.read_csv('test_date_pure.csv', usecols=['Id']).Id.values
# the samples in training set 2
np.random.seed(0)
train_id_part = np.random.choice(train_id, len(train_id)/2, replace=False)
df = create_bag_features(train_id_part, id_window_sizes=[3, 5, 7, 9],
                        time_window_sizes=[0.01, 0.05, 0.1, 0.5])
train_id_diff = np.setdiff1d(train_id, train_id_part)
x_train = df.loc[df.Id.isin(train_id_diff)].copy()
y_train = x_train.Response.copy().values.astype(np.uint8)
x_train.drop('Response', axis=1, inplace=True)
x_test = df.loc[df.Id.isin(test_id)].copy()
x_test.drop('Response', axis=1, inplace=True)
prior = 1.*np.sum(y_train)/len(y_train)

del df
gc.collect()

clf = xgb.XGBClassifier(max_depth=7, objective='binary:logistic', 
                        learning_rate=0.021, colsample_bytree=0.82,
                        min_child_weight=3, base_score=prior, 
                        n_estimators=690)
clf.fit(x_train, y_train, eval_set=[(x_train, y_train)], eval_metric='auc',
        verbose=True, early_stopping_rounds=10)
y_train_pred = clf.predict_proba(x_train)[:, 1]
best_proba, best_mcc, _ = eval_mcc(y_train, y_train_pred, True)
y_test_pred = clf.predict_proba(x_test)[:, 1]
y_test_pred = (y_test_pred >= best_proba).astype(np.uint8)

save_submission(y_test_pred, 'leakyFeatureSubmission.csv', test_id)