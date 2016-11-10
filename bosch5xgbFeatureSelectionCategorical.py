# -*- coding: utf-8 -*-
"""
Created on Mon Nov 07 23:18:04 2016

@author: lyaa
"""

from boschStart2 import *

#%% extract usefull columns
feat_cat_stat = read_data('feat_cat_stat.pkl')
cat_mean = [feat_cat_stat[i].Sum.sum() for i in range(len(feat_cat_stat))]
cat_nz_stat = [feat_cat_stat[i] for i in 
                    list(np.nonzero(cat_mean)[0].astype(int))]
#cat_nz_sum = np.array([i.Sum.sum() for i in cat_nz_stat])
cat_nz = [i.columns[0] for i in cat_nz_stat]
#cat_nz.append('Id')

ele_str = list(set.union(*[set(i.iloc[:, 0].values.tolist()) 
    for i in cat_nz_stat]))
ele_num = [int(i[1:]) for i in ele_str]

a = pd.read_csv('input/train_categorical.csv', usecols=cat_nz, nrows=100)
a.replace(ele_str, ele_num, inplace=True)

#%% xgboost calculate feature importance
clf = xgb.XGBClassifier(max_depth=9, objective='binary:logistic', 
                        learning_rate=0.01, colsample_bytree=0.5,
                        min_child_weight=1, n_estimators=15, subsample=0.6,
                        reg_alpha=2, reg_lambda=1)

CHUNKSIZE = 20000
feature_importance = []
for x, y in zip(pd.read_csv('input/train_categorical.csv', chunksize=CHUNKSIZE, 
                            usecols=cat_nz, dtype=np.str), 
                pd.read_csv('train_id_response.csv', chunksize=CHUNKSIZE, 
                            usecols=['Response'], dtype=np.uint8)):
    x.fillna(-9999999999, inplace=True)
    x.replace(ele_str, ele_num, inplace=True)
    y = np.array(y.values)
    y = y.ravel()
    prior = 1.*y.sum()/len(y)
    clf.base_score = prior
    clf.fit(x, y, eval_set=[(x, y)], eval_metric='auc', verbose=True)
    feature_importance.append(clf.feature_importances_)
    
#%%