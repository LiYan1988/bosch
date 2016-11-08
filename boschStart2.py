# -*- coding: utf-8 -*-
"""
Created on Wed Nov 02 22:38:50 2016

@author: lyaa
"""

import numpy as np
import pandas as pd
import gc
import matplotlib.pyplot as plt
import cPickle as pickle
import os
from scipy import sparse
from sys import getsizeof
import copy
import seaborn as sns

from sklearn.metrics import matthews_corrcoef
import xgboost as xgb
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn import (preprocessing, manifold, decomposition, ensemble,
                     feature_extraction, model_selection, cross_validation,
                     calibration, linear_model, metrics, neighbors, 
                     naive_bayes)
#from numba import jit

def twoplot(df, col, y_lims=None, xaxis=None):
    ''' scatter plot a feature split into response values as two subgraphs '''
    if col not in df.columns.values:
        print('ERROR: %s not a column' % col)
    ndf = pd.DataFrame(index = df.index)
    ndf[col] = df[col]
    ndf[xaxis] = df[xaxis] if xaxis else df.index
    ndf['Response'] = df['Response']
    
    g = sns.FacetGrid(ndf, col="Response", hue="Response")
    g.map(plt.scatter, xaxis, col, alpha=.7, s=1)
    g.add_legend();
    axes = g.axes
    if y_lims is not None:
        axes[0, 0].set_ylim(y_lims[0],y_lims[1])
#        axes[0, 1].set_xlim(axes_lims['x'][0],axes_lims['x'][0])
    
    del ndf

#@jit
def mcc(tp, tn, fp, fn):
    sup = tp * tn - fp * fn
    inf = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if inf==0:
        return 0
    else:
        return sup / np.sqrt(inf)
        
def mcc_eval(y_prob, dtrain):
    y_true = dtrain.get_label()
    best_mcc = -eval_mcc(y_true, y_prob)
    return 'MCC', best_mcc

#@jit
def eval_mcc(y_true, y_prob, show=False):
    idx = np.argsort(y_prob)
    y_true_sort = y_true[idx]
    n = y_true.shape[0]
    nump = 1.0 * np.sum(y_true) # number of positive
    numn = n - nump # number of negative
    tp = nump
    tn = 0.0
    fp = numn
    fn = 0.0
    best_mcc = 0.0
    best_id = -1
    prev_proba = -1
    best_proba = -1
    mccs = np.zeros(n)
    for i in range(n):
        # all items with idx < i are predicted negative while others are predicted positive
        # only evaluate mcc when probability changes
        proba = y_prob[idx[i]]
        if proba != prev_proba:
            prev_proba = proba
            new_mcc = mcc(tp, tn, fp, fn)
            if new_mcc >= best_mcc:
                best_mcc = new_mcc
                best_id = i
                best_proba = proba
        mccs[i] = new_mcc
        if y_true_sort[i] == 1:
            tp -= 1.0
            fn += 1.0
        else:
            fp -= 1.0
            tn += 1.0
    if show:
        y_pred = (y_prob >= best_proba).astype(int)
        score = matthews_corrcoef(y_true, y_pred)
        print(score, best_mcc)
        plt.plot(mccs)
        return best_proba, best_mcc, y_pred
    else:
        return best_mcc
        
def save_submission(y_pred, file_name, test_id):
    df = pd.DataFrame(data=y_pred, columns=['Response'])
    if test_id is None:
        test_id = pd.read_csv('input/test_numeric.csv', usecols=['Id'])
        df.index = list(test_id.values.ravel())
    else:
        df.index = test_id
    df.index.name = 'Id'
    df.to_csv(file_name, index=True)
    
def create_date_features():
    # sample columns
    train_date_part = pd.read_csv('input/train_date.csv', nrows=10000)
    date_cols = train_date_part.drop('Id', axis=1).count().reset_index().\
        sort_values(by=0, ascending=False)
    date_cols['station'] = date_cols['index'].apply(lambda s: s.split('_')[1])
    date_cols = date_cols.drop_duplicates('station', keep='first')['index'].\
        tolist()
    del train_date_part
    # train start date
    train_start_date = pd.read_csv('input/train_date.csv', 
        usecols=['Id']+date_cols)
    train_start_date['start_date'] = train_start_date[date_cols].min(axis=1)
#    train_start_date['end_date]
    train_start_date.drop(date_cols, axis=1, inplace=True)
    n_train = train_start_date.shape[0]
    # test start date
    test_start_date = pd.read_csv('input/test_date.csv', 
        usecols=['Id']+date_cols)
    test_start_date['start_date'] = test_start_date[date_cols].min(axis=1)
    test_start_date.drop(date_cols, axis=1, inplace=True)
    n_test = test_start_date.shape[0]
    # concate train and test
    start_date = pd.concat([train_start_date, test_start_date])
    del train_start_date, test_start_date
    gc.collect()
    
    train_id = pd.read_csv('input/train_numeric.csv', usecols=['Id'])
    test_id = pd.read_csv('input/test_numeric.csv', usecols=['Id'])
    train_id = train_id.merge(start_date, on='Id')
    test_id = test_id.merge(start_date, on='Id')
    train_test_id = pd.concat((train_id, test_id)).reset_index(drop=True).\
        reset_index(drop=True)
        
    train_test_id['stDiff1'] = train_test_id['start_date'].diff().\
        fillna(9999999).astype(float)
    train_test_id['stDiff2'] = train_test_id['start_date'].iloc[::-1].diff().\
        fillna(9999999).astype(float)
    train_test_id['idDiff1'] = train_test_id['Id'].diff().fillna(9999999).astype(int)
    train_test_id['idDiff2'] = train_test_id['Id'].iloc[::-1].diff().\
        fillna(9999999).astype(int)
    train_test_id = train_test_id.sort_values(by=['start_date', 'Id'], 
        ascending=True)
    train_test_id['idDiff3'] = train_test_id['Id'].diff().fillna(9999999).astype(int)
    train_test_id['idDiff4'] = train_test_id['Id'].iloc[::-1].diff().\
        fillna(9999999).astype(int)
    train_test_id['idDiff5'] = 1+2*(train_test_id['idDiff3']>1)+\
        1*(train_test_id['idDiff4']<-1)
    train_test_id['stDiff3'] = train_test_id['start_date'].diff().\
        fillna(9999999).astype(float)
    train_test_id['stDiff4'] = train_test_id['start_date'].iloc[::-1].diff().\
        fillna(9999999).astype(float)
    train_test_id.drop(['start_date'], axis=1, inplace=True)
    
    train_set = train_id.merge(train_test_id, on='Id')
    test_set = test_id.merge(train_test_id, on='Id')
    
    train_set.to_csv('train_date_feats.csv')
    test_set.to_csv('test_date_feats.csv')
    
def save_data(data, file_name):
    """File name must ends with .pkl
    """
    with open(file_name, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        
def read_data(file_name):
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
        
    return data

def prepare_data(file_name, cols_good, file_save_name):
    df = []
    
    # detect data format
    file_name_ = 'input/'+file_name+'.csv'
    file_save_name_ = file_save_name+'.pkl'
    if not os.path.exists(file_name_):
        file_name_ = 'input/'+file_name+'.csv.zip'
        
    # detect columns
    file_part = pd.read_csv(file_name_, nrows=1)
    cols = file_part.columns.tolist()
    cols.remove('Id')
    data_types = {}
    data_types['Id'] = np.uint32
    if file_name == 'train_numeric':
        cols.remove('Response')
        data_types['Response'] = np.uint8
    if file_name.split('_')[1] == 'categorical':
        data_types.update({c: np.str for c in cols})
    else:
        data_types.update({c: np.float16 for c in cols})

    if cols_good is None:
        numeric_chunks = pd.read_csv(file_name_, chunksize=10000, 
            dtype=data_types)
    else:
        numeric_chunks = pd.read_csv(file_name_, chunksize=10000, 
            dtype=data_types, usecols=cols_good)
        
    for i, dchunk in enumerate(numeric_chunks):
        df.append(dchunk.to_sparse())
        print(i, df[-1].memory_usage().sum(), dchunk.memory_usage().sum())
        print(df[-1].Id[-1])
        del dchunk
        gc.collect()
    
    df = pd.concat(df)
    print(df.memory_usage().sum())

    save_data(df, file_save_name_)
    del df
    gc.collect()
    
def prepare_selected_features():
    x_train = read_data('train_numeric_good.pkl')
    x_test = read_data('test_numeric_good.pkl')
    
    x_train_date = pd.read_csv('train_date_feats.csv')
    x_train_date.drop(['Unnamed: 0'], axis=1, inplace=True)
    x_test_date = pd.read_csv('test_date_feats.csv')
    x_test_date.drop(['Unnamed: 0'], axis=1, inplace=True)
    
    x_train.merge(x_train_date, how='left', on='Id')
    x_test.merge(x_test_date, how='left', on='Id')
    y_train = pd.read_csv('input/train_numeric.csv', usecols=['Response'], 
        dtype=np.uint8)
    
    save_data(x_train, 'x_train_partcols.pkl')
    save_data(y_train, 'y_train.pkl')
    save_data(x_test, 'x_test_partcols.pkl')
    prior = np.sum(y_train)/(1.0*len(y_train))
    prior = prior.iloc[0]
    
    # indexes of positive samples
    posidx = y_train.Response[y_train.Response==1].index.tolist()
    # inexes of negative samples
    negidx = y_train.Response[y_train.Response==0].index.tolist()
    
    save_data(posidx, 'posidx.pkl')
    save_data(negidx, 'negidx.pkl')
    
def xgb_predict_proba(clfxgb, x_test):
    
    n_test = len(x_test)
    y_pred = []
    n_start = 0
    chunksize = 100000
    n_chunks = int(np.ceil(1.*n_test/chunksize))
    for i in range(n_chunks):
#        print('Predicting {} to {}'.format(n_start, min(n_test, n_start+chunksize)))
        a = x_test.iloc[np.arange(n_start, min(n_test, n_start+chunksize))]
        y_pred.append(clfxgb.predict_proba(a)[:,1])
        n_start += chunksize
    y_pred = np.concatenate(y_pred)
    
    return y_pred

class MCSClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, estimator, n_runs, p_ratio, random_state=0):
#        self.x_train = x_train
#        self.y_train = y_train
#        self.x_test = x_test
        self.n_runs = n_runs
        self.estimator = estimator
        self.p_ratio = p_ratio
        self.random_state = random_state
        self.y_pred_samples = []
        np.random.seed(self.random_state)
        
    def one_run_(self, x_train, y_train, x_test):
        """Fit and predict"""
        estimator_ = copy.deepcopy(self.estimator)
        
        posidx = y_train.Response[y_train.Response==1].index.tolist()
        negidx = y_train.Response[y_train.Response==0].index.tolist()
        
        # the negative samples used in training
        sample_idx = np.random.choice(negidx, 
            len(posidx)*int(1./self.p_ratio-1), False)
        sample_idx = np.sort(np.concatenate((sample_idx, posidx)))
        
        x_train_sample = x_train.iloc[sample_idx]
        y_train_sample = y_train.iloc[sample_idx]
        if hasattr(estimator_, 'base_score'):
            prior = np.sum(y_train_sample)/(1.0*len(y_train_sample))
            prior = prior.iloc[0]
            estimator_.base_score = prior            
        y_train_sample = y_train_sample.values.ravel()
#        print(estimator_)
        estimator_.fit(x_train_sample, y_train_sample)
        y_pred = self.one_run_predict_(estimator_, x_test)
        
        return y_pred        
        
    def one_run_predict_(self, estimator_, x_test):
        n_test = len(x_test)
        y_pred = []
        n_start = 0
        chunksize = 100000
        n_chunks = int(np.ceil(1.*n_test/chunksize))
        for i in range(n_chunks):
#            print('Predicting {} to {}'.format(n_start, min(n_test, n_start+chunksize)))
            a = x_test.iloc[np.arange(n_start, min(n_test, n_start+chunksize))]
            y_pred.append(estimator_.predict_proba(a)[:,1])
            n_start += chunksize
        y_pred = np.concatenate(y_pred)
    
        return y_pred
        
    def multi_run(self, x_train, y_train, x_test):
        y_test_pred = None
        for i in range(self.n_runs):
            print('run {}'.format(i))
            if y_test_pred is None:
                y_test_pred = self.one_run_(x_train, y_train, x_test)
            else:
                y_test_pred += self.one_run_(x_train, y_train, x_test)
        y_test_pred /= self.n_runs*1.0
        
        return y_test_pred
    
def create_date_meta_features(out_indexes, id_window_sizes=[3, 5, 7, 9],
                        time_window_sizes=[0.01, 0.05, 0.1, 0.5]):
    """Create leave out features, leave some train samples out to create these
    features. Out of Bag features are used to create meta features, in bag 
    features are used to predict.
    """
    train_test = pd.read_csv('train_test_date.csv')
    train_id = pd.read_csv('train_date_pure.csv', usecols=['Id']).Id.values
    # the in bag indexes 
    in_index = np.setdiff1d(train_id, out_indexes)
    Response_true = \
        train_test.loc[train_test.Id.isin(in_index), 'Response'].copy()
    train_test.loc[train_test.Id.isin(in_index), 'Response'] = 0
    train_test.sort_values(by='Id', ascending=True, inplace=True)
    for w in id_window_sizes:
        train_test['id_window_'+str(w)] = \
                   train_test.Response.rolling(window=10, min_periods=1, 
                                               center=True).sum()
        
    for j, w in enumerate(time_window_sizes):
        n_bins = int(1718.48/w)
        train_test['tmp'] = pd.cut(train_test['min_date'], bins=n_bins, 
            labels=np.arange(n_bins))
        binned = train_test.groupby('tmp')['Response'].sum().\
            fillna(0).to_frame()
        train_test = pd.merge(train_test, binned, how='left', left_on='tmp',
                             right_index=True, 
                             suffixes=('', '_t_window_'+str(j)))
        train_test.drop('tmp', axis=1, inplace=True)
        cols = list(train_test.columns)
        cols[-1] = 't_window_'+str(j)
        train_test.columns = cols
        del binned
        gc.collect()
        
#    train_test.loc[train_test.Id.isin(in_index), 'Response'] = Response_true
    x_train_work_date = train_test.loc[train_test.Id.isin(in_index)].copy()
    x_train_work_date.drop('Response', axis=1, inplace=True)
                                       
    test_id = pd.read_csv('input/test_numeric.csv', usecols=['Id']).Id
    x_test_date = train_test.loc[train_test.Id.isin(test_id)].copy()
    x_test_date.drop('Response', axis=1, inplace=True)
        
    return x_train_work_date, x_test_date, train_test
    
#class MetaBagClassifier(BaseEstimator, ClassifierMixin):
#    def __init__(self, meta_estimators, estimators, n_runs, bag_size, 
#                 random_state=0):
#        self.meta_estimators = meta_estimators
#        self.estimators = estimators
#        self.n_runs = n_runs
#        self.bag_size = bag_size
#        self.random_state = random_state
#        self.train_test = pd.read_csv('train_test_date.csv')
#        self.train_id = pd.read_csv('train_date_pure.csv', usecols=['Id']).Id.values
#        self.test_id = pd.read_csv('test_date_pure.csv', usecols=['Id']).Id.values
#        
#    def fit(x_train=None, y_train=None):
        
def create_numeric_meta_features(train_meta_id, estimator_meta, train_numeric, 
                                 x_test_file):
#    train_numeric = read_data('train_numeric_feature_set1.pkl')
    train_id = np.array(train_numeric.Id.values)
    train_work_id = np.setdiff1d(train_id, train_meta_id)
    
    x_train_meta = train_numeric[train_numeric.Id.isin(train_meta_id)].copy()
    y_train_meta = np.array(x_train_meta['Response'].values)
    x_train_meta.drop('Response', axis=1, inplace=True)
    x_train_meta.fillna(-999, inplace=True)
    
    x_train_work = train_numeric[train_numeric.Id.isin(train_work_id)].copy()
    y_train_work = np.array(x_train_work['Response'].values)
    x_train_work.drop('Response', axis=1, inplace=True)
    x_train_work.fillna(-999, inplace=True)
    
    del train_numeric
    gc.collect()
    
    estimator_meta.fit(x_train_meta, y_train_meta)
    
    train_work_new_feature = estimator_meta.predict_proba(x_train_work)[:, 1]
    del x_train_work
    gc.collect()

    x_test = read_data(x_test_file)
    x_test.fillna(-999, inplace=True)
    test_new_feature = estimator_meta.predict_proba(x_test)[:, 1]
    del x_test
    gc.collect()

    return train_work_new_feature, test_new_feature, y_train_work
    
def create_train_test_sets(train_numeric, x_test_file, train_meta_id, 
                           meta_estimators, 
                           id_window_sizes=[3, 5, 7, 9], 
                           time_window_sizes=[0.01, 0.05, 0.1, 0.5]):
    
    # create numeric features
    train_id = np.array(train_numeric.Id.values)
    train_work_id = np.setdiff1d(train_id, train_meta_id)
#    del train_numeric
#    gc.collect()
    
    train_work_new_features = []
    test_new_features = []
    for estimator in meta_estimators:
        train_work_new_feature, test_new_feature, y_train_work = \
            create_numeric_meta_features(train_meta_id, estimator, 
                                         train_numeric, x_test_file)
        train_work_new_features.append(train_work_new_feature)
        test_new_features.append(test_new_feature)
        
    # create date features
    x_train_work_date, x_test_date, _ = \
        create_date_meta_features(train_meta_id, 
                                   id_window_sizes=id_window_sizes, 
                                   time_window_sizes=time_window_sizes)
    
    # merge new features with train 
    x_train_work = train_numeric[train_numeric.Id.isin(train_work_id)].copy()
    y_train_work = np.array(x_train_work['Response'].values)
    x_train_work.drop('Response', axis=1, inplace=True)
    x_train_work.fillna(-999, inplace=True)
    del train_numeric
    gc.collect()
    
    for i, f in enumerate(train_work_new_features):
        x_train_work['numeric_feature_'+str(i)] = f
    x_train_work = pd.merge(x_train_work, x_train_work_date, how='left',
                            left_on='Id', right_on='Id')
    
    x_test = read_data(x_test_file)
    for i, f in enumerate(test_new_features):
        x_test['numeric_feature_'+str(i)] =f
    x_test = pd.merge(x_test, x_test_date, how='left', left_on='Id', 
                      right_on='Id')
    
    return x_train_work, y_train_work, x_test
    
def meta_predict(estimator, meta_estimators, train_numeric, x_test_file,
                 id_window_sizes=[3, 5, 7, 9], 
                 time_window_sizes=[0.01, 0.05, 0.1, 0.5], random_state=0):
#    train_numeric = read_data('train_numeric_feature_set1.pkl')
    train_id = np.array(train_numeric.Id.values)
    np.random.seed(random_state)
    train_meta_id = np.random.choice(train_id, len(train_id)/2, replace=False)
    train_work_id = np.setdiff1d(train_id, train_meta_id)
#    y_train_pred = pd.DataFrame({'Id': train_id, 
#                                 'Response': np.zeros((len(train_id),), 
#                                                      dtype=np.float16)})
    y_train_pred = []
    ids = []
    
    x_train_work1, y_train_work1, x_test1 = \
        create_train_test_sets(train_numeric, x_test_file, train_meta_id, 
                               meta_estimators, 
                               id_window_sizes=id_window_sizes, 
                               time_window_sizes=time_window_sizes)
    estimator.base_score = 1.*np.sum(y_train_work1)/len(y_train_work1)
    estimator.fit(x_train_work1, y_train_work1)
    tmp = estimator.predict_proba(x_train_work1)[:, 1]
    ids.append(train_work_id)
    y_train_pred.append(tmp)
#    y_train_pred[y_train_pred.Id.isin(train_work_id), 'Response'] = tmp
    y_test_pred = estimator.predict_proba(x_test1)[:, 1]
    del x_train_work1, x_test1
    gc.collect()
    
    x_train_work2, y_train_work2, x_test2 = \
        create_train_test_sets(train_numeric, x_test_file, train_work_id, 
                               meta_estimators, 
                               id_window_sizes=id_window_sizes, 
                               time_window_sizes=time_window_sizes)
    estimator.base_score = 1.*np.sum(y_train_work2)/len(y_train_work2)
    estimator.fit(x_train_work2, y_train_work2)
    tmp = estimator.predict_proba(x_train_work2)[:, 1]
    ids.append(train_meta_id)
    y_train_pred.append(tmp)
#    y_train_pred[y_train_pred.Id.isin(train_meta_id), 'Response'] = tmp
#    y_train_pred = np.array(y_train_pred.Response.values)
    y_test_pred += estimator.predict_proba(x_test2)[:, 1]
    y_test_pred /= 2.0
    del x_train_work2, x_test2

    return ids, y_train_pred, y_test_pred
    
if __name__=='__main__':
    print('main')
    