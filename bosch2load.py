# -*- coding: utf-8 -*-
"""
Created on Thu Nov 03 09:41:11 2016

@author: lyaa
"""

from boschStart2 import *

#train_part = pd.read_csv('input/train_numeric.csv', nrows=10000)
#cols = list(train_part.columns)[1:]

cols_good = pd.read_csv('others/imp_matrix_H2O_GBM_allFeatures.csv')
cols_good.drop(cols_good.columns[1:], axis=1, inplace=True)
cols_good['type'] = cols_good['variable'].apply(lambda x: x.split('_')[2][0])
cols_good_F = list(cols_good.variable[cols_good.type=='F'])
cols_good_F.remove('L3_S32_F3854')
cols_good_D = list(cols_good.variable[cols_good.type=='D'])

n_train = 1183747
n_test = 1183748
#samples = np.random.choice(n_train, 50000, False)

#train= pd.read_csv('input/train_numeric.csv', usecols=['Id']).\
#    to_sparse(fill_value=None)
#for i, c in enumerate(cols):
#    print('Loading column {}: {}'.format(i, c))
#    train_c = pd.read_csv('input/train_numeric.csv', usecols=[c]).\
#        to_sparse(fill_value=None)
#    train[c] = train_c

#train_numeric = []
#numeric_chunks = pd.read_csv('input/train_numeric.csv', 
#    chunksize=10000, dtype=np.float16, usecols=cols_good_F)
#for i, dchunk in enumerate(numeric_chunks):
#    train_numeric.append(dchunk.to_sparse())
#    print(i, train_numeric[-1].memory_usage().sum(), 
#          dchunk.memory_usage().sum())
#    del dchunk
#    gc.collect()
#
#train_numeric = pd.concat(train_numeric)
#print(train_numeric.memory_usage().sum())
#
#save_data(train_numeric, 'train_numeric.pkl')

#prepare_data('train_numeric')
#prepare_data('train_date')
#prepare_data('train_categorical')
#prepare_data('test_numeric')
#prepare_data('test_date')
#prepare_data('test_categorical')