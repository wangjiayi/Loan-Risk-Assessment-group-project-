#!/usr/bin/env python
#
# Univariate Pearson Correlation test
# 
# Conduct Pearson test on each feature w/e train_y
# Save test result(a list of feature score, from high to low) 
# to pearson_uni.pickle
#
import numpy as np
import pandas as pd
import scipy as sp
import pickle
from sklearn.preprocessing import StandardScaler, Imputer

def _binarize(a):
	if a > 0.:
		return 1
	else:
		return 0

binarize = np.vectorize(_binarize)

# train data
train_dataset = 'ecs171train.csv'

#
# Split train dataset in x, y
#
train = pd.read_csv(train_dataset)
# train_x = train[valid_cols]
train_x = train.iloc[:,1:-1]
train_y = train.iloc[:,-1]
train_y = binarize(train_y) # all to 0/1

# extract features
features = train_x.columns.values.tolist()

# normalize data [optional]
imp=Imputer(strategy='median')
sd=StandardScaler()
x = imp.fit_transform(train_x)
x = sd.fit_transform(x)
x = pd.DataFrame(x, columns=features)

#
# univariate pearson test
#
uni_result = []
for f in features:
	corr = sp.stats.pearsonr(x[f], train_y)[0] # (corr, p-val)
	if not np.isnan(corr):
		uni_result.append((f, corr)) # (feature, corr)

uni_result = sorted(uni_result, key=lambda tup: tup[1], reverse=True) # descending
print("10 highest scored features:")
print(uni_result[:10])

#
# Save result to file
#
pickle.dump(uni_result, open( "pearson_uni.pickle", "wb" ))
# uni_result = pickle.load(open( "pearson_uni.pickle", "r" ))


print('Done!')
