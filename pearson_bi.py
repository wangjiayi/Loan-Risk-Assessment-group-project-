#!/usr/bin/env python
#
# Bivariate Pearson Correlation test
# 
# Conduct Pearson test on each pair of features w/e train_y
# Save test result(a list of feature score, from high to low) 
# to pearson_bi_add/sub/mul/div.pickle
#
# Arthrimetics to be tested:
# addition, substraction, multipication, division
#
import numpy as np
import pandas as pd
import scipy as sp
import pickle
import itertools
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
# bivariate pearson test - substraction
#

# get permutation of feature pair
perms=list(itertools.permutations(features, 2))
print("total {} pairs".format(len(perms)))

# substractions
sub_result = []
for i, p in enumerate(perms):
	# alive...
	if i % 1000 == 0:
		print("{}-th pair".format(i))
	# construct feature pair
	col_x = x[p[0]] - x[p[1]]
	corr = sp.stats.pearsonr(col_x, train_y)[0] # (corr, p-val)
	if not np.isnan(corr):
		f = "{}-{}".format(p[0], p[1]) # f11-f22
		sub_result.append((f, corr)) # (feature, corr)

sub_result = sorted(sub_result, key=lambda tup: tup[1], reverse=True) # descending
print("10 highest scored substractions:")
print(sub_result[:10])

# save to file
pickle.dump(sub_result, open( "pearson_bi.pickle", "wb" ))
# sub_result = pickle.load(open( "pearson_bi.pickle", "r" ))


#
# bivariate pearson test - addition
#

# get combination of feature pair
combs=list(itertools.combinations(features, 2))
print("total {} pairs".format(len(combs)))

# additions
add_result = []
for i, p in enumerate(combs):
	# alive...
	if i % 1000 == 0:
		print("{}-th pair".format(i))
	# construct feature pair
	col_x = x[p[0]] + x[p[1]]
	corr = sp.stats.pearsonr(col_x, train_y)[0] # (corr, p-val)
	if not np.isnan(corr):
		f = "{}+{}".format(p[0], p[1]) # f11-f22
		add_result.append((f, corr)) # (feature, corr)

add_result = sorted(add_result, key=lambda tup: tup[1], reverse=True) # descending
print("10 highest scored additions:")
print(add_result[:10])

# save to file
pickle.dump(add_result, open( "pearson_bi_add.pickle", "wb" ))
# add_result = pickle.load(open( "pearson_bi_add.pickle", "r" ))


#
# bivariate pearson test - multipication
#

# get combination of feature pair
combs=list(itertools.combinations(features, 2))
print("total {} pairs".format(len(combs)))

# multipication
mul_result = []
for i, p in enumerate(combs):
	# alive...
	if i % 1000 == 0:
		print("{}-th pair".format(i))
	# construct feature pair
	col_x = x[p[0]] * x[p[1]]
	corr = sp.stats.pearsonr(col_x, train_y)[0] # (corr, p-val)
	if not np.isnan(corr):
		f = "{}*{}".format(p[0], p[1]) # f11-f22
		mul_result.append((f, corr)) # (feature, corr)

mul_result = sorted(mul_result, key=lambda tup: tup[1], reverse=True) # descending
print("10 highest scored multipications:")
print(mul_result[:10])

# save to file
pickle.dump(mul_result, open( "pearson_bi_mul.pickle", "wb" ))
# mul_result = pickle.load(open( "pearson_bi_mul.pickle", "r" ))



#
# bivariate pearson test - division
#

# get permutation of feature pair
perms=list(itertools.permutations(features, 2))
print("total {} pairs".format(len(perms)))

# multipication
div_result = []
for i, p in enumerate(perms):
	# alive...
	if i % 1000 == 0:
		print("{}-th pair".format(i))
	# construct feature pair
	col_x = x[p[0]] / x[p[1]]
	corr = sp.stats.pearsonr(col_x, train_y)[0] # (corr, p-val)
	if not np.isnan(corr):
		f = "{}/{}".format(p[0], p[1]) # f11-f22
		div_result.append((f, corr)) # (feature, corr)

div_result = sorted(div_result, key=lambda tup: tup[1], reverse=True) # descending
print("10 highest scored divisions:")
print(div_result[:10])

# save to file
pickle.dump(div_result, open( "pearson_bi_div.pickle", "wb" ))
# div_result = pickle.load(open( "pearson_bi_div.pickle", "r" ))


print('Done!')
