#!/usr/bin/env python
#
# Train & predict
# Submission code
#
import numpy as np
import pandas as pd
import scipy as sp
from sklearn import ensemble
import itertools
import pickle
import operator
import re
from sklearn.preprocessing import StandardScaler, Imputer, RobustScaler, QuantileTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

#############################
#
# Config
#
#############################
train_dataset = 'ecs171train.csv'
test_dataset = 'ecs171test.csv'


#############################
#
# Utility functions
#
#############################
binarize = np.vectorize(lambda y: 1 if y>0 else 0)

# merge classifier and regressor result
# if regressor doesn't have result but classifer has result, return classifer result
def _merge(clf_result, reg_result):
	clf_result = float(clf_result)
	if clf_result>0 and reg_result>0:
		return clf_result*reg_result
	return clf_result

merge=np.vectorize(_merge)


#############################
#
# Read datasets
#
#############################
print("Reading train dataset ...")
train = pd.read_csv(train_dataset)
train_x = train.iloc[:,1:-1]
train_y_raw = train.iloc[:,-1]
train_y = binarize(train_y_raw)

print("Reading test dataset ...")
test = pd.read_csv(test_dataset)
test_x = test.iloc[:,1:]
test_id = test.iloc[:,:1]


#############################
#
# Feature Selection
#
#############################

#
# Retyrn top @top_n pairs of feature from pearson test
#
def feature_rank_uni(train_x, train_y, top_n=100):

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
	print("total {} pairs".format(len(features)))
	for f in features:
		corr = sp.stats.pearsonr(x[f], train_y)[0] # (corr, p-val)
		if not np.isnan(corr):
			uni_result.append((f, corr)) # (feature, corr)

	uni_result = sorted(uni_result, key=lambda tup: tup[1], reverse=True) # descending

	return uni_result[:top_n]

#
# Retyrn top @top_n pairs of feature from pearson test
#
def feature_rank_bi(train_x, train_y, top_n=100):

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
	perms=list(itertools.permutations(features, 2))
	print("substraction - total {} pairs".format(len(perms)))
	sub_result = []
	for i, p in enumerate(perms):
		# alive...
		if i % 10000 == 0:
			print("{}-th pair".format(i))
		# construct feature pair
		col_x = x[p[0]] - x[p[1]]
		corr = sp.stats.pearsonr(col_x, train_y)[0] # (corr, p-val)
		if not np.isnan(corr):
			f = "{}-{}".format(p[0], p[1]) # f11-f22
			sub_result.append((f, corr)) # (feature, corr)

	sub_result = sorted(sub_result, key=lambda tup: tup[1], reverse=True) # descending
	# pickle.dump(sub_result, open("pearson_bi_sub.pickle", "wb"))
	# sub_result = pickle.load(open("pearson_bi_sub.pickle", "r"))

	#
	# bivariate pearson test - addition
	#
	combs=list(itertools.combinations(features, 2))
	print("addition - total {} pairs".format(len(combs)))
	add_result = []
	for i, p in enumerate(combs):
		# alive...
		if i % 10000 == 0:
			print("{}-th pair".format(i))
		# construct feature pair
		col_x = x[p[0]] + x[p[1]]
		corr = sp.stats.pearsonr(col_x, train_y)[0] # (corr, p-val)
		if not np.isnan(corr):
			f = "{}+{}".format(p[0], p[1]) # f11-f22
			add_result.append((f, corr)) # (feature, corr)

	add_result = sorted(add_result, key=lambda tup: tup[1], reverse=True) # descending
	# pickle.dump(add_result, open("pearson_bi_add.pickle", "wb"))
	# add_result = pickle.load(open("pearson_bi_add.pickle", "r"))

	#
	# bivariate pearson test - multipication
	#
	combs=list(itertools.combinations(features, 2))
	print("multiplication - total {} pairs".format(len(combs)))
	mul_result = []
	for i, p in enumerate(combs):
		# alive...
		if i % 10000 == 0:
			print("{}-th pair".format(i))
		# construct feature pair
		col_x = x[p[0]] * x[p[1]]
		corr = sp.stats.pearsonr(col_x, train_y)[0] # (corr, p-val)
		if not np.isnan(corr):
			f = "{}*{}".format(p[0], p[1]) # f11-f22
			mul_result.append((f, corr)) # (feature, corr)

	mul_result = sorted(mul_result, key=lambda tup: tup[1], reverse=True) # descending
	# pickle.dump(mul_result, open("pearson_bi_mul.pickle", "wb"))
	# mul_result = pickle.load(open("pearson_bi_mul.pickle", "r"))

	#
	# bivariate pearson test - division
	#
	perms=list(itertools.permutations(features, 2))
	print("division - total {} pairs".format(len(perms)))
	div_result = []
	for i, p in enumerate(perms):
		# alive...
		if i % 10000 == 0:
			print("{}-th pair".format(i))
		# construct feature pair
		col_x = x[p[0]] / x[p[1]]
		corr = sp.stats.pearsonr(col_x, train_y)[0] # (corr, p-val)
		if not np.isnan(corr):
			f = "{}/{}".format(p[0], p[1]) # f11-f22
			div_result.append((f, corr)) # (feature, corr)

	div_result = sorted(div_result, key=lambda tup: tup[1], reverse=True) # descending
	# pickle.dump(div_result, open("pearson_bi_div.pickle", "wb"))
	# div_result = pickle.load(open("pearson_bi_div.pickle", "r"))

	# output
	final_result = add_result[:top_n] + sub_result[:top_n] + mul_result[:top_n] + div_result[:top_n]
	final_result = sorted(final_result, key=lambda tup: tup[1], reverse=True) # descending
	return final_result[:top_n]

#
# Get top N univariate/bivariate features
#
def top_features(uni_result, bi_result, top_n=100):

	#
	# univariate
	#
	uni_result = sorted(uni_result, key=lambda tup: tup[1], reverse=True)
	uni_set = uni_result[:top_n]

	#
	# bivariate
	#
	bi_result = sorted(bi_result, key=lambda tup: tup[1], reverse=True)
	bi_set = bi_result[:top_n]

	#
	# Combine and pick top_n largest
	#
	bi_set = sorted(bi_set, key=lambda tup: tup[1], reverse=True)

	all_set = bi_set + uni_set
	# all_set = bi_set
	all_set = sorted(all_set, key=lambda tup: tup[1], reverse=True)

	def get_feature_list(result):
		lst = []
		for f, corr in result:
			lst.append(f)
		return lst

	feature_list = get_feature_list(all_set[:top_n])
	return feature_list

#
# Append selected features to dataset
#
def add_features(X, features):
	pd.options.mode.chained_assignment = None # disable warning
	opt_func = {"+": operator.add, "-": operator.sub, "*": operator.mul, "/": operator.div}
	for f in features:
		match = re.match(r'(\w+)([+|\-|*|\\])(\w+)', f)
		if match:
			grps = match.groups()
			if len(grps) == 3:
				a = grps[0]
				opt = grps[1]
				b = grps[2]
				# add new features
				X[f] = opt_func[opt](X[a], X[b]) # a+b, a-b
	return X


print("Pearson test on univariate features ...")
uni_result = feature_rank_uni(train_x, train_y, 100)

print("Pearson test on bivariate features ...")
bi_result = feature_rank_bi(train_x, train_y, 100)

print("Getting top 100 features ...")
feature_list = top_features(uni_result, bi_result, 100)
# feature_list = pickle.load(open( "pearson_features.pickle", "r" ))

print("Append selected features to dataset ...")
train_x = add_features(train_x, feature_list)
test_x = add_features(test_x, feature_list)




#############################
#
# Data Preprocessing
#
#############################
print("Preprocessing dataset...")
imp=Imputer(strategy='median')
sd=RobustScaler() # 0.987 ~ 0.989 best
x1 = imp.fit_transform(train_x)
x2 = sd.fit_transform(x1) # scale data to normal distribution
t1 = imp.fit_transform(test_x)
t2 = sd.fit_transform(t1) # scale data to normal distribution



#############################
#
# Classification
#
#############################
print("Training classifier ...")
clf = ensemble.GradientBoostingClassifier(verbose=1, n_estimators=1000, max_depth=9)
# pickle.dump(clf, open("clf.pickle", "wb"))
# clf = pickle.load(open("gbc_clf_iter1000_alldata.pickle", "r"))
clf.fit(x2, train_y)

print("Predicting classification ...")
clf_result = clf.predict(t2)



#############################
#
# Regression
#
#############################

# select all default Xs
x3=x2[train_y_raw!=0]

# select all default Ys, apply log transformation then scale
y = train_y_raw[train_y_raw!=0]
y = y[:] # make a copy
y = np.log(y) # log transform
y = y.reshape(-1,1)
scaler = RobustScaler()
y3 = scaler.fit_transform(y)
y3 = y3.reshape(1,-1)[0]

print("Training regressor ...")
reg = ensemble.RandomForestRegressor(verbose=1, n_estimators=1000, n_jobs=3)
# pickle.dump(clf, open("reg.pickle", "wb"))
# reg = pickle.load(open("rfr_reg_iter1000_alldata.pickle", "r"))
reg.fit(x3, y3)

print("Predicting regression ...")
reg_result = reg.predict(t2)
reg_result = reg_result.reshape(-1,1)
# scaler = pickle.load(open("scaler.pickle", "r"))
reg_result = scaler.inverse_transform(reg_result)
reg_result = np.exp(reg_result)
reg_result = reg_result.reshape(1,-1)[0]




#############################
#
# Submission Output
#
#############################
print("Output submission ...")
res_merged = merge(clf_result, reg_result)
res_merged = np.round(res_merged, 2)

test_id = test_id.astype(int)
test_loss = pd.DataFrame(res_merged, columns=['loss'])

output_df = pd.concat([test_id, test_loss], axis=1)
output_df.to_csv("out.csv", index=False)

print('Done!')
