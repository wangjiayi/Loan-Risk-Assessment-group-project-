#!/usr/bin/env python
#
# Gradient Boosting classifier w/e
# 100 Bivariate features that scored highest in pearson test 
#
# 0.98340000000000005 accuracy on splitted training data
# Accuracy can go higher if filter out lowest scored univariate features
#
import numpy as np
import pandas as pd
from sklearn import linear_model, neighbors, svm
from sklearn import decomposition
from sklearn import discriminant_analysis
from sklearn import ensemble
# import xgboost
import pickle
import operator
import re
from sklearn import model_selection
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler, Imputer, RobustScaler, QuantileTransformer
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

binarize = np.vectorize(lambda y: 1 if y>0 else 0)

# merge classifier and regressor result
# if regressor doesn't have result but classifer has result, return classifer result
def _merge(clf_result, reg_result):
	clf_result = float(clf_result)
	if clf_result>0 and reg_result>0:
		return clf_result*reg_result
	return clf_result

merge=np.vectorize(_merge)


train_dataset = 'ecs171train.csv'
# test_dataset = 'ecs171test.csv'



#
# Selected features
#
feature_list = pickle.load(open( "pearson_features.pickle", "r" ))



#
# Split train dataset in x, y
#
train = pd.read_csv(train_dataset)
train_x = train.iloc[:,1:-1]
train_y = train.iloc[:,-1]

# test = pd.read_csv(test_dataset)
# test_x = test.iloc[:,1:]
# test_id = test.iloc[:,:1]



#
# Construct new dataset with custom features
#
pd.options.mode.chained_assignment = None # disable warning
opt_func = {"+": operator.add, "-": operator.sub, "*": operator.mul, "/": operator.div}
for feature in feature_list:
	match = re.match(r'(\w+)([+|\-|*|\\])(\w+)', feature)
	if match:
		grps = match.groups()
		if len(grps) == 3:
			a = grps[0]
			opt = grps[1]
			b = grps[2]
			# add new features
			train_x[feature] = opt_func[opt](train_x[a], train_x[b]) # a+b, a-b


#
# Split & train, then validate
#
x_train, x_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.2, random_state=42)

y_test_raw=y_test
y_train_raw=y_train
y_test = binarize(y_test) # all to 0/1
y_train = binarize(y_train) # all to 0/1

imp=Imputer(strategy='median')
sd=StandardScaler() # 0.983
sd=QuantileTransformer(n_quantiles=10, random_state=0, output_distribution='normal') # 0.984
sd=RobustScaler() # 0.987 ~ 0.989 best
x1 = imp.fit_transform(x_train)
x2 = sd.fit_transform(x1) # scale data to normal distribution
t1 = imp.fit_transform(x_test)
t2 = sd.fit_transform(t1) # scale data to normal distribution

#
# Gradient Boosting Classifier
#
clf = ensemble.RandomForestClassifier(verbose=1, n_estimators=100, warm_start=True, oob_score=True, max_features='sqrt')
clf = ensemble.GradientBoostingClassifier(verbose=1)
clf = ensemble.GradientBoostingClassifier(verbose=1, n_estimators=100, max_depth=9)
clf = pickle.load(open( "gbc_clf_iter400.pickle", "r" ))

# clf.fit(x2, y_train)
clf.score(t2, y_test)
res = clf.predict(t2)
mean_absolute_error(y_test_raw, res)


#
# Regression
#

# reg = RandomForestRegressor(criterion='mae', verbose=1, n_jobs=4) # too slow
reg = RandomForestRegressor(verbose=1, n_estimators=100, n_jobs=4)
# reg = GradientBoostingRegressor(verbose=1, n_estimators=400, n_jobs=4)
reg = pickle.load(open( "rfr_reg_iter400.pickle", "r" ))

# reg.fit(x2, y_train_raw)
res_raw = reg.predict(t2)
reg.score(t2, y_test_raw)
mean_absolute_error(y_test_raw, res_raw)

res_merged = merge(res, res_raw)
mean_absolute_error(y_test_raw, res_merged)

print('Done!')
