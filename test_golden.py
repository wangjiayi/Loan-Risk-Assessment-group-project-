#!/usr/bin/env python
#
import numpy as np
import pandas as pd
from sklearn import linear_model, neighbors, svm
from sklearn import decomposition
from sklearn import discriminant_analysis
from sklearn import ensemble
# import xgboost
# import pickle
from sklearn import model_selection
from sklearn.cross_validation import KFold
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler, Imputer

def _binarize(a):
	if a > 0.:
		return 1
	else:
		return 0

binarize = np.vectorize(_binarize)

def _threshhold(a, b):
	if a > b:
		return 1
	else:
		return 0

threshold = np.vectorize(_threshhold)

train_dataset = 'ecs171train.csv'
test_dataset = 'ecs171test.csv'


#
# Split train dataset in x, y
#
train = pd.read_csv(train_dataset)
# train_x = train[valid_cols]
train_x = train.iloc[:,1:-1]
train_y = train.iloc[:,-1]
train_y = binarize(train_y) # all to 0/1

#
# Select features (3 golden features)
#
pd.options.mode.chained_assignment = None
train_x['f528-f527'] = train_x['f528'] - train_x['f527']
train_x['f528-f274'] = train_x['f528'] - train_x['f274']
train_x['f527-f274'] = train_x['f527'] - train_x['f274']



#
# Read test data
#
test = pd.read_csv(test_dataset)
test_x = test.iloc[:,1:]
test_id = test.iloc[:,:1]
pd.options.mode.chained_assignment = None
test_x['f528-f527'] = test_x['f528'] - test_x['f527']
test_x['f528-f274'] = test_x['f528'] - test_x['f274']
test_x['f527-f274'] = test_x['f527'] - test_x['f274']

#
# Gradient Boosting Classifier
#
clf = ensemble.RandomForestClassifier(verbose=1)

clf = ensemble.GradientBoostingClassifier(
	loss='deviance', learning_rate=0.01, n_estimators=100,
	subsample=0.6, min_samples_split=12, min_samples_leaf=12,
	max_depth=6, random_state=1357, verbose=1)
clf = ensemble.GradientBoostingClassifier(verbose=1)

#
# Split train.csv into train and test
#
x_train, x_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.2, random_state=42)

imp=Imputer(strategy='median')
sd=StandardScaler()
x1 = imp.fit_transform(x_train)
x2 = sd.fit_transform(x1) # scale data to normal distribution
t1 = imp.fit_transform(x_test)
t2 = sd.fit_transform(t1) # scale data to normal distribution

clf.fit(x2, y_train)
clf.score(t2, y_test)
res = clf.predict(t2)



#
# Train on train.csv, test on test.csv
#
imp=Imputer(strategy='median')
sd=StandardScaler()
x1 = imp.fit_transform(train_x)
x2 = sd.fit_transform(x1) # scale data to normal distribution
t1 = imp.fit_transform(test_x)
t2 = sd.fit_transform(t1) # scale data to normal distribution

clf.fit(x2, train_y)
test_y = clf.predict(t2)

#
# Wrap in data frame, and save
#
test_id = test_id.astype(int)
test_loss = pd.DataFrame(test_y, columns=['loss'])

output_df = pd.concat([test_id, test_loss], axis=1)
output_df.to_csv("out.csv", index=False)