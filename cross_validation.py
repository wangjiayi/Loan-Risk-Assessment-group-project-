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
# from sklearn.cross_validation import train_test_split

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
# test_dataset = 'ecs171test.csv'



#
# Selected features
#
valid_cols = ['f2', 'f271', 'f274', 'f332', 'f336', 'f4', 'f5', 'f527', 'f528', 'f647', 'f776', 'f777']

#
# Split train dataset in x, y
#
train = pd.read_csv(train_dataset)
train_x = train[valid_cols]
train_y = train.iloc[:,-1]
train_y = binarize(train_y) # all to 0/1


#
# Gradient Boosting Classifier
#
clf = ensemble.GradientBoostingClassifier(
	loss='deviance', learning_rate=0.01, n_estimators=3000,
	subsample=0.6, min_samples_split=12, min_samples_leaf=12,
	max_depth=6, random_state=1357, verbose=1)


#
# cross validation
#
scores = model_selection.cross_val_score(clf, train_x, train_y, cv=3) # 5-fold cross-validation
print(scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


