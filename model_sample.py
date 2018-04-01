#!/usr/bin/env python
#
import numpy as np
import pandas as pd
from sklearn import linear_model, neighbors, svm
from sklearn import decomposition
from sklearn import discriminant_analysis
# import xgboost
# from sklearn.cross_validation import train_test_split

train_dataset = 'ecs171train.csv'
test_dataset = 'ecs171test.csv'
# train_dataset = 'train1000.csv'
# test_dataset = 'test1000.csv'

train = pd.read_csv(train_dataset)

#
# Split dataset in 2
#
train_y = train.iloc[:,-1]       # loss
train_x = train.iloc[:,1:-1]     # dataset w/o loss or id

test = pd.read_csv(test_dataset)
test_id = test.iloc[:,:1]
test_x = test.iloc[:,1:]         # dataset w/o loss or id

# valid_cols = ['f521','f522','f269','f767','f259','f270','f219','f250']
# train_x2 = train_x[valid_cols]
# test_x2 = test[valid_cols]

# # training w/e Linear Regression
# lr = linear_model.LinearRegression()
# lr.fit(train_x, train_y)
# lr.predict(test_x)

# # training w/e Logistic Regression
# logreg = linear_model.LogisticRegression(C=1.0)
# logreg.fit(train_x, train_y)
# logreg.predict(test_x)

# training w/e K-Nearest Neighbour
knn = neighbors.KNeighborsClassifier()
knn.fit(train_x, train_y)
knn.predict(test_x)

# brr = linear_model.BayesianRidge(compute_score=True)
# brr.fit(train_x, train_y)
# brr.predict(test_x)

# svr_rbf = svm.SVR(kernel='rbf', C=1e3, gamma=0.1)
# svr_lin = svm.SVR(kernel='linear', C=1e3)
# svr_poly = svm.SVR(kernel='poly', C=1e3, degree=2)
# svr = svr_lin
# svr.fit(train_x, train_y)
# svr.predict(test_x)

# xr = xgboost.XGBRegressor()
# xr = xgboost.XGBClassifier()
# xr.fit(train_x, train_y)
# xr.predict(test_x)

# # training w/e Linear SVM
# clf = svm.LinearSVC()
# clf.fit(train_x, train_y)
# clf.predict(test_x)

# lda = discriminant_analysis.LinearDiscriminantAnalysis()
# lda.fit(train_x, train_y)
# lda.predict(test_x)

losses = knn.predict(test_x)
losses = losses.astype(int)
losses = pd.DataFrame(losses, columns=['loss'])
output_df = pd.concat([test_id, losses], axis=1)


#
# Save output
#
output_df.to_csv("out.csv", index=False)

# lr.coef_
# lr.score(test_x, test_y)

