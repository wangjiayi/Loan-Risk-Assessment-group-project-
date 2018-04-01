#!/usr/bin/env python
#
# Binarize training data (loss=0,1)
# Use PCA to reduce training data dimension to 2
# Make scatter plot
# Blue=0, Red=1
#
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import binarize

binarize = np.vectorize(lambda y: 1 if y>0 else 0)

train_dataset = 'ecs171train.csv'
train = pd.read_csv(train_dataset)
train_x = train.iloc[:,1:-1]
train_y = train.iloc[:,-1]
train_y = binarize(train_y) # all to 0/1

pca=PCA(n_components=2)
x1 = imp.fit_transform(x_train)
x2 = pca.fit_transform(x1) # 2d dataset

# scatter plot
for l, c, m in zip((0,1), ('blue','red'), ('o','s')):
	plt.scatter(x2[train_y == l, 0], x2[train_y == l, 1], color=c, label='class %s' % l, alpha=0.2, marker=m)

plt.show()

