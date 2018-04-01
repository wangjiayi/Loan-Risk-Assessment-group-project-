#!/usr/bin/env python
#
# Load univariate, bivariate test result and combine them
# Pick K largest features from the combined result
#
import numpy as np
import pandas as pd
import scipy as sp
import pickle

def get_feature_list(result):
	lst = []
	for f, corr in result:
		lst.append(f)
	return lst

K = 100 # pick K features that has largest correlation coefficient


#
# uni-variate
#
uni_result = pickle.load(open( "pearson_uni.pickle", "r" ))

# should already be sorted, but just in case
uni_result = sorted(uni_result, key=lambda tup: tup[1], reverse=True)

uni_set = uni_result[:K]



#
# bi-variate
#
bi_add_result = pickle.load(open( "pearson_bi_add.pickle", "r" ))
bi_sub_result = pickle.load(open( "pearson_bi_sub.pickle", "r" ))
bi_mul_result = pickle.load(open( "pearson_bi_mul.pickle", "r" ))
bi_div_result = pickle.load(open( "pearson_bi_div.pickle", "r" ))

# should already be sorted, but just in case
bi_add_result = sorted(bi_add_result, key=lambda tup: tup[1], reverse=True)
bi_sub_result = sorted(bi_sub_result, key=lambda tup: tup[1], reverse=True)
bi_mul_result = sorted(bi_mul_result, key=lambda tup: tup[1], reverse=True)
bi_div_result = sorted(bi_div_result, key=lambda tup: tup[1], reverse=True)

bi_set = bi_add_result[:K] + bi_sub_result[:K] + bi_mul_result[:K] + bi_div_result[:K]



#
# Combine and pick K largest
#
bi_set = sorted(bi_set, key=lambda tup: tup[1], reverse=True)
# print(bi_set[:K])

all_set = bi_set + uni_set
all_set = sorted(all_set, key=lambda tup: tup[1], reverse=True)
# print(all_set[:K])

feature_list = get_feature_list(all_set[:K])
print(feature_list)

# save to file
pickle.dump(feature_list, open( "pearson_features.pickle", "wb" ))
