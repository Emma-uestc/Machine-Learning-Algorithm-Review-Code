import numpy as np 
from math import sqrt

def accuracy_score(y,y_predict):
	assert len(y) == len(y_predict)
	return np.sum(y == y_predict) /len(y)

def mean_square_error(y,y_predict):
	assert len(y) == len(y_predict)
	return np.sum((y - y_predict)**2) / len(y)

def root_mean_square_error(y,y_predict):
	assert len(y) == len(y_predict)
	return sqrt(mean_square_error(y,y_predict))

def mean_abs_error(y,y_predict):
	assert len(y) == len(y_predict)
	return np.sum(np.absolute(y - y_predict)) /len(y)

def r2_score(y,y_predict):
	return 1-mean_square_error(y,y_predict) / np.var(y)