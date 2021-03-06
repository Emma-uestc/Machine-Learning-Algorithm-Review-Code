import numpy as np 
class SimpleLinearRegression1:
	def __init__(self):
		self.a_ = None
		self.b_ = None

	def fit(self,x_train,y_train):
		assert x_train.ndim ==1,\
		"Simple linear Regression can only solve single feature training data."
		assert len(x_train) == len(y_train),\
		"the size of x_train must be equal to the size of y_train"

		x_mean = np.mean(x_train)
		y_mean = np.mean(y_train)

		num = 0.0
		d = 0.0
		for x,y in zip(x_train,y_train):
			num += (x-x_mean) * (y- y_mean);
			d += (x- x_mean)**2
		self.a_ = num/d;
		self.b_ = y_mean - self.a_ * x_mean
		return self

	def predict(self,x_predict):
		assert x_predict.ndim ==1
		self.a_ is not None and self.b_ is not None
		return np.array([self._predict(x) for x in x_predict])

	def _predict(self,x):
		return self.a_ * x + self.b_

		def __repr__(self):
			return "Simple Linear Regression()"


class SimpleLinearRegression2:
	def __init__(self):
		self._a = None
		self._b = None

	def fit(self,x_train,y_train):
		assert x_train.ndim ==1,\
		"This linear regression can only solve the simple linear regression"
		assert len(x_train) == len(y_train)

		x_mean = np.mean(x_train)
		y_mean = np.mean(y_train)

		self._a = (x_train - x_mean).dot(y_train - y_mean) /(x_train - x_mean).dot(x_train - x_mean)
		self._b = y_mean - self._a * x_mean
		return self

	def predict(self,X_test):
		assert X_test.ndim ==1
		assert self._a is not None and self._b is not None

		return np.array([self._predict(x) for x in X_test])

	def _predict(self,x):
		return self._a * x + self._b

	def __repr__(self):
		return "SimpleLinearRegression2()"

