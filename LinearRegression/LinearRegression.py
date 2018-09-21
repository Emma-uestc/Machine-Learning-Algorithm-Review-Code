import numpy as np 
from metrics import r2_score

class LinearRegression:
	def __init__(self):
		self.coef_ = None
		self.intercept_ = None
		self._theta = None
# analytic solution
	def fit_analytic(self,X_train,y_train):
		assert X_train.shape[0] == y_train.shape[0]
		X_b = np.hstack([np.ones((len(X_train),1)),X_train])
		self._theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T.dot(y_train))
		self.intercept_ = self._theta[0]
		self.coef_ = self._theta[1:]

		return self

	#use Gradient Descent
	def fit_gd(self,X_train,y_train,alpha = 0.01,n_iters = 1e3):
		
		assert X_train.shape[0] == y_train.shape[0]
		# loss function
		def J(theta,X_b,y):
			try:
				return np.sum((y - X_b.dot(theta)) **2) /(2 * len(y))
			except:
				return float('inf')
			

		#partial with respect to theta
		# def dJ(theta,X_b,y):
		# 	d_theta = np.empty(len(theta))
		# 	d_theta[0] = np.sum(X_b.dot(theta) - y)
		# 	for j in range(1,len(theta)):
		# 		d_theta[j] = (X_b.dot(theta) - y).dot(X_b[:,j])
		# 	return d_theta /len(X_b)
		def dJ(theta,X_b,y):
			return X_b.T.dot(X_b.dot(theta) - y) /len(y)
		

		# gradient Descent
		def gradient_descent(X_b,y,init_theta,alpha,n_iters = 1e3,epsilon = 1e8):
			theta = init_theta
			i_iter = 0
			while i_iter < n_iters:
				d_theta = dJ(theta,X_b,y)
				last_theta = theta
				theta = theta - alpha * d_theta
				if(abs(J(theta,X_b,y) - J(last_theta,X_b,y)) < epsilon):
					break
				i_iter += 1
			return theta

		X_b = np.hstack([np.ones((len(X_train),1)),X_train])
		init_theta = np.zeros(X_b.shape[1])
		self._theta = gradient_descent(X_b,y_train,init_theta,alpha,n_iters)

		self.intercept_ = self._theta[0]
		self.coef_ = self._theta[1:]


		return self

	def fit_sgd(self,X_train,y_train,n_iters = 50,t0 = 5,t1 = 50):
		assert X_train.shape[0] == y_train.shape[0]
		assert n_iters >= 1

		def dJ_sgd(theta,X_bi,y_i):
			return X_bi * (X_bi.dot(theta) - y_i)

		def sgd(X_b,y,init_theta,n_iters = 10,t0 = 5,t1 = 50):
			def learning_rate(t):
				return t0 / (t + t1)

			theta = init_theta
			m = len(X_b)
			for i_iter in range(n_iters):
				indexs = np.random.permutation(m)
				X_b_new = X_b[indexs,:]
				y_new = y[indexs]
				for i in range(m):
					d_theta = dJ_sgd(theta,X_b_new[i],y_new[i])
					theta = theta - learning_rate(i_iter * m +i) * d_theta
				return theta


		X_b = np.hstack([np.ones((len(X_train),1)),X_train])
		init_theta = np.random.randn(X_b.shape[1])
		self._theta = sgd(X_b,y_train,init_theta,n_iters,t0,t1)

		self.intercept_ = self._theta[0]
		self.coef_ = self._theta[1:]
		return self
		
	def predict(self,X_test):
		assert self.intercept_ is not None and self.coef_ is not None
		assert X_test.shape[1] == len(self.coef_)
		X_b = np.hstack([np.ones((len(X_test),1)),X_test])
		return X_b.dot(self._theta)

	def score(self,X_test,y_test):
		y_predict = self.predict(X_test)
		return r2_score(y_test,y_predict)

	def __repr__(self):
		return "LinearRegression()"