import pandas as pd
import numpy as np

dataset = pd.read_csv('Data.csv')

X = dataset.iloc[:, 2:-1].values
y = dataset.iloc[:, -1].values




# def sigmoid(r, w):
#     y_pred = 0

#     for i in range(len(r)):

#         y_pred += w[i]*r[i]

#     return 1 / (1 + np.exp(-y_pred))    




# def log_reg(X, y, learning_rate, n_iter):

#     weights = np.zeros(X.shape[0])

#     for _ in range(n_iter):
        
#         sq_error = 0
#         for r in X:
#             z = sigmoid(r, weights)
#             error = z - y
#             sq_error += error ** 2

            

#             for i in range(len(r) - 1):
#                 weights[i] -= learning_rate * error * r[i]

 

 
# weights = log_reg(X, y, learning_rate = 0.01, n_iter = 1000)

# print(weights)


def predict(row, coefficients):
	yhat = coefficients[0]
	for i in range(len(row)-1):
		yhat += coefficients[i + 1] * row[i]
	return 1.0 / (1.0 + np.exp(-yhat))
 
# Estimate logistic regression coefficients using stochastic gradient descent
def coefficients_sgd(train, y, l_rate, n_epoch):
	coef = [0.0 for i in range(len(train[0]))]
	for epoch in range(n_epoch):
		sum_error = 0
		for row in train:
			yhat = predict(row, coef)
			error = y - yhat
			sum_error += error**2
			coef[0] = coef[0] + l_rate * error * yhat * (1.0 - yhat)
			for i in range(len(row)-1):
				coef[i + 1] = coef[i + 1] + l_rate * error * yhat * (1.0 - yhat) * row[i]
		print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
	return coef



c = coefficients_sgd(X, y, 0.001, 1000)
print(c)