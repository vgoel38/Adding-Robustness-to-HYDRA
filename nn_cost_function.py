import numpy as np
import math
from sigmoid_gradient import sigmoid_gradient

def nn_cost_function(nn_params, input_layer_size, hidden_layer_size, num_labels, X, Y, lamda):
#Implements the neural network cost function for a two layer neural network

	#Reshaping nn_params back into the parameters params_1 and params_2
	params_1 = nn_params[0:hidden_layer_size * (input_layer_size + 1)]
	params_1 = (np.reshape(params_1, (input_layer_size + 1, hidden_layer_size))).T
	params_2 = nn_params[hidden_layer_size * (input_layer_size + 1):]
	params_2 = (np.reshape(params_2, (hidden_layer_size + 1, -1))).T

	m = len(X)
	         
	J = 0

	def sigmoid(x):
	#returns the value of the sigmoid function evaluated at z (can be scalar/vector/matrix)
		return 1 / (1 + np.exp(-x))

	sigmoid = np.vectorize(sigmoid)

	#Adding bias unit in the input layer
	X = np.insert(X, 0, 1, axis = 1)
	z2 = X @ params_1.T
	a2 = sigmoid(z2)

	#Adding bias unit in the hidden layer
	a2 = np.insert(a2, 0, 1, axis = 1)
	h = a2 @ params_2.T
	h = sigmoid(h)

	J = np.sum(np.square(h - Y))/(2*m)

	#Calculating Jreg
	params_1_reg = np.square(params_1[:,1::])
	params_2_reg = np.square(params_2[:,1::])
	Jreg = (np.sum(params_1_reg) + np.sum(params_2_reg)) * (lamda/(2*m))
	J = J + Jreg




	params_1_grad = np.zeros(params_1.shape)
	params_2_grad = np.zeros(params_2.shape)

	delta_a1 = np.zeros(params_1.shape)
	delta_a2 = np.zeros(params_2.shape)

	error_h = h - Y
	error_a2 = np.multiply((error_h @ params_2), np.insert(sigmoid_gradient(z2),0,1,axis=1))

	delta_a2 = delta_a2 + error_h.T @ a2
	delta_a1 = delta_a1 + (error_a2[:,1::]).T @ X

	params_1_grad = delta_a1/m
	params_2_grad = delta_a2/m

	params_1_grad[:,1::] += (lamda/m) * params_1[:,1::]
	params_2_grad[:,1::] += (lamda/m) * params_2[:,1::]

	#Unroll gradients
	grad = np.concatenate(((params_1_grad.T).ravel(),(params_2_grad.T).ravel()), axis=None)

	print(J)

	return J, grad
