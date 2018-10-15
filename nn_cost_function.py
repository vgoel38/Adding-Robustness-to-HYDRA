import numpy as np
# from sigmoid import sigmoid
import math

# sigmoid = np.vectorize(sigmoid)

def nn_cost_function(nn_params, input_layer_size, hidden_layer_size, num_labels, X, Y, lamda):
#Implements the neural network cost function for a two layer neural network

	#Reshaping nn_params back into the parameters params_1 and params_2
	params_1 = nn_params[0:hidden_layer_size * (input_layer_size + 1)]
	params_1 = (np.reshape(params_1, (input_layer_size + 1, hidden_layer_size))).T
	params_2 = nn_params[hidden_layer_size * (input_layer_size + 1):]
	params_2 = (np.reshape(params_2, (-1, hidden_layer_size + 1))).T

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

	# Y = zeros(size(y,1),num_labels); %converting the output values into vectors of size "num_labels"
	# for i=1:size(y,1),
	# 	Y(i,y(i,1))=1;
	# end;

	# J = sum((Y .* log(h) + (1-Y) .* log(1-h))(:)) * (-1/m); %calculating J

	# %calculating Jreg
	# Theta1_reg = (Theta1 .^ 2)(:,[2:size(Theta1,2)]); 
	# Theta2_reg = (Theta2 .^ 2)(:,[2:size(Theta2,2)]);
	# Jreg = ( sum(Theta1_reg(:)) + sum(Theta2_reg(:)) ) * ( lambda/(2*m) );
	# J = J + Jreg;