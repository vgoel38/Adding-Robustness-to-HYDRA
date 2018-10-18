import numpy as np
import sys

def predict(table,X,Y_norm,Y_unnorm, params_1_file, params_2_file, pred_file, table_card):

	params_1 = np.loadtxt(params_1_file, delimiter=' ')
	params_2 = np.loadtxt(params_2_file)

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

	cost_norm = np.sum(np.square(h - Y_norm))/(2*m)

	h = h*np.log(table_card)
	h = np.exp(h)

	cost_unnorm = np.sum(np.absolute(h - Y_unnorm))/m

	sys.stdout = open(pred_file,'w')
	h = np.reshape(h,(-1,1))
	Y_unnorm = np.reshape(Y_unnorm,(-1,1))
	h = np.append(Y_unnorm,h,axis=1)
	for val in h:
		print(*val)

	sys.stdout = sys.__stdout__

	return cost_norm, cost_unnorm

if __name__ == "__main__":
	# table = 'movie_companies'
	# table_card = 3691809

	table = 'aka_title'
	table_card = 425692

	test_input_file = 'table_'+table+'/'+table+'_test_input.txt'
	test_output_norm_file = 'table_'+table+'/'+table+'_test_output_normalised.txt'
	test_output_unnorm_file = 'table_'+table+'/'+table+'_test_output_unnormalised.txt'
	ult_opt_params_1_file = 'table_'+table+'/'+table+'_ult_opt_params_1.txt'
	ult_opt_params_2_file = 'table_'+table+'/'+table+'_ult_opt_params_2.txt'
	pred_file = 'table_'+table+'/'+table+'_predictions.txt'

	X = np.loadtxt(test_input_file)
	Y_norm = np.loadtxt(test_output_norm_file)
	Y_unnorm = np.loadtxt(test_output_unnorm_file)

	print(predict(table,X,Y_norm,Y_unnorm, ult_opt_params_1_file, ult_opt_params_2_file, pred_file, table_card))