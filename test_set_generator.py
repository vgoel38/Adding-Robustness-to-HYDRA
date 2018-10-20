import numpy as np
import sys

def test_set_generator(table):

	input_file = 'table_'+table+'/'+table+'_input.txt'
	output_norm_file = 'table_' + table + '/' + table + '_output_normalised.txt'
	output_unnorm_file = 'table_'+table+'/'+table+'_output_unnormalised.txt'

	training_input_file = 'table_'+table+'/'+table+'_training_input'
	training_output_norm_file = 'table_'+table+'/'+table+'_training_output_normalised'
	training_output_unnorm_file = 'table_'+table+'/'+table+'_training_output_unnormalised'

	test_input_file = 'table_'+table+'/'+table+'_test_input.txt'
	test_output_norm_file = 'table_'+table+'/'+table+'_test_output_normalised.txt'
	test_output_unnorm_file = 'table_'+table+'/'+table+'_test_output_unnormalised.txt'

	test_set_fraction = 0.3


	X = np.loadtxt(input_file, delimiter=' ')
	Y_norm = np.loadtxt(output_norm_file)
	Y_norm = np.reshape(Y_norm,(-1,1))
	Y_unnorm = np.loadtxt(output_unnorm_file)
	Y_unnorm = np.reshape(Y_unnorm,(-1,1))

	sys.stdout = open(test_input_file,'w')
	for val in X[0:int(test_set_fraction*(len(X))),:]:
		print(*val)

	sys.stdout = open(test_output_norm_file,'w')
	for val in Y_norm[0:int(test_set_fraction*(len(Y_norm))),:]:
		print(*val)

	sys.stdout = open(test_output_unnorm_file,'w')
	for val in Y_unnorm[0:int(test_set_fraction*(len(Y_unnorm))),:]:
		print(*val)

	X = X[int(test_set_fraction*(len(X))):len(X),:]
	Y_norm = Y_norm[int(test_set_fraction*(len(Y_norm))):len(Y_norm),:]
	Y_unnorm = Y_unnorm[int(test_set_fraction*(len(Y_unnorm))):len(Y_unnorm),:]



	sys.stdout = open(training_input_file+str(1)+'.txt','w')
	for val in X:
		print(*val)

	sys.stdout = open(training_output_norm_file+str(1)+'.txt','w')
	for val in Y_norm:
		print(*val)

	sys.stdout = open(training_output_unnorm_file+str(1)+'.txt','w')
	for val in Y_unnorm:
		print(*val)


	# i=1
	# while i <= 5:
	# 	sys.stdout = open(training_input_file+str(i)+'.txt','w')
	# 	for val in X[0:int(0.2*(len(X))),:]:
	# 		print(*val)

	# 	sys.stdout = open(training_output_norm_file+str(i)+'.txt','w')
	# 	for val in Y_norm[0:int(0.2*(len(Y_norm))),:]:
	# 		print(*val)

	# 	sys.stdout = open(training_output_unnorm_file+str(i)+'.txt','w')
	# 	for val in Y_unnorm[0:int(0.2*(len(Y_unnorm))),:]:
	# 		print(*val)

	# 	X = X[int(0.2*(len(X))):len(X),:]
	# 	Y_norm = Y_norm[int(0.2*(len(Y_norm))):len(Y_norm),:]
	# 	Y_unnorm = Y_unnorm[int(0.2*(len(Y_unnorm))):len(Y_unnorm),:]

	# 	i+=1

	sys.stdout = sys.__stdout__

if __name__ == "__main__":
	test_set_generator('aka_title')