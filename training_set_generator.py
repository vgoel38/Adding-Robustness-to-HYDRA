#The following code takes a training set (containing queries on a specific table) as an input
#generates training data (input and output files) compatible with the neural network

import sys
import numpy as np

def training_set_generator(table, attr_list, attr_min, attr_max, table_card):

	input_file = 'table_' + table + '/' + 'training_set_' + table + '.txt'
	output_query_matrix_file = 'table_' + table + '/' + table + '_input.txt'
	output_cardinality_matrix_file_normalised = 'table_' + table + '/' + table + '_output_normalised.txt'
	output_cardinality_matrix_file_unnormalised = 'table_' + table + '/' + table + '_output_unnormalised.txt'

	num_attr = len(attr_list)

	input = open(input_file, 'r')
	lines_list = input.readlines()

	#input training set of neural network
	query_matrix = []

	#output training set of neural network
	cardinality_matrix = []

	count = 0
	for line in lines_list:
		
		query = [ val for val in line.split() ]

		#records whether there exists a filter w.r.t. an operator (<= or >=) of an attribute
		is_attr_set = [0]*2*num_attr
		
		#records whether the current val is 'attr'/'op'/'value'
		type_val = 'attr'
		
		attr_index = 0
		
		query_matrix.append([0]*2*num_attr)
		
		#setting filter values of each query in the query matrix
		#skipping first val (=table name) and last val(=output cardinality)
		for val in query[1:len(query)-1]:
			if type_val == 'attr':
				attr_index = attr_list.index(val.split('.')[1])
				type_val = 'op'
			elif type_val == 'op':
				if val == '<=':
					node_index = attr_index*2
				else:
					node_index = attr_index*2 + 1
				type_val = 'val'
			else:
				query_matrix[count][node_index] = val
				is_attr_set[node_index] = 1
				type_val = 'attr'

		#setting min and max values of the absent filter predicates
		i = 0
		for val in query_matrix[count]:
			if is_attr_set[i] == 0:
				if i%2 == 0:
					query_matrix[count][i] = attr_max[int(i/2)]
				else:
					query_matrix[count][i] = attr_min[int((i-1)/2)]
			i += 1

		#Standardizing the input
		i=0
		for val in query_matrix[count]:
			if i%2 == 0:
				query_matrix[count][i] = (int(query_matrix[count][i]) - attr_min[int(i/2)])/(attr_max[int(i/2)] - attr_min[int(i/2)])
			else:
				query_matrix[count][i] = (attr_min[int((i-1)/2)] - int(query_matrix[count][i]))/(attr_max[int((i-1)/2)] - attr_min[int((i-1)/2)])
			i += 1


		#setting output cardinality of the present query 
		cardinality_matrix.append(query[len(query)-1])

		count += 1

	sys.stdout = open(output_query_matrix_file,'w')
	for val in query_matrix:
			print(*val)

	sys.stdout = open(output_cardinality_matrix_file_unnormalised,'w')
	for val in cardinality_matrix:
			print(val)

	sys.stdout = open(output_cardinality_matrix_file_normalised,'w')
	for val in cardinality_matrix:
		val = int(val)
		if val == 0:
			print(val)
		else:
			print(np.log(val)/np.log(table_card))

	sys.stdout = sys.__stdout__

if __name__ == "__main__":
	training_set_generator('aka_title')
