table = 'aka_title'
input_file = 'training_set_'+ table + '.txt'
attr_list = ['id', 'movie_id', 'kind_id', 'production_year']
num_attr = 4
attr_min = [0, 0, 0, 0]
attr_max = [1000000000, 1000000000, 1000000000, 1000000000]

input = open(input_file, 'r')
lines_list = input.readlines()

query_matrix = []
cardinality_matrix = []

count = 0
for line in lines_list[0:4]:
	query = [ val for val in line.split() ]
	is_attr_set = [0]*2*num_attr
	type_val = 'attr'
	attr_index = 0
	query_matrix.append([0]*2*num_attr)
	for val in query[1:len(query)-1]:
		if type_val == 'attr':
			attr_index = attr_list.index(val.split('.')[1])
			type_val = 'op'
		elif type_val == 'op':
			if val == '<=':
				attr_index = attr_index*2
			else:
				attr_index = attr_index*2 + 1
			type_val = 'val'
		else:
			query_matrix[count][attr_index] = val
			is_attr_set[attr_index] = 1
			type_val = 'attr'

	i = 0
	for val in query_matrix[count]:
		if is_attr_set[i] == 0:
			if i%2 == 0:
				query_matrix[count][i] = attr_max[i/2]
			else:
				query_matrix[count][i] = attr_min[(i-1)/2]
		i += 1

	cardinality_matrix.append(query[len(query)-1])

	count += 1

output_1 = open(table+'_input.txt','w')
output_2 = open(table+'_output.txt','w')

for val1 in query_matrix:
	for val2 in val1:
		print >> output_1, val2

for val in cardinality_matrix:
	print >> output_2, val

	
