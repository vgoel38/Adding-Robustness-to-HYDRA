import random
tablename = 'movie_companies'
table_nickname = 'mc'
table_columns = ['id','movie_id','company_id','company_type_id']
operator_list = ['<=','>=']
table_columns_ranges = [1,3691809,2,3398517,1,292129,1,2]
output_file = 'queryListImdb_' + tablename + '.txt'
file_handler = open(output_file,'w')
query_init = 'select count(*) from ' + tablename + ' ' + table_nickname + ' ' + 'where '
query_no = 1
for no_of_col in range(1,len(table_columns)+1):
	print 'generating query no', query_no
	for i in range(len(table_columns)-no_of_col+1):
		#print i
		for j in range(100000):
			query_final = ''
			col_no = i
			iter_no = 0
			while(col_no < i + no_of_col):
				iter_no = iter_no + 1
				rand_val = random.randint(table_columns_ranges[2*col_no],table_columns_ranges[2*col_no+1])
				rand_op = random.randint(0,1)
				if iter_no != 1:
					query_final = query_final + ' and '
				query_final = query_final + table_nickname + '.' + table_columns[col_no] + ' ' + operator_list[rand_op] + ' ' + str(rand_val);
				col_no = col_no + 1
			query = query_init + query_final + ' ;' 
			file_handler.write(query + '\n')
			query_no = query_no + 1
file_handler.close()
