import gc
import sys
import mysql.connector
import timeit
from mysql.connector import Error

#NOTE:DON'T FORGET TO START MYSQL SERVER BEFORE RUNNING

#written training set is of type
#[table name,attrib,op,val,attrib,op,val,.......,actual cardinality]

def connect():
    """ Connect to MySQL database """
    try:
        conn = mysql.connector.connect(host='localhost',
                                       database='imdb',
                                       user='root',
                                       password='khuranakapil')
        if conn.is_connected():
            print('Connected to MySQL database')
 
    except Error as e:
        print(e)
        return False
    return conn


if __name__ == '__main__':
    conn = connect()
if conn == False:
	print "Start MYSQL server first\n"
	exit()
db_cursor = conn.cursor()

inp_file_name = sys.argv[1]
total_queries = 0
query = list()
output_training_set = list()
out_file_handler = open('training_set.txt','w')
start = timeit.default_timer()
with open(inp_file_name,'r') as inp_file_handler:
	for line in inp_file_handler:
		total_queries+=1
		sys.stdout.flush()
		print 'Processing query no',total_queries
		query = line.split(' ')
		temp = list()
		if len(query)<4:
			total_queries-=1
			continue
		#run query to get result
		db_cursor.execute(line)
		db_result = db_cursor.fetchall()
		temp.append(query[3])
		index = 6
		while (index + 2)<len(query):
			temp.append(query[index])
			temp.append(query[index+1])
			temp.append(query[index+2])
			index+=4
		temp.append(str(long(db_result[0][0])))
		output_training_set.append(temp)
		gc.collect()

for i in range(len(output_training_set)):
	for j in range(len(output_training_set[i])):
		out_file_handler.write(str(output_training_set[i][j]) + ' ')
	out_file_handler.write('\n')
	#print output_training_set[i]
conn.close()
out_file_handler.close()
stop = timeit.default_timer()
print 'Run Time: ' , (stop-start), 'seconds'
print 'training set with total',total_queries,' queries has been saved in file training_set.txt'