import random
import numpy as np

from main import main
from training_set_generator import training_set_generator
from test_set_generator import test_set_generator
from predict import predict 

table = 'movie_companies'
table_card = 3691809
attr_list = ['id', 'movie_id', 'company_id', 'company_type_id']
attr_min = [1, 2, 1, 1]
attr_max = [3691809, 3398517, 292129, 2]

# table = 'aka_title'
# table_card = 425692
# attr_list = ['id', 'kind_id', 'movie_id', 'production_year']
# attr_min = [1, 1, 0, 1875]
# attr_max = [448814, 7, 3398411, 2022]

#========== Phase 1 : Shuffling the complete training set =====================
with open('table_'+table+'/training_set_'+table+'.txt','r') as source:
    data = [ (random.random(), line) for line in source ]
data.sort()
with open('table_'+table+'/training_set_'+table+'.txt','w') as target:
    for _, line in data:
        target.write( line )


#========== Phase 2 : Generating nn compatible training set =====================
training_set_generator(table, attr_list, attr_min, attr_max, table_card)



#========== Phase 3 : Dividing training set into training and test sets =====================
test_set_generator(table)



#========== Phase 4 : Model selection =====================
min_hidden_layer_size = 10
max_hidden_layer_size = 10
min_lamda = 0
max_lamda = 0
opt_hidden_layer_size = 0
opt_lamda = 0
inf = 100000000

training_input_file = 'table_'+table+'/'+table+'_training_input'
training_output_norm_file = 'table_'+table+'/'+table+'_training_output_normalised'
training_output_unnorm_file = 'table_'+table+'/'+table+'_training_output_unnormalised'
params_1_file = 'table_'+table+'/'+table+'_params_1.txt'
params_2_file = 'table_'+table+'/'+table+'_params_2.txt'
opt_params_1_file = 'table_'+table+'/'+table+'_opt_params_1.txt'
opt_params_2_file = 'table_'+table+'/'+table+'_opt_params_2.txt'
ult_opt_params_1_file = 'table_'+table+'/'+table+'_ult_opt_params_1.txt'
ult_opt_params_2_file = 'table_'+table+'/'+table+'_ult_opt_params_2.txt'
pred_file = 'table_'+table+'/'+table+'_predictions.txt'

min_cost_norm = inf
hidden_layer_size = min_hidden_layer_size


X_train = np.loadtxt(training_input_file+str(1)+'.txt', delimiter=' ')
Y_train_norm = np.loadtxt(training_output_norm_file+str(1)+'.txt')
Y_train_norm = np.reshape(Y_train_norm,(-1,1))
main(table, X_train, Y_train_norm, 11, 0, params_1_file, params_2_file)
open(ult_opt_params_1_file, "w").writelines([l for l in open(params_1_file).readlines()])
open(ult_opt_params_2_file, "w").writelines([l for l in open(params_2_file).readlines()])


# while hidden_layer_size <= max_hidden_layer_size:
# 	lamda = min_lamda
# 	while lamda <= max_lamda:
# 		cost_norm = 0
# 		cost_unnorm = 0

# 		i=1
# 		while i<=5:
# 			min_tmp_cost_norm = inf
# 			X_vld = np.loadtxt(training_input_file+str(i)+'.txt', delimiter=' ')
# 			Y_vld_norm = np.loadtxt(training_output_norm_file+str(i)+'.txt')
# 			Y_vld_norm = np.reshape(Y_vld_norm,(-1,1))
# 			Y_vld_unnorm = np.loadtxt(training_output_norm_file+str(i)+'.txt')
# 			Y_vld_unnorm = np.reshape(Y_vld_unnorm,(-1,1))

# 			X_train = []
# 			Y_train_norm = []
			
# 			j=1
# 			while j<=5:
# 				if j!=i:
# 					X_tmp = np.loadtxt(training_input_file+str(j)+'.txt', delimiter=' ')
# 					Y_tmp = np.loadtxt(training_output_norm_file+str(j)+'.txt')
# 					Y_tmp = np.reshape(Y_tmp,(-1,1))
# 					if len(X_train) == 0:
# 						X_train = X_tmp
# 					else:
# 						X_train = np.append(X_train,X_tmp,axis=0)
# 					if len(Y_train_norm) == 0:
# 						Y_train_norm = Y_tmp
# 					else:
# 						Y_train_norm = np.append(Y_train_norm,Y_tmp,axis=0)
# 				j+=1

# 			main(table, X_train, Y_train_norm, hidden_layer_size, lamda, params_1_file, params_2_file)
# 			tmp_cost_norm,tmp_cost_unnorm = predict(table,X_vld, Y_vld_norm, Y_vld_unnorm, params_1_file, params_2_file, pred_file, table_card)
# 			cost_norm += tmp_cost_norm
# 			cost_unnorm += tmp_cost_unnorm
			
# 			if tmp_cost_norm < min_tmp_cost_norm:
# 				min_tmp_cost_norm = tmp_cost_norm
# 				open(opt_params_1_file, "w").writelines([l for l in open(params_1_file).readlines()])
# 				open(opt_params_2_file, "w").writelines([l for l in open(params_2_file).readlines()])

# 			print('\nHiddenLayerSize:'+str(hidden_layer_size)+'|Lamda:'+str(lamda)+'|Iteration:'+str(i)+'|Cost:'+str(tmp_cost_norm)+'\n')

# 			i+=1

# 		cost_norm/=5
# 		cost_unnorm/=5
# 		if cost_norm < min_cost_norm:
# 			min_cost_norm = cost_norm
# 			opt_hidden_layer_size = hidden_layer_size
# 			opt_lamda = lamda
# 			opt_cost = cost_unnorm
# 			open(ult_opt_params_1_file, "w").writelines([l for l in open(opt_params_1_file).readlines()])
# 			open(ult_opt_params_2_file, "w").writelines([l for l in open(opt_params_2_file).readlines()])
# 		if lamda == 0:
# 			lamda = 0.01
# 		else:
# 			lamda = 2*lamda

# 	hidden_layer_size+=1
# print('OptHiddenLayerSize:'+str(opt_hidden_layer_size)+'|OptLamda:'+str(opt_lamda))
