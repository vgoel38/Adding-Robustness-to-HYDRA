import sys
import numpy as np

from init_params import init_params
from nn_cost_function import nn_cost_function
from scipy.optimize import minimize

def main(table, X, Y, hidden_layer_size, lamda, params_1_file, params_2_file):

  input_layer_size  = len(X[0])
  num_labels = 1
  m = len(X)

  # =========== Initializing Parameters ================

  initial_params_1 = init_params(input_layer_size, hidden_layer_size)
  initial_params_2 = init_params(hidden_layer_size, num_labels)

  #Unrolling into a single vector of parameters
  nn_params = np.concatenate(((initial_params_1.T).ravel(),(initial_params_2.T).ravel()), axis=None)

  # ========= Training NN ===================

  # options = {'maxiter': 50}

  result = minimize(nn_cost_function, nn_params, jac=True, args = (input_layer_size, hidden_layer_size, num_labels, X, Y, lamda))

  #Reshaping nn_params back into the parameters params_1 and params_2
  params_1 = (result.x)[0:hidden_layer_size * (input_layer_size + 1)]
  params_1 = (np.reshape(params_1, (input_layer_size + 1, hidden_layer_size))).T
  params_2 = (result.x)[hidden_layer_size * (input_layer_size + 1):]
  params_2 = (np.reshape(params_2, (hidden_layer_size + 1, -1))).T

  sys.stdout = open(params_1_file,'w')
  for val in params_1:
      print(*val)

  sys.stdout = open(params_2_file,'w')
  for val in params_2:
      print(*val)

  sys.stdout = sys.__stdout__