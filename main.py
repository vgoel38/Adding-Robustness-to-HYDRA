from init_params import init_params
from nn_cost_function import nn_cost_function
import numpy as np
from scipy.optimize import minimize
import sys

# Setting up the parameters
num_attributes = 4
# For each attribute, first input node represents operator <= and second input node represents operator >=
input_layer_size  = 2*num_attributes
hidden_layer_size = 2*num_attributes
num_labels = 1


# =========== Phase 1 : Loading Training Data =============

print('Loading and Visualizing Data ...\n')

X = np.loadtxt('table_aka_title/aka_title_input.txt', delimiter=' ')
Y = np.loadtxt('table_aka_title/aka_title_output.txt')
Y = np.reshape(Y,(-1,1))
#size of training data
m = len(X)

wait = input("PRESS ENTER TO CONTINUE.")



# =========== Phase 2 : Initializing Parameters ================

print('\nInitializing Neural Network Parameters ...\n')

initial_params_1 = init_params(input_layer_size, hidden_layer_size)
initial_params_2 = init_params(hidden_layer_size, num_labels)

#Unrolling into a single vector of parameters
nn_params = np.concatenate(((initial_params_1.T).ravel(),(initial_params_2.T).ravel()), axis=None)

wait = input("PRESS ENTER TO CONTINUE.")



# ============ Phase 3 : Feedforward Phase ================

print('\nFeedforward Phase ...\n')

#Weight regularization parameter
lamda = 1

J, grad = nn_cost_function(nn_params, input_layer_size, hidden_layer_size, num_labels, X, Y, lamda)

wait = input("PRESS ENTER TO CONTINUE.")


# ========= Phase 4: Training NN ===================

print('\nTraining Neural Network... \n')

options = {'maxiter': 50}

result = minimize(nn_cost_function, nn_params, jac=True, args = (input_layer_size, hidden_layer_size, num_labels, X, Y, lamda), options=options)

#Reshaping nn_params back into the parameters params_1 and params_2
params_1 = (result.x)[0:hidden_layer_size * (input_layer_size + 1)]
params_1 = (np.reshape(params_1, (input_layer_size + 1, hidden_layer_size))).T
params_2 = (result.x)[hidden_layer_size * (input_layer_size + 1):]
params_2 = (np.reshape(params_2, (hidden_layer_size + 1, -1))).T

sys.stdout = open('table_aka_title/aka_title_params_1.txt','w')
for val in params_1:
    print(*val)

sys.stdout = open('table_aka_title/aka_title_params_2.txt','w')
for val in params_2:
    print(*val)

sys.stdout = sys.__stdout__

wait = input("PRESS ENTER TO CONTINUE.")




# %% ================= Phase 5: Prediction =================
# %  After training the neural network, we would like to use it to predict
# %  the labels. You will now implement the "predict" function to use the
# %  neural network to predict the labels of the training set. This lets
# %  you compute the training set accuracy.

# pred = predict(Theta1, Theta2, X);

# fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);


