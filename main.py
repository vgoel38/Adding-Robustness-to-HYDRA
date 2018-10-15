from init_params import init_params
from nn_cost_function import nn_cost_function
import numpy as np

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
#size of training data
m = len(X)

wait = input("PRESS ENTER TO CONTINUE.")



# =========== Phase 2 : Initializing Parameters ================

print('\nInitializing Neural Network Parameters ...\n')

initial_params_1 = init_params(input_layer_size, hidden_layer_size);
initial_params_2 = init_params(hidden_layer_size, num_labels);

#Unrolling into a single vector of parameters
nn_params = np.concatenate(((initial_params_1.T).ravel(),(initial_params_2.T).ravel()), axis=None)

wait = input("PRESS ENTER TO CONTINUE.")



# ============ Phase 3 : Feedforward Phase ================

print('\nFeedforward Phase ...\n')

#Weight regularization parameter
lamda = 1

J = nn_cost_function(nn_params, input_layer_size, hidden_layer_size, num_labels, X, Y, lamda);

# print('Cost at parameters : ' + J)

wait = input("PRESS ENTER TO CONTINUE.")



# ============ Phase 4 : Backpropagation Phase ================

print('\nBackpropagation Phase ...\n')

# grad = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lamda);

# print('Cost at parameters : ' + J)

wait = input("PRESS ENTER TO CONTINUE.")


# %% ========= Phase 4: Training NN ===================

print('\nTraining Neural Network... \n')

# %  After you have completed the assignment, change the MaxIter to a larger
# %  value to see how more training helps.
# options = optimset('MaxIter', 50);

# % Create "short hand" for the cost function to be minimized
# costFunction = @(p) nnCostFunction(p, ...
#                                    input_layer_size, ...
#                                    hidden_layer_size, ...
#                                    num_labels, X, y, lambda);

# % Now, costFunction is a function that takes in only one argument (the
# % neural network parameters)
# [nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

# % Obtain Theta1 and Theta2 back from nn_params
# Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
#                  hidden_layer_size, (input_layer_size + 1));

# Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
#                  num_labels, (hidden_layer_size + 1));

# fprintf('Program paused. Press enter to continue.\n');
# pause;




# %% ================= Phase 5: Prediction =================
# %  After training the neural network, we would like to use it to predict
# %  the labels. You will now implement the "predict" function to use the
# %  neural network to predict the labels of the training set. This lets
# %  you compute the training set accuracy.

# pred = predict(Theta1, Theta2, X);

# fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);


