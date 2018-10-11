from init_params import init_params

# Setting up the parameters
num_attributes = 4
# For each attribute, first input node represents operator <= and second input node represents operator >=
input_layer_size  = 2*num_attributes
hidden_layer_size = 2*num_attributes
num_labels = 1


# =========== Loading and Visualizing Data =============

# % Load Training Data
# fprintf('Loading and Visualizing Data ...\n')

# load('ex4data1.mat');
# m = size(X, 1);

# % Randomly select 100 data points to display
# sel = randperm(m);
# sel = sel(1:100);

# displayData(X(sel, :));

# fprintf('Program paused. Press enter to continue.\n');
# pause;

# ================ Initializing Parameters ================

print('\nInitializing Neural Network Parameters ...\n')

initial_params_1 = init_params(input_layer_size, hidden_layer_size);
initial_params_2 = init_params(hidden_layer_size, num_labels);

nn_params = []

# Unrolling parameters
for params in initial_params_1:
    for param in params:
        nn_params.append(param)

for params in initial_params_2:
    for param in params:
        nn_params.append(param)

# ================ Feedforward Phase ================

print('\nFeedforward Phase ...\n')

# Weight regularization parameter (we set this to 0 here).
lamda = 0

# J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lamda);

# print('Cost at parameters : ' + J)

wait = input("PRESS ENTER TO CONTINUE.")

# %% =============== Part 4: Implement Regularization ===============
# %  Once your cost function implementation is correct, you should now
# %  continue to implement the regularization with the cost.
# %

# fprintf('\nChecking Cost Function (w/ Regularization) ... \n')

# % Weight regularization parameter (we set this to 1 here).
# lambda = 1;

# J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, ...
#                    num_labels, X, y, lambda);

# fprintf(['Cost at parameters (loaded from ex4weights): %f '...
#          '\n(this value should be about 0.383770)\n'], J);

# fprintf('Program paused. Press enter to continue.\n');
# pause;


# %% ================ Part 5: Sigmoid Gradient  ================
# %  Before you start implementing the neural network, you will first
# %  implement the gradient for the sigmoid function. You should complete the
# %  code in the sigmoidGradient.m file.
# %

# fprintf('\nEvaluating sigmoid gradient...\n')

# g = sigmoidGradient([-1 -0.5 0 0.5 1]);
# fprintf('Sigmoid gradient evaluated at [-1 -0.5 0 0.5 1]:\n  ');
# fprintf('%f ', g);
# fprintf('\n\n');

# fprintf('Program paused. Press enter to continue.\n');
# pause;


# %% =============== Part 7: Implement Backpropagation ===============
# %  Once your cost matches up with ours, you should proceed to implement the
# %  backpropagation algorithm for the neural network. You should add to the
# %  code you've written in nnCostFunction.m to return the partial
# %  derivatives of the parameters.
# %
# fprintf('\nChecking Backpropagation... \n');

# %  Check gradients by running checkNNGradients
# checkNNGradients;

# fprintf('\nProgram paused. Press enter to continue.\n');
# pause;


# %% =============== Part 8: Implement Regularization ===============
# %  Once your backpropagation implementation is correct, you should now
# %  continue to implement the regularization with the cost and gradient.
# %

# fprintf('\nChecking Backpropagation (w/ Regularization) ... \n')

# %  Check gradients by running checkNNGradients
# lambda = 3;
# checkNNGradients(lambda);

# % Also output the costFunction debugging values
# debug_J  = nnCostFunction(nn_params, input_layer_size, ...
#                           hidden_layer_size, num_labels, X, y, lambda);

# fprintf(['\n\nCost at (fixed) debugging parameters (w/ lambda = %f): %f ' ...
#          '\n(for lambda = 3, this value should be about 0.576051)\n\n'], lambda, debug_J);

# fprintf('Program paused. Press enter to continue.\n');
# pause;


# %% =================== Part 8: Training NN ===================
# %  You have now implemented all the code necessary to train a neural 
# %  network. To train your neural network, we will now use "fmincg", which
# %  is a function which works similarly to "fminunc". Recall that these
# %  advanced optimizers are able to train our cost functions efficiently as
# %  long as we provide them with the gradient computations.
# %
# fprintf('\nTraining Neural Network... \n')

# %  After you have completed the assignment, change the MaxIter to a larger
# %  value to see how more training helps.
# options = optimset('MaxIter', 50);

# %  You should also try different values of lambda
# lambda = 1;

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


# %% ================= Part 9: Visualize Weights =================
# %  You can now "visualize" what the neural network is learning by 
# %  displaying the hidden units to see what features they are capturing in 
# %  the data.

# fprintf('\nVisualizing Neural Network... \n')

# displayData(Theta1(:, 2:end));

# fprintf('\nProgram paused. Press enter to continue.\n');
# pause;

# %% ================= Part 10: Implement Predict =================
# %  After training the neural network, we would like to use it to predict
# %  the labels. You will now implement the "predict" function to use the
# %  neural network to predict the labels of the training set. This lets
# %  you compute the training set accuracy.

# pred = predict(Theta1, Theta2, X);

# fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);


