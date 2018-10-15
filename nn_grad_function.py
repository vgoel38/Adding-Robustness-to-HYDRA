% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.

# Theta1_grad = zeros(size(Theta1));
# Theta2_grad = zeros(size(Theta2));

delta_a1 = zeros(hidden_layer_size, input_layer_size+1);
delta_a2 = zeros(num_labels, hidden_layer_size+1);

error_h = h - Y;
error_a2 = (error_h * Theta2) .* [ones(size(z2,1),1) sigmoidGradient(z2)];

delta_a2 = delta_a2 + error_h' * a2;
delta_a1 = delta_a1 + error_a2(:,[2:size(error_a2,2)])' * X;

Theta1_grad = delta_a1/m;
Theta2_grad = delta_a2/m;

%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


Theta1_grad(:,[2:size(Theta1_grad,2)]) += (lambda/m) * Theta1(:,[2:size(Theta1,2)]);
Theta2_grad(:,[2:size(Theta2_grad,2)]) += (lambda/m) * Theta2(:,[2:size(Theta2,2)]);














% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
