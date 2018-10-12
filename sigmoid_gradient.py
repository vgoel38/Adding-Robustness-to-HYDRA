import math
import numpy as np
import sys

def sigmoid_gradient(z):
#returns the gradient of the sigmoid function evaluated at z (can be scalar/vector/matrix)

	def sigmoid(x):
	  return 1 / (1 + math.exp(-x))

	sigmoid = np.vectorize(sigmoid)

	z = sigmoid(z);

	z = np.multiply(z, (1-z))
	return z

if __name__ == "__main__":
    print(sigmoid_gradient(np.full((3, 5), 0)))