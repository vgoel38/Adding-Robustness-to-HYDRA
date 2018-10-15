import math
import numpy as np
import sys

def sigmoid(x):
#returns the value of the sigmoid function evaluated at z (can be scalar/vector/matrix)

	return 1 / (1 + math.exp(-x))

if __name__ == "__main__":
	sigmoid = np.vectorize(sigmoid)
	print(sigmoid(np.array([1,2])))