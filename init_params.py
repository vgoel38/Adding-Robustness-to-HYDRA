import random
import sys
import numpy as np

def init_params(layer_in, layer_out):
#Randomly initialize the parameters of a neural network layer with layer_in
#incoming connections and layer_out outgoing connections

	epsilon_init = 0.12

	def f(x):
		return (x * 2 * epsilon_init - epsilon_init)

	f = np.vectorize(f)

	params = np.random.rand(layer_out,layer_in + 1)

	return f(params)

if __name__ == "__main__":
    print(init_params(int(sys.argv[1]),int(sys.argv[2])))