import random
import sys

def init_params(layer_in, layer_out):
#Randomly initialize the parameters of a neural network layer with layer_in
#incoming connections and layer_out outgoing connections

	epsilon_init = 0.12;

	params = []

	for i in range(layer_out):
		params.append([])
		#one extra node in the layer_in represents the bias node
		for j in range(layer_in + 1):
			params[i].append(random.random() * 2 * epsilon_init - epsilon_init)

	return params

if __name__ == "__main__":
    print(init_params(int(sys.argv[1]),int(sys.argv[2])))