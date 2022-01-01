import numpy as np
from layer_constructor import layer_constructor

class feedforward():
    def run_feedforward(self, X, layers):
        layers[0].input = X
        for count,layer in enumerate(layers):
            layer.add_bias()
            layer.calculate_z()
            layer.activation()
            if count != len(layers)-1: 
                layers[count+1].input = layer.layer_activation
        return layer.layer_activation

    def add_bias(self, layer_input):
        return np.append([[1]]*layer_input.shape[0],layer_input,axis=1)