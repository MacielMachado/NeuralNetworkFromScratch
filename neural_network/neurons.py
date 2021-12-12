import numpy as np

class neuron_builder():
    def __init__():
        pass

class activations():
    def __init__(self):
        pass
    def sigmoid(self,z):
        return 1/(1+np.power(np.e,-z))

class activations_derivative(activations):
    def __init__(self):
        pass
    def sigmoid__derivative(self,z):
        return activations().sigmoid(z)*(1-activations().sigmoid(z))