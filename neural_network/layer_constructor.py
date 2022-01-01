import numpy as np
from neurons import activations

class layer_constructor():
    def __init__(self,input_dim,output_dim,neuron_activation):
        # self.input = []
        np.random.seed(42)
        self.weights = np.random.uniform(low=0.25,high=0.75,size=[output_dim,input_dim+1])
        self.neuron_activation = neuron_activation
        # self.z = []
        # self.layer_activation = []
        # self.weight_derivatives = []

    def calculate_z(self):
        self.z = np.dot(self.weights,self.input.T).reshape(self.input.T.shape[-1],self.weights.shape[0])

    def activation(self):
        if self.neuron_activation == 'tanh':
            self.layer_activation = activations().tanh(self.z)
        elif self.neuron_activation == 'sigmoid':
            self.layer_activation = activations().sigmoid(self.z)
        elif self.neuron_activation == 'relu':
            self.layer_activation = activations.relu(self.z)

    def add_bias(self):
        self.input = np.append([[1]]*self.input.shape[0],self.input,axis=1)