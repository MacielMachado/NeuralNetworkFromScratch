import numpy as np

class neuron_builder():
    def __init__():
        pass

class cost_functions():
    def __init__(self):
        pass
    def mean_absolute_error(self,y,y_hat):
        return np.mean(np.abs(y-y_hat))
    def mean_squared_error(self,y,y_hat):
        return np.mean(np.power(y-y_hat,2))/2

class cost_functions_derivative():
    def __init__():
        pass
    def mean_absolute_error_derivative(self,y,y_hat):
        return np.where(y_hat>y,1,-1)/len(y)
    def mean_squared_error_derivative(self,y,y_hat):
        return -(y-y_hat)/len(y)

class activations():
    def __init__(self):
        pass
    def sigmoid(self,z):
        return 1/(1+np.power(np.e,-z))
    def relu(self,z):
        return z*(z>0)
    def tanh(self,z):
        return np.tanh(z)

class activations_derivative(activations):
    def __init__(self):
        pass
    def sigmoid__derivative(self,z):
        return activations().sigmoid(z)*(1-activations().sigmoid(z))
    def relu_derivative(self,z):
        return 1.*(z>0)
    def tanh_derivative(self,z):
        return 1 - np.power(activations().tanh(z),2)