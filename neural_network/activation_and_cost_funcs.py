import numpy as np

class ActivationFunctions():
    def __init__(self):
        pass

    def get_activation(self, activation_function, z, derivative = False):
        if activation_function == 'linear':
            return self.linear(z, derivative=False)
        elif activation_function == 'sigmoid':
            return self.sigmoid(z, derivative=False)
        elif activation_function == 'tanh':
            return self.tanh(z, derivative=False)
        elif activation_function == 'relu':
            return self.relu(z, derivative=False)

    def linear(self, z, derivative=False):
        if derivative: return np.ones_like(z)  
        else: return z

    def sigmoid(self, z, derivative=False):
        if derivative: return self.sigmoid(z)*(1 - self.sigmoid(z)) 
        else: return 1.0/(1.0 + np.exp(-z))

    def tanh(self, z, derivative=False):
        if derivative: return 1 - self.tanh(z)**2 
        else: return np.tanh(z)

    def relu(self, z, derivative=False):
        return 1.*(z>0) if derivative else z*(z>0)

class CostFunctions():
    def __init__(self):
        pass

    def get_cost(self, cost_function, y, y_hat, derivative = False):
        if cost_function == 'mae':
            return self.mae(y, y_hat, derivative)
        elif cost_function == 'mse':
            return self.mse(y, y_hat, derivative)
        elif cost_function == 'cross_entropy':
            return self.cross_entropy(y, y_hat, derivative)

    def mae(self, y, y_hat, derivative=False):
        if derivative: return np.where(y_hat>y,1,-1)/y.shape[0]
        else: return np.mean(np.abs(y-y_hat))

    def mse(self, y, y_hat, derivative=False):
        if derivative: return -(y - y_hat)/y.shape[0]
        else: return 0.5*np.mean((y-y_hat)**2)

    def cross_entropy(self, y, y_hat, derivative=False):
        if derivative: return -(y-y_hat)/(y_hat*(1-y_hat)*y.shape[0])
        else: return -np.mean(y*np.log(y_hat) + (1-y)*np.log(1-y_hat))