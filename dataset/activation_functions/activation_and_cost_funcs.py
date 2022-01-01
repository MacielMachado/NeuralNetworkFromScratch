import numpy as np

class ActivationFunctions():
    def __init__(self):
        pass

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

    def mae(y,y_hat,derivative=False):
        if derivative: return np.where(y_hat>y,1,-1)/y.shape[0]
        else: return np.mean(np.abs(y-y_hat))

    def mse(y,y_hat,derivative=False):
        if derivative: return -(y - y_hat)/y.shape[0]
        else: return 0.5*np.mean((y-y_hat)**2)

    def cross_entropy(y,y_hat,derivative=False):
        if derivative: return -(y-y_hat)/(y_hat*(1-y_hat)*y.shape[0])
        else: return -np.mean(y*np.log(y_hat) + (1-y)*np.log(1-y_hat))