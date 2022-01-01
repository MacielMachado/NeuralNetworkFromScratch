import numpy as np

class ActivationFunctions():
    def __init__():
        pass
    def linear(self, z, derivative=False):
        if derivative:
            return np.ones_like(z)  
        else:
            return z

    def sigmoid(self, z, derivative=False):
        if derivative:
            return self.sigmoid(z)*(1 - self.sigmoid(z)) 
        else:
            return 1.0/(1.0 + np.exp(-z))

    def tanh(self, z, derivative=False):
        if derivative:
          return 1 - self.tanh(z)**2 
        else:
            return np.tanh(z)

    def relu(self, z, derivative=False):
        return 1.*(z>0) if derivative else z*(z>0)