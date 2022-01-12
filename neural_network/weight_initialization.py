import numpy as np

class WeightInializator():
    def __init__(self):
        pass
    def initializate(self, inializator, output_shape, input_shape):
        if inializator == 'random_uniform':
            return self.random_uniform_initializator(output_shape, input_shape)
        elif inializator == 'random_normal':
            return self.random_normal_initializator(output_shape, input_shape)
        elif inializator == 'glorot_uniform':
            return self.glorot_uniform_initializator(output_shape, input_shape)
        elif inializator == 'glorot_normal':
            return self.glorot_normal_initializator(output_shape, input_shape)
        elif inializator == 'ones':
            return self.ones_initializator(output_shape, input_shape)
        elif inializator == 'zeros':
            return self.zeros_initializator(output_shape, input_shape)

    def random_uniform_initializator(self, output_shape, input_shape):
        return np.random.uniform(size = (output_shape, input_shape))

    def random_normal_initializator(self, output_shape, input_shape):
        return np.random.normal(size = (output_shape, input_shape))

    def glorot_uniform_initializator(self, output_shape, input_shape):
        sigma = np.sqrt(6 / (output_shape + input_shape))
        return 2*sigma*np.random.uniform(size = (output_shape, input_shape)) \
               - sigma

    def glorot_normal_initializator(pself, output_shape, input_shape):
        mu = 0
        sigma = np.sqrt(2 / (output_shape + input_shape))
        return sigma*np.random.normal(size = (output_shape, input_shape)) + mu

    def ones_initializator(pself, output_shape, input_shape):
        return np.ones((output_shape, input_shape))

    def zeros_initializator(self, output_shape, input_shape):
        return np.zeros((output_shape, input_shape))
