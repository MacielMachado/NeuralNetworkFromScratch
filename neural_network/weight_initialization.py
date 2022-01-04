from _typeshed import IdentityFunction
import numpy as np

class WeightInializator():
    def __init__(self):
        pass
    def initializate_weight(self, inializator):
        if inializator == 'random_uniform':
            pass
        elif inializator == 'random_normal':
            pass
        elif inializator == 'glorot_uniform':
            pass
        elif inializator == 'glorot_normal':
            pass
        elif inializator == 'ones':
            pass
        elif inializator == 'zeros':
            pass

    def random_uniform_initializator(self, output_shape, input_shape):
        return np.random.uniform(size = (output_shape, input_shape))

    def random_normal_initializator(self, output_shape, input_shape):
        return np.random.normal(size = (output_shape, input_shape))

    def glorot_uniform_initializator(self, output_shape, input_shape):
        return

    def glorot_normal_initializator(pself, output_shape, input_shape):
        return

    def ones_initializator(pself, output_shape, input_shape):
        return np.ones((output_shape, input_shape))

    def zeros_initializator(self, output_shape, input_shape):
        return np.zeros((output_shape, input_shape))
