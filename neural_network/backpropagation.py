import numpy as np
from neurons import cost_functions, cost_functions_derivative, activations, activations_derivative

class backpropagation():
    def run_backpropagation(self,y,y_hat,layers,cost_function,learning_rate):
        cost_function_derivative = self.get_cost_derivative(cost_function,y,y_hat)
        cost = self.get_cost(cost_function,y,y_hat)
        for count,layer in enumerate(reversed(layers)):
            # if count != 0:
            #     layer.z = np.append([[1]]*layer.z.shape[0],layer.z,axis=1)
            activation_derivative = self.activation_derivative(layer.z,layer.neuron_activation)*cost_function_derivative.reshape(layer.z.shape)
            input_derivative = np.dot(activation_derivative,layer.weights[:,1:])
            layer.weight_derivative = np.dot(activation_derivative.T,layer.input)
            layer.weights = layer.weights - learning_rate*layer.weight_derivative
            cost_function_derivative = input_derivative
        return cost

    def get_cost(self,cost_function,y,y_hat):
        if cost_function=='cross_entropy':
            return cost_functions().cross_entropy(y,y_hat)
        elif cost_function=='mean_absolute_error':
            return cost_functions().mean_absolute_error(y,y_hat)
        elif cost_function=='mean_squared_error':
            return cost_functions().mean_squared_error(y,y_hat)

    def get_cost_derivative(self,cost_function,y,y_hat):
        y_hat = y_hat.reshape(y.shape)
        if cost_function=='cross_entropy':
            return cost_functions_derivative().cross_entropy_derivative(y,y_hat)
        elif cost_function=='mean_absolute_error':
            return cost_functions_derivative().mean_absolute_error_derivative(y,y_hat)
        elif cost_function=='mean_squared_error':
            return cost_functions_derivative().mean_squared_error_derivative(y,y_hat)    
            
    def activation(self,activation_function, z):
        if activation_function == 'tanh':
            return activations().tanh(z)
        elif activation_function == 'sigmoid':
            return activations().sigmoid(z)
        elif activation_function == 'relu':
            return activations.relu(z)

    def activation_derivative(self,z, activation_function):
        if activation_function == 'tanh':
            return activations_derivative().tanh_derivative(z)
        elif activation_function == 'sigmoid':
            return activations_derivative().sigmoid_derivative(z)
        elif activation_function == 'relu':
            return activations_derivative.relu_derivative(z)
