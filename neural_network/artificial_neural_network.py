import numpy as np
from backpropagation import backpropagation
from feedforward import feedforward
from layer_constructor import layer_constructor
from neurons import cost_functions

class artificial_neural_network():
    def __init__(self):
        self.layers = []

    def add_layer(self,input_dim,output_dim,activation_function):
        self.layers.append(layer_constructor(input_dim=input_dim,output_dim=output_dim,neuron_activation=activation_function))

    def compile(self,cost_function,learning_rate):
        self.learning_rate = learning_rate
        self.cost_function = cost_function 

    def fit(self,X,y,epochs):
        cost_list = []
        for _ in range(epochs):
            y_hat = feedforward().run_feedforward(X, self.layers)
            cost = backpropagation().run_backpropagation(y,y_hat,self.layers,self.cost_function,self.learning_rate)
            cost_list.append(cost)
        return cost_list
        print('hi')

    def predict(self,X):
        return feedforward().run_feedforward(X,self.layers)