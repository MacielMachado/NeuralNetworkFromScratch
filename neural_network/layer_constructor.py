import numpy as np
from activation_and_cost_funcs import ActivationFunctions, CostFunctions

class LayerConstructor(CostFunctions):
  def __init__(self,input_shape,output_shape,activation):
    self.weights = np.random.randn(output_shape,input_shape)
    self.bias = np.random.randn(1,output_shape)
    self.activation = activation
    self.a = None
    self.da_dz = None
    self.delta = None
    self.dz_dw = None
    self.dz_db = None

  def calculate_z(self):
    self.z = np.dot(self.input, self.weights.T) + self.bias

  def calculate_activation(self):
      self.a = ActivationFunctions().get_activation(self.activation, self.z)

class NeuralNetworkConstructor():
  def __init__(self):
    self.layers = []

  def compile(self,cost_func='mse',learning_rate=1e-3):
    self.cost_func = cost_func
    self.learning_rate = learning_rate

  def add_layer(self,input_shape,output_shape,activation):
    self.layers.append(LayerConstructor(input_shape=input_shape,
                                        output_shape=output_shape,
                                        activation=activation))

  def fit(self, X, y, epochs=100, verbose=10):
    for epoch in range(epochs + 1):
      y_hat = self.feedforward(X)
      self.backpropagation(y,y_hat)
      if epoch % verbose == 0:
        cost = CostFunctions()    
        # loss_train = self.cost_func(y, self.predict(X))
        loss_train = cost.get_cost(self.cost_func, y, self.predict(X))
        print(f'Epoch: {epoch}/{epochs} loss: {loss_train:.8f}')

  def predict(self, x):
    return self.feedforward(x)

  def feedforward(self, x):
    self.layers[0].input = x
    for count,layer in enumerate(self.layers):
      layer.calculate_z()
      layer.calculate_activation()
      if count != len(self.layers)-1: 
          self.layers[count+1].input = layer.a
    return self.layers[-1].a
  
  def backpropagation(self, y, y_pred):
    activation = ActivationFunctions()
    cost = CostFunctions()
    delta = cost.get_cost(self.cost_func, y, y_pred, derivative=True)
    for layer in reversed(self.layers):
      activation = ActivationFunctions()
      cost = CostFunctions()
      da_dz = activation.get_activation(layer.activation, layer.z, True) * delta
      delta = np.dot(da_dz, layer.weights)
      layer.dz_dw = np.dot(da_dz.T, layer.input)
      layer.dz_db = da_dz.sum(axis = 0,keepdims=True)
      layer.weights = layer.weights - self.learning_rate*layer.dz_dw
      layer.bias = layer.bias - self.learning_rate*layer.dz_db