import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Activation functions
def linear(z, derivative=False):
    return np.ones_like(x) if derivative else z
def sigmoid(z, derivative=False):
    return sigmoid(z)*(1 - sigmoid(z)) if derivative else 1.0/(1.0 + np.exp(-z))
def tanh(z, derivative=False):
    return 1 - tanh(z)**2 if derivative else np.tanh(z)
def relu(z, derivative=False):
    return 1.*(z>0) if derivative else z*(z>0)

# Cost functions
def mae(y,y_hat,derivative=False):
    if derivative:
        return np.where(y_hat>y,1,-1)/y.shape[0]
    else:
        return np.mean(np.abs(y-y_hat))

def mse(y,y_hat,derivative=False):
    if derivative:
        return -(y - y_hat)/y.shape[0]
    else:
        return 0.5*np.mean((y-y_hat)**2)

def cross_entropy(y,y_hat,derivative=False):
    if derivative:
        return -(y-y_hat)/(y_hat*(1-y_hat)*y.shape[0])
    else:
        return -np.mean(y*np.log(y_hat) + (1-y)*np.log(1-y_hat))

# Implementation
class LayerConstructor():
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
      self.a = self.activation(self.z)

class NeuralNetworkConstructor():
  def __init__(self):
    self.layers = []

  def compile(self,cost_func=mse,learning_rate=1e-3):
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
        loss_train = self.cost_func(y, self.predict(x))
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
    delta = self.cost_func(y,y_pred,derivative=True)
    for layer in reversed(self.layers):
      da_dz = layer.activation(layer.z, derivative=True)*delta
      delta = np.dot(da_dz, layer.weights)
      layer.dz_dw = np.dot(da_dz.T, layer.input)
      layer.dz_db = da_dz.sum(axis = 0,keepdims=True)
      layer.weights = layer.weights - self.learning_rate*layer.dz_dw
      layer.bias = layer.bias - self.learning_rate*layer.dz_db

# Circle plot
df=pd.read_csv('./dataset/classification2.txt', header=None)
X=df.iloc[0:100,:-1].values
x=X
y=df.iloc[0:100,-1].values.reshape(-1,1)
pos , neg = (y==1).reshape(100,1) , (y==0).reshape(100,1)
plt.scatter(X[pos[:,0],0],X[pos[:,0],1],c="r",marker="+")
plt.scatter(X[neg[:,0],0],X[neg[:,0],1],marker="o",s=10)
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend(["Accepted","Rejected"],loc=0)

# Training
input_shape, output_shape = x.shape[1], y.shape[1]

nn = NeuralNetworkConstructor()
nn.add_layer(input_shape=input_shape, output_shape=8,activation=relu)
nn.add_layer(input_shape=8, output_shape=8,activation=relu)
nn.add_layer(input_shape=8, output_shape=8,activation=relu)
nn.add_layer(input_shape=8, output_shape=4,activation=relu)
nn.add_layer(input_shape=4, output_shape=output_shape,activation=sigmoid)
nn.compile(cost_func=cross_entropy,learning_rate=1e-2)
nn.fit(x,y,epochs=5000,verbose=100)

# Plot
x1, x2 = np.meshgrid(np.linspace(-0.75, 1, 100), np.linspace(-0.75, 1, 100))
x_mesh = np.array([x1.ravel(), x2.ravel()]).T
plt.figure(figsize=(6,6))
y_mesh = np.where(nn.predict(x_mesh) <= 0.5, 0, 1)

plt.scatter(x[:,0], x[:,1], c=list(np.array(y).ravel()), cmap='RdYlBu')
plt.contourf(x1, x2, y_mesh.reshape(x1.shape), cmap='RdYlBu', alpha=0.5)

# Accuracy - training set
df=pd.read_csv('./dataset/classification2.txt', header=None)
X=df.iloc[100:,:-1].values
x=X
y=df.iloc[100:,-1].values.reshape(-1,1)
y_pred = nn.predict(x)
print('Accuracy: {:.2f}%'.format((100*sum(y==(y_pred>0.5).astype(int))/len(y))[0]))

# Accuracy - test set
x1, x2 = np.meshgrid(np.linspace(-0.75, 1, 100), np.linspace(-0.75, 1, 100))
x_mesh = np.array([x1.ravel(), x2.ravel()]).T
plt.figure(figsize=(6,6))
y_mesh = np.where(nn.predict(x_mesh) <= 0.5, 0, 1)

plt.scatter(x[:,0], x[:,1], c=list(np.array(y).ravel()), cmap='RdYlBu')
plt.contourf(x1, x2, y_mesh.reshape(x1.shape), cmap='RdYlBu', alpha=0.5)