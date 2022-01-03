import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from activation_and_cost_funcs import ActivationFunctions, CostFunctions
from layer_constructor import LayerConstructor, NeuralNetworkConstructor

# Data:
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
plt.waitforbuttonpress()

# Training
input_shape, output_shape = x.shape[1], y.shape[1]
nn = NeuralNetworkConstructor()
nn.add_layer(input_shape=input_shape, output_shape=8,activation='relu')
nn.add_layer(input_shape=8, output_shape=8,activation='relu')
nn.add_layer(input_shape=8, output_shape=8,activation='relu')
nn.add_layer(input_shape=8, output_shape=4,activation='relu')
nn.add_layer(input_shape=4, output_shape=output_shape,activation='sigmoid')
nn.compile(cost_func='cross_entropy',learning_rate=1e-2)
nn.fit(x,y,epochs=5000,verbose=100)
print('Hi')


# Plot
x1, x2 = np.meshgrid(np.linspace(-0.75, 1, 100), np.linspace(-0.75, 1, 100))
x_mesh = np.array([x1.ravel(), x2.ravel()]).T
plt.figure(figsize=(6,6))
y_mesh = np.where(nn.predict(x_mesh) <= 0.5, 0, 1)

plt.scatter(x[:,0], x[:,1], c=list(np.array(y).ravel()), cmap='RdYlBu')
plt.contourf(x1, x2, y_mesh.reshape(x1.shape), cmap='RdYlBu', alpha=0.5)
plt.waitforbuttonpress()