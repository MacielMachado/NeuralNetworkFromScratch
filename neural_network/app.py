import numpy as np
import pandas as pd
from neurons import activations, activations_derivative, cost_functions
from artificial_neural_network import artificial_neural_network
import matplotlib.pyplot as plt

# Dataset
ds_path = '/Users/brunomaciel/Documents/git/NeuralNetworkFromScratch/dataset/classification2.txt'
dataset = pd.read_csv(ds_path,sep=',',names=['X1','X2','y'])
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values
train_test_split_lim = round(len(X)*0.7)
X_train, y_train = X[:train_test_split_lim], y[:train_test_split_lim]
X_test, y_test = X[train_test_split_lim:], y[train_test_split_lim:]

if False:
    df=pd.read_csv(ds_path, header=None)
    X=df.iloc[:,:-1].values
    y=df.iloc[:,-1].values
    pos , neg = (y==1).reshape(118,1) , (y==0).reshape(118,1)
    plt.scatter(X[pos[:,0],0],X[pos[:,0],1],c="r",marker="+")
    plt.scatter(X[neg[:,0],0],X[neg[:,0],1],marker="o",s=10)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend(["Accepted","Rejected"],loc=0)

X_train = np.array([[0.05, 0.1]])
y_train = np.array([[0.01, 0.99]])
# X_train = np.array(list(range(100))).reshape(-1, 1)
# y_train = X_train + np.pi*100
# plt.figure()
# plt.plot(X_train)
# plt.plot(y_train)

x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 0, 1]).reshape(-1, 1)

print(x.shape, y.shape)
plt.scatter(x[:,0], x[:,1], c=list(np.array(y).ravel()), s=15, cmap='bwr')

input_dim, output_dim = x.shape[1], y.shape[1]


ann = artificial_neural_network()
ann.add_layer(input_dim=X_train.shape[1],output_dim=2,activation_function='sigmoid')
ann.add_layer(input_dim=2,output_dim=2,activation_function='sigmoid')
ann.add_layer(input_dim=2,output_dim=2,activation_function='sigmoid')
ann.compile(cost_function='mean_squared_error',learning_rate=0.5)
ann.fit(X=X_train,y=y_train,epochs=1)
prediction = ann.predict(X_train)
print(f'Prediction: {((prediction > 0.5).astype(int)).reshape(y_train.shape)}\nExpected velue: {y_train}')
# print(f'The)
print(ann.layers)

