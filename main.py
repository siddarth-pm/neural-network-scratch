import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
import pandas as pd

class DeepNeuralNet():
    # Done
    def __init__(self, layer_architecture, activations):
        self.initialization(layer_architecture, activations)
        self.train_data, self.test_data = self.import_data()
    
    # Done
    def initialization(self, layer_architecture, activations):
        self.params = {}
        for i in range(1, len(layer_architecture)):
            self.params["W"+str(i)] = np.random.randn(layer_architecture[i], layer_architecture[i-1]) * 0.01 # initialize weights to random small numbers
            self.params["b"+str(i)] = np.zeros((layer_architecture[i], 1)) # initialize bias to zeros
        for i in range(1, len(activations)):
            self.params["activation"+str(i)] = activations[i]
    
    # Done
    def import_data():
        X,  y = datasets.load_wine(return_X_y=True, as_frame=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        return ((X_train, y_train), (X_test, y_test))
    
    # Done
    def activation_derivative(x, activation):
        if activation == "sigmoid":
            return x*(1-x)
        elif activation == "relu":
            if x <= 0:
                return 0
            else:
                return 1
    
    # Done
    def activation_function(x, activation):
        if activation == "sigmoid":
            return 1/(1+np.exp(-x))
        if activation == "relu":
            return max(0, x)
    
    # Not Done
    def forward_prop(self, X, layer_architecture):
        prev_output = X
        info_for_backprop = []
        for i in range(1, len(layer_architecture)):
            Z = np.matmul(prev_output, self.params["W"+str(i)].T) + self.params["b"+str(i)]
            current_activation = self.params["activation"+str(i)]
            dZ = self.activation_derivative(Z, current_activation)
            info = [dZ, self.params["W"+str(i)], self.params["b"]+str(i)]
            info_for_backprop.append(info)
        
        return prev_output, info_for_backprop
    
    # Not Done
    def back_prop(self, info_for_backprop, alpha):
        derivatives = {}

        # need to calculate the activations already before doing this?
        cost = cost(Y, Yhat)
        cost_derivative = 1 # Find loss derivative

        # Compute derivative and end up finding 
        dA, dW, db = something

        for i in range(len(info_for_backprop)-1, 1, -1):
            # use dA from previous calculations to compute the dA, dW, and db for the layer one step backwards in the model


            # Then update dA, dW, and db for the previous layer
            self.params["W"+str(i)] = self.params["W"+str(i)] - dW * alpha
            self.params["b"+str(i)] = self.params["b"+str(i)] - db * alpha

        
    # Not Done
    def cost(Y, Y_hat, cost_function="crossentropy"):
        if cost_function == "crossentropy":
            ans = (-1/len(Y)) * np.sum((np.multiply(Y,np.log(Y_hat)) + np.multiply((1-Y),np.log(1-Y_hat))))
        elif cost_function == "MSE":
            ans = (1/len(Y)) * np.sum(Y - Y_hat)
        
        return np.squeeze(ans)
    
    # assuming last layer activation is sigmoid
    def cost_derivative(self, Y, Y_hat, cost_function="crossentropy"):
        if cost_function == "crossentropy":
            ans = -(np.divide(Y, Y_hat)) - np.divide(1 - Y, 1 - Y_hat)
        elif cost_function == "MSE":
            ans = 2*(Y-Y_hat) * self.activation_derivative(Y_hat, "sigmoid") * 
    # Not Done
    def train_model(self, X, learning_rate=0.01, iterations=1000):
        self.forward_prop(X, layer_architecture)

if __name__ == "main":
    layer_architecture = [3, 5, 3, 1]
    activations = ["relu", "relu", "relu", "sigmoid"]
    dnn = DeepNeuralNet(layer_architecture, activations)
    print()
