import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
import pandas as pd

class DeepNeuralNet():
    def __init__(self, layer_architecture, activations):
        self.initialization(layer_architecture, activations)
    
    def initialization(self, layer_architecture, activations):
        self.params = {}
        self.layer_architecture = layer_architecture
        for i in range(1, len(layer_architecture)):
            self.params["W"+str(i)] = np.random.randn(layer_architecture[i], layer_architecture[i-1]) * 0.01 # initialize weights to random small numbers
            self.params["b"+str(i)] = np.zeros((layer_architecture[i], 1)) # initialize bias to zeros
        for i in range(1, len(activations)):
            self.params["activation"+str(i)] = activations[i]

    def activation_derivative(self, x, activation):
        if activation == "sigmoid":
            return x*(1-x)
        elif activation == "relu":
            return np.heaviside(x, 1)

    def activation_function(self, x, activation):
        if activation == "sigmoid":
            return 1/(1+np.exp(-x))
        if activation == "relu":
            return np.maximum(0, x)
    
    def forward_prop(self, X):
        prev_output = X
        info_for_backprop = []
        for i in range(1, len(self.layer_architecture)):
            print(self.params["W"+str(i)])
            Z = np.matmul(self.params["W"+str(i)], prev_output) + self.params["b"+str(i)]
            current_activation = self.params["activation"+str(i)]
            A = self.activation_function(Z, current_activation)
            info = [prev_output, Z, self.params["W"+str(i)], self.params["b"+str(i)]]
            info_for_backprop.append(info)
            prev_output = A
        return prev_output, info_for_backprop
    
    def back_prop(self, Y, Y_hat, info_for_backprop, alpha, clipping=False):
        # Y = Y.reshape(Y_hat.shape)
        cost_derivative = self.cost_derivative(Y, Y_hat, "crossentropy") # Find cost derivative
        print(Y)
        print(Y_hat)
        L = len(info_for_backprop)
        A, Z, W, b = info_for_backprop[L-1]
        dZ = cost_derivative * self.activation_derivative(Y_hat, "sigmoid")
        dA = np.matmul(W.T, dZ)
        dW = (1/A.shape[0])*np.matmul(dZ, A.T)
        db = (1/A.shape[0])*np.sum(dZ, axis=1, keepdims=True)
        if clipping:
            self.params["W"+str(L)] = self.params["W"+str(L)] - self.gradient_clip(dW, 1) * alpha
            self.params["b"+str(L)] = self.params["b"+str(L)] - self.gradient_clip(db, 1) * alpha
        else:
            self.params["W"+str(L)] = self.params["W"+str(L)] - dW * alpha
            self.params["b"+str(L)] = self.params["b"+str(L)] - db * alpha
        for i in reversed(range(L-1)):
            # use dA from previous calculations to compute the dA, dW, and db for the layer one step backwards in the model
            A, Z, W, b = info_for_backprop[i]
            dZ = dA  * self.activation_derivative(Z, self.params["activation"+str(i+1)])
            dW = (1/A.shape[1])*np.matmul(dZ,A.T)
            db = (1/A.shape[1])*np.sum(dZ, axis=1, keepdims=True)
            dA = np.matmul(W.T, dZ)
            if clipping:
                self.params["W"+str(i+1)] = self.params["W"+str(i+1)] - self.gradient_clip(dW, 1) * alpha
                self.params["b"+str(i+1)] = self.params["b"+str(i+1)] - self.gradient_clip(db, 1) * alpha
            else:
                self.params["W"+str(i+1)] = self.params["W"+str(i+1)] - dW * alpha
                self.params["b"+str(i+1)] = self.params["b"+str(i+1)] - db * alpha

    # Done
    def cost(self, Y, Y_hat, cost_function="crossentropy"):
        if cost_function == "crossentropy":
            ans = (-1/len(Y)) * np.sum(np.multiply(Y,np.log(Y_hat)) + np.multiply((1-Y),np.log(1-Y_hat)))
        return np.squeeze(ans)
    
    # assuming last layer activation is sigmoid
    def cost_derivative(self, Y, Y_hat, cost_function="crossentropy"):
        if cost_function == "crossentropy":
            ans = -(np.divide(Y, Y_hat)) - np.divide(1 - Y, 1 - Y_hat)
        return ans

    def fit_model(self, X, Y, alpha=0.01, iterations=100, clipping=False):
        for iteration in range(iterations): 
            prev_output, info_for_backprop = self.forward_prop(X)
            # print("cost:", self.cost(Y, prev_output, "crossentropy"))
            self.back_prop(Y, prev_output, info_for_backprop, alpha, clipping)

    def gradient_clip(self, gradient, c):
        return c * np.divide(gradient, np.linalg.norm(gradient))

    def predict(self, X):
        output = X
        for i in range(1, len(layer_architecture)):
            output = np.dot(self.params["W"+str(i)], output) + self.params["b"+str(i)]
            output = self.activation_function(output, self.params["activation"+str(i)])
        # print(output)
        return output

if __name__ == "__main__":
    X,  y = datasets.load_breast_cancer(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    X_train = X_train.to_numpy().T
    y_train = y_train.to_numpy().T
    print(X_train.shape)
    layer_architecture = [X_train.shape[0], 4, 1]
    activations = ["relu", "relu", "sigmoid"]
    dnn = DeepNeuralNet(layer_architecture, activations)
    dnn.fit_model(X_train, y_train, 0.0001, 10, clipping=False)
    X_test = X_test.to_numpy().T
    y_test = y_test.to_numpy().T
    print("final cost: ", dnn.cost(dnn.predict(X_test), y_test))