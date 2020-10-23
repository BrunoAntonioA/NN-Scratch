import math
import numpy as np
import random

#Neuron Class
class Neuron:
    def __init__(self, previousLayer, previousLayerWeights, value, activationFunction):
        self.output = 0
        self.delta = 0
        self.value = value
        self.activationFunction = activationFunction
        self.previousLayer = previousLayer
        self.previousLayerWeights = previousLayerWeights
        self.nextLayer = []
    

#Neural Network Class
class NeuralNetwork:
    # Where n is the input_layer len and m is the output_layer len
    def __init__(self, n, m):
        self.input_layer = np.empty(n)
        self.output_layer = np.empty(m)
        self.hidden_layers = np.array([])

    def addLayer(self, n, activationFunction):
        layer = np.array([])
        lenHidLay = self.hidden_layers.shape[0]
        for i in range(n):
            if lenHidLay != 0:
                lenPreviousLayer = self.hidden_layers[lenHidLay - 1].shape[0]
                neuron = Neuron(self.hidden_layers[lenHidLay - 1], np.random.random(size=(lenPreviousLayer, 1)), 0, activationFunction)
                layer = np.append(layer, neuron)
                print("tiene hidden layers")
            else: 
                print("no tiene hidden layers")
                neuron = Neuron(np.array([]), np.array([]), 0, activationFunction)
                layer = np.append(layer, neuron)
        print("self.hidden_layers: ", self.hidden_layers)    
        self.hidden_layers = np.concatenate((self.hidden_layers, layer), axis=0)    
        self.hidden_layers = np.concatenate((self.hidden_layers, layer), axis=0)
        print("self.hidden_layers: ", self.hidden_layers) 

    def forward(self, x):
        if self.hidden_layer.shape[0] != 0:
            self.input_layer = x
        else:
            print("To use the neural network your need at least an input layer, a hidden layer and an output layer")    

# Sigmoid function
def sigmoid(x, derivada=False):
    if derivada:
        return sigmoid(x)*(1-sigmoid(x)) 
    return 1 / (1 + np.exp(-x))

def test():
    print("testing function")

nn = NeuralNetwork(8, 2)
nn.addLayer(10, sigmoid)
#nn.addLayer(10, sigmoid)


