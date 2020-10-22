import math
import numpy as np
import random

#Neuron Class
class Neuron:
    def __init__(self, previousLayer, previousLayerWeights, value, activationFunction):
        self.output = 0
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
        self.hidden_layers = []
                
    def forward(self, x):
        if len(self.hidden_layers) != 0:
            self.input_layer = x

        else:
            print("Is not possible to use the neural network without hidden layers")    

# Sigmoid function
def sigmoid(x, derivada=False):
    if derivada:
        return sigmoid(x)*(1-sigmoid(x)) 
    return 1 / (1 + np.exp(-x))





