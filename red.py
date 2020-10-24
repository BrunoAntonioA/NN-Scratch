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
    
    def calculateActivation(self):
        sum = 0
        lenPrevLayer = len(self.previousLayer)
        for i in range(lenPrevLayer):
            sum = sum + self.previousLayer[i].value * self.previousLayerWeights[i]
        self.value = self.activationFunction(sum)
        return self.value
    
#Neural Network Class
class NeuralNetwork:
    # Where n is the input_layer len and m is the output_layer len
    def __init__(self, n):
        self.n = n
        self.input_layer = np.empty(n)
        self.hidden_layers = []

    def addLayer(self, n, activationFunction):
        layer = np.array([])
        lenHidLay = len(self.hidden_layers)
        for i in range(n):
            if lenHidLay != 0:
                lenPreviousLayer = len(self.hidden_layers[lenHidLay - 1])
                neuron = Neuron(self.hidden_layers[lenHidLay - 1], np.random.random(size=(lenPreviousLayer, 1)), 0, activationFunction)
                layer = np.append(layer, neuron)
            else: 
                neuron = Neuron(np.array([]), np.full(self.n, 1), 0, activationFunction)
                layer = np.append(layer, neuron)
        self.hidden_layers.append(layer)

    def forward(self, x):
        lenHidLayers = len(self.hidden_layers)
        j = 0
        sum = 0
        if lenHidLayers > 1:
            self.input_layer = x
            for i in range(lenHidLayers):
                #first hidden layer
                if i == 0:
                    for neuron in self.hidden_layers[0]:          
                        neuron.value = neuron.activationFunction(np.sum(self.input_layer))
                else:
                    for neuron in self.hidden_layers[i]:
                        neuron.calculateActivation()
            return self.getLastLayerValues()
        else:
            print("To use the neural network your need at least an input layer, a hidden layer and an output layer")    

    def getLastLayerValues(self):
        lenHidLayers = len(self.hidden_layers)
        output = np.array([])
        for neuron in self.hidden_layers[lenHidLayers - 1]:
            output = np.append(output, neuron.value)
        print("output: ", output)
        return softmax(output)
        

# Sigmoid function
def sigmoid(x, derivada=False):
    if derivada:
        return sigmoid(x)*(1-sigmoid(x)) 
    return 1 / (1 + np.exp(-x))

def softmax(z):
    z_exp = [math.exp(i) for i in z]
    sum_z_exp = sum(z_exp)
    return [round(i / sum_z_exp, 3) for i in z_exp]

x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
#print("np.sum(x): ", np.sum(x))
nn = NeuralNetwork(8)
nn.addLayer(5, sigmoid)
nn.addLayer(15, sigmoid)
nn.addLayer(10, sigmoid)
nn.addLayer(2, sigmoid)
output = nn.forward(x)
print("output: ", output)


