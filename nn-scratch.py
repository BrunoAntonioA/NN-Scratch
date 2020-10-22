import numpy as np

#Neuron Class
class Neuron:
    def __init__(self, neurons, weights, value, activationFunction):
        self.neurons = neurons
        self.weights = weights
        self.value = value
        self.activationFunction = activationFunction

    def calculateOutput(self):
        sum = 0
        for i in range(len(self.neurons)):
            sum += self.neurons[i] * self.weights[i]
        return self.activationFunction(sum)

def sigmoid(x, derivada=False):
    if derivada:
        return sigmoid(x)*(1-sigmoid(x)) 
    return 1 / (1 + np.exp(-x))

print( "sigmoid(20): ", sigmoid(20, True))

