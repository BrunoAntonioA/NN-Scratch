import math
import random

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

#Neural Network Class
class NeuralNetwork:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.layers = []

    # Adding new layer
    def addLayer(self, n, activationFunction):
        layer = []
        # Adding n neurons
        for i in range(n):
            neuron = Neuron([], [], 0, activationFunction)
            layer.append(neuron)
            # If there is more than an input layer
            if len(self.layers) > 1:
                # l -> length of the previous layer
                l = len(self.layers)
                # m -> amount of neurons of the previous layer
                m = len(self.layers[ l-1 ])
                # Referencing previous layer to the i neuron of the new layer
                neuron.neurons = self.layers[ l-1 ]
                # Initialize weights of i neuron of the new layer
                for j in range(m):
                    neuron.weights.append(random.randint(0, 20))
        # Appending the new layer to the NN
        self.layers.append(layer)

# Sigmoid function
def sigmoidFunction(x):
    return ( 1 / (1 + math.exp(-x) ) )

nn = NeuralNetwork(9)
nn.addLayer(10, sigmoidFunction)
print("nn: ", nn.layers)



