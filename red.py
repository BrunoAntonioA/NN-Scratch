import math
import numpy as np
import random
import loader as loader

#Neuron Class
class Neuron:
    def __init__(self, previousLayer, previousLayerWeights, output, activationFunction):
        self.output = output
        self.delta = 0
        self.activationFunction = activationFunction
        self.previousLayer = previousLayer
        self.previousLayerWeights = previousLayerWeights
        self.nextLayer = []
    
    def calculateActivation(self):
        sum = 0
        lenPrevLayer = len(self.previousLayer)
        for i in range(lenPrevLayer):
            sum = sum + self.previousLayer[i].output * self.previousLayerWeights[i]
        self.output = self.activationFunction(sum)
        return self.output

    def calculateDelta(self, y, i, lastLayer):
        if lastLayer:
            self.delta = (self.activationFunction(self.output) - y) * self.activationFunction(self.output, True)
        else:
            sum = 0
            for neuron in self.nextLayer:
                sum = sum + neuron.delta * neuron.previousLayerWeights[i]
            self.delta = sum + self.activationFunction(self.output, True)
        return self.delta
    
    def updateWeight(self, t, i):
        self.previousLayerWeights[i] = self.previousLayerWeights[i] - (t * self.delta * self.activationFunction(self.previousLayer[i].output))
        
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
                        sum = np.sum(self.input_layer)  
                        neuron.output = neuron.activationFunction(sum)
                else:
                    for neuron in self.hidden_layers[i]:
                        neuron.output = neuron.calculateActivation()
            return self.getLastLayerValues()
        else:
            print("To use the neural network your need at least an input layer, a hidden layer and an output layer")    

    def getLastLayerValues(self):
        lenHidLayers = len(self.hidden_layers)
        output = np.array([])
        for neuron in self.hidden_layers[lenHidLayers - 1]:
            output = np.append(output, neuron.output)
        return softmax(output)

    def updateNextLayer(self):
        lenHidLayers = len(self.hidden_layers)
        for i in range(lenHidLayers - 1):
            for neuron in self.hidden_layers[i]:
                    neuron.nextLayer = self.hidden_layers[i + 1]   

    def backPropagation(self, y):
        lenHidLayers = len(self.hidden_layers)
        for i in range(lenHidLayers-1, -1, -1):
            lenHidLay = len(self.hidden_layers[i])
            for j in range(lenHidLay):
                if i == lenHidLayers - 1:
                    #if predict value is the j neuron, ex: y = 4, neuron = 4
                    if j == y:    
                        self.hidden_layers[i][j].calculateDelta(y, j, True)
                    else: 
                        self.hidden_layers[i][j].calculateDelta(0, j, True)
                else:
                    self.hidden_layers[i][j].calculateDelta(0, j, False)
            self.updateWeights(i)
    
    # in this case 0.4 is Tlearn factor
    def updateWeights(self, i):
        for neuron in self.hidden_layers[i]:
            if i != 0:
                for j in range(neuron.previousLayer.shape[0]):
                    neuron.updateWeight(0.4, j)
            else:
                break

    def training(self, x, y):
        for i in range(x.shape[0]):
            self.forward(x[i])
            self.updateNextLayer()
            self.backPropagation(y[i])

    def predict(self, x):
        return self.forward(x)

    def test(self, x, y):
        total = x.shape[0]
        success = 0
        for i in range(x.shape[0]):
            predict = self.predict(x[i])
            if predict.index(max(predict)) == y[i]:
                success = success + 1
        return success / total

# Sigmoid function
def sigmoid(x, derivada=False):
    if derivada:
        return sigmoid(x)*(1-sigmoid(x)) 
    return 1 / (1 + np.exp(-x))

def softmax(z):
    z_exp = [math.exp(i) for i in z]
    sum_z_exp = sum(z_exp)
    return [round(i / sum_z_exp, 3) for i in z_exp]


datos = loader.get_datos('./Fashion-DataSet-master/')
dt1 = datos[0]
labels1 = datos[2]
dt2 = datos[1]
labels2 = datos[3]

x = dt1[:2500]
y = labels1[:2500]

nn = NeuralNetwork(784)
nn.addLayer(64, sigmoid)
nn.addLayer(32, sigmoid)
nn.addLayer(32, sigmoid)
nn.addLayer(16, sigmoid)
nn.addLayer(10, sigmoid)

nn.training(x, y)

xtest = dt2[:10]
ytest = labels2[:10]

acc = nn.test(xtest, ytest)
print("acc: ", acc)

xt = dt2[1]
yt = labels2[1]
prec = nn.predict(xt)
print("prec: ", prec)
print("prec: ", prec.index(max(prec)))
print("yt: ", yt)


