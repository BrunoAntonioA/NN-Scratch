import numpy as np

def sigmoid(x, derivada=False):
    if derivada:
        return sigmoid(x)*(1-sigmoid(x)) 
    return 1 / (1 + np.exp(-x))

print( "sigmoid(20): ", sigmoid(20, True))

