import numpy as np
from common import *
from layers import *
from collections import OrderedDict

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.random.randn(hidden_size)  # (hidden_size, ) 利用numpy广播
        
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.random.randn(output_size)  # (output_size, ) 利用numpy广播

        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['ReLU1'] = ReLU()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])

        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        
        return np.sum(y == t) / x.shape[0]

    def gradient(self, x, t):
        dout = 1
        layers = list(self.layers).reverse()

        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        grads['W1'] = self.layers['Affine1'].dW1
        grads['b1'] = self.layers['Affine1'].db1
        grads['W2'] = self.layers['Affine2'].dW2
        grads['b2'] = self.layers['Affine2'].db2

        return grads


