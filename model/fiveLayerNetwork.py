import numpy as np
import os
import sys
sys.path.append('..')
from utils.common import *
from utils.layers import *
from collections import OrderedDict

class FiveLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.random.randn(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, hidden_size)
        self.params['b2'] = np.random.randn(hidden_size)
        self.params['W3'] = weight_init_std * np.random.randn(hidden_size, 50)
        self.params['b3'] = np.random.randn(50)
        self.params['W4'] = weight_init_std * np.random.randn(50, 50)
        self.params['b4'] = np.random.randn(50)
        self.params['W5'] = weight_init_std * np.random.randn(50, output_size)
        self.params['b5'] = np.random.randn(output_size)

        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['ReLU1'] = ReLU()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.layers['ReLU2'] = ReLU()
        self.layers['Affine3'] = Affine(self.params['W3'], self.params['b3'])
        self.layers['ReLU3'] = ReLU()
        self.layers['Affine4'] = Affine(self.params['W4'], self.params['b4'])
        self.layers['ReLU4'] = ReLU()
        self.layers['Affine5'] = Affine(self.params['W5'], self.params['b5'])

        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
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
        dout = self.lastLayer.backward()

        layers = list(self.layers.values())
        layers.reverse() # 注意这个reverse是in-place的 返回值为None

        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db
        grads['W3'] = self.layers['Affine3'].dW
        grads['b3'] = self.layers['Affine3'].db
        grads['W4'] = self.layers['Affine4'].dW
        grads['b4'] = self.layers['Affine4'].db
        grads['W5'] = self.layers['Affine5'].dW
        grads['b5'] = self.layers['Affine5'].db

        return grads
