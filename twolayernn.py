import numpy as np
from common import *

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.random.randn(hidden_size)  # (hidden_size, ) 利用numpy广播
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.random.randn(output_size)  # (output_size, ) 利用numpy广播

    def forward(self, x):
        a1 = np.dot(x, self.params['W1']) + self.params['b1']  # 这里会广播
        z1 = ReLU(a1)
        a2 = np.dot(z1, self.params['W2']) + self.params['b2']
        y = softmax(a2)
        return y

    def backward(self, dout):
        pass
