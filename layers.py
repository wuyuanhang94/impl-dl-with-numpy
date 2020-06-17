import numpy as np

class ReLU:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = x <= 0
        return np.maximum(0, x)

    def backward(self, dout):
        dout[self.mask] = 0
        return dout