import numpy as np
from common import softmax
from common import crossEntropyLoss

class ReLU:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = x <= 0
        return np.maximum(0, x)

    def backward(self, dout):
        dout[self.mask] = 0
        return dout

class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.dW = None
        self.db = None
        self.x = None
        self.original_x = None

    def forward(self, x):
        self.original_x = x
        self.x = x
        if self.x.ndim == 1:
            self.x.reshape(1, -1)
        return np.dot(self.x, self.W) + self.b
    
    def backward(self, dout):
        self.dW = np.dot(self.x, dout)
        self.db = np.sum(dout, axis=0)

        dout = np.dot(dout, self.W.T)
        return dout

class SoftmaxWithLoss:
    def __init__(self):
        self.y = None
        self.dy = None
        self.t = None
    
    def forward(self, x, t):
        self.y = softmax(x)
        loss = crossEntropyLoss(self.y, t)
        return loss

    def backward(self, dout=1):
        self.dy = self.y - self.t
        self.dy *= dout
        return self.dy


if __name__ == "__main__":
    x = np.array([[-1, 2, -2], [0, .2, 3]])
    r1 = ReLU()
    print("ReLU(x):\n", r1.forward(x))