import numpy as np

class SGD:
    def __init__(self, lr=1e-2):
        self.learning_rate = lr

    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.learning_rate * grads[key]

class Momemtum:
    def __init__(self, lr=1e-2, momentum=0.9):
        self.momentum = momentum
        self.V = None
        self.learning_rate = lr

    def update(self, params, grads):
        
        for key in params.keys():
            if self.V is None:
                self.V = -self.learning_rate * grads[key]
            else:
                self.V = self.momentum * self.V - self.learning_rate * grads[key]
            params[key] += self.V
