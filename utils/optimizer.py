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
        self.V = {}
        # 这里的V是个dict 每个key即是w，b 有自己的独特的历史记录
        self.learning_rate = lr

    def update(self, params, grads):
        for key in params.keys():
            if key not in self.V.keys():
                self.V[key] = -self.learning_rate * grads[key]
            else:
                self.V[key] = self.momentum * self.V[key] - self.learning_rate * grads[key]
            params[key] += self.V[key]

class AdaGrad:
    def __init__(self, lr=1e-2):
        self.learning_rate = lr
        self.h = {}

    def update(self, params, grads):
        for key in params.keys():
            if key not in self.h:
                self.h[key] = grads[key] * grads[key]
            else:
                self.h[key] += grads[key] * grads[key]
            delta = 1e-7 # 防止除零
            params[key] -= self.learning_rate * 1/(np.sqrt(self.h[key]) + delta) * grads[key]