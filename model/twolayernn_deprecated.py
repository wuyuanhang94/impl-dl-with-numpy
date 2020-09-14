import numpy as np
import os
import sys
sys.path.append('..')
from utils.common import *
from utils.layers import *

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}
        # weight_init_std 不能太大 会过拟合 初始值设置参考Xavier or Kaiming
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        # NN 有随机初始化的weights 已经打破对称性 bias可初始化全零
        self.params['b1'] = np.random.randn(hidden_size)  # (hidden_size, ) 利用numpy广播
        self.params['ReLU1'] = ReLU()
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.random.randn(output_size)  # (output_size, ) 利用numpy广播

    def predict(self, x):
        a1 = np.dot(x, self.params['W1']) + self.params['b1']  # 这里会广播 backward要小心
        z1 = self.params['ReLU1'].forward(a1)
        a2 = np.dot(z1, self.params['W2']) + self.params['b2']
        y = softmax(a2)
        return y

    def loss(self, x, t):
        y = self.predict(x)
        return crossEntropyLoss(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        
        return np.sum(y == t) / x.shape[0]

    def gradient(self, x, t):
        # todo：这里实现不高效
        # 详见refactTLNN.py 
        a1 = np.dot(x, self.params['W1']) + self.params['b1']
        z1 = self.params['ReLU1'].forward(a1)
        a2 = np.dot(z1, self.params['W2']) + self.params['b2']
        y = softmax(a2)

        dy = y - t #注意这里的batch形式
        dW2 = np.dot(z1.T, dy)
        # 这里要注意db2的计算并不直接是dy
        # 因为Y = X×W + b 对广播 forward 要注意 backward也要特殊处理
        db2 = np.sum(dy, axis=0)

        dz1 = np.dot(dy, self.params['W2'].T)
        # 这里ReLU的backward做不动，因为forward时mask没有保存
        # 同理其他layer也需要单独封装 保存中间结果 才能backward
        da1 = self.params['ReLU1'].backward(dz1)

        dW1 = np.dot(x.T, da1)
        db1 = np.sum(da1, axis=0)

        grads = {}
        grads['W1'] = dW1
        grads['b1'] = db1
        grads['W2'] = dW2
        grads['b2'] = db2

        return grads
