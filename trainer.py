import numpy as np
from common import *
from optimizer import *

class Trainer:
    def __init__(self, network, x_train, t_train, x_test, t_test,
                epochs=20, mini_batch_size=100,
                optimizer='SGD', optimizer_param={'lr':0.01}):
        # 1, 网络model
        self.network = network

        # 2, 数据集
        self.x_train = x_train
        self.t_train = t_train
        self.x_test = x_test
        self.t_test = t_test
        self.batch_size = mini_batch_size

        # 3, 训练周期
        self.epochs = epochs

        # 4, 优化方法
        optimizer_dict = {'SGD': SGD(), 'Momentum': Momemtum(), 'AdaGrad': AdaGrad()}
        self.optimizer = optimizer_dict[optimizer]

    def train_step(self):
        batch_mask = np.random.choice(self.x_train.shape[0], self.batch_size)
        x_batch = self.x_train[batch_mask]
        t_batch = self.t_train[batch_mask]

        loss = self.network.loss(x_batch, t_batch)
        acc = self.network.accuracy(x_batch, t_batch)


