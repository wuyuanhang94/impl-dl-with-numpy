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

        # 5, 保存模型参数
        self.best_train_loss = 1e7

    def train_step(self, epoch):
        iterations = self.x_train.shape[0] // self.batch_size
        for i in range(iterations):
            batch_mask = np.random.choice(self.x_train.shape[0], self.batch_size)
            x_batch = self.x_train[batch_mask]
            t_batch = self.t_train[batch_mask]

            train_loss = self.network.loss(x_batch, t_batch)
            train_acc = self.network.accuracy(x_batch, t_batch)
            # print("train_loss:", train_loss, "train_acc", train_acc)

            if train_loss < self.best_train_loss:
                self.best_train_loss = train_loss
                self.network.saveParams()

            grads = self.network.gradient(x_batch, t_batch)
            self.optimizer.update(self.network.params, grads)

            if i % 5 == 4:
                test_batch_mask = np.random.choice(self.x_test.shape[0], 128)
                x_test_batch = self.x_test[test_batch_mask]
                t_test_batch = self.t_test[test_batch_mask]
                test_loss = self.network.loss(x_test_batch, t_test_batch)
                test_acc = self.network.accuracy(x_test_batch, t_test_batch)
                print("epoch:%2d" % epoch, "iteration:%4d" % i, 
                      "train_loss:%6f" % train_loss, "train_acc:%6f" % train_acc,
                      "test_loss:%6f" % test_loss, "test_acc:%6f" % test_acc)

    def train(self):
        for epoch in range(self.epochs):
            self.train_step(epoch)


