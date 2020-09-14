import numpy as np
from common import *
from layers import *
from dataset.cifar10 import load_cifar10
from dataset.mnist import load_mnist
# from refactTLNN import TwoLayerNet
from fiveLayerNetwork import FiveLayerNet
from optimizer import *

(x_train, t_train), (x_test, t_test) = load_cifar10()
network = FiveLayerNet(input_size=3*32*32, hidden_size=500, output_size=10)

batch_size = 512
train_size = x_train.shape[0]
learning_rate = 1e-4
epoch_num = 80000
# 20000 55% acc
# 5层过拟合太严重了 52% max
# 也就调调learning_rate, batch_size

optimizer = SGD(lr=learning_rate)

for epoch in range(epoch_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    train_loss = network.loss(x_batch, t_batch)
    train_acc = network.accuracy(x_batch, t_batch)
    grads = network.gradient(x_batch, t_batch)

    optimizer.update(network.params, grads)

    if epoch % 100 == 0:
        test_loss = network.loss(x_test, t_test)
        test_acc = network.accuracy(x_test, t_test)
        print("epoch: %4d" % epoch, "train_loss: %6f" % train_loss, "train_acc: %6f" % train_acc,
              "test_loss: %6f" % test_loss, "test_acc: %6f" % test_acc)
    
