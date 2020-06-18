import numpy as np
from common import *
from layers import *
from dataset.cifar10 import load_cifar10
from refactTLNN import TwoLayerNet
from optimizer import *

(x_train, t_train), (x_test, t_test) = load_cifar10()

network = TwoLayerNet(input_size=3*32*32, hidden_size=100, output_size=10)

# print(x_train.shape)
# print(t_train.shape)
# print(x_test.shape)
# print(t_test.shape)

batch_size = 256
train_size = x_train.shape[0]
learning_rate = 1e-4
epoch_num = int(1e5)
# 这个网络层次太浅 epoch_num到20000 精度已经上不去了 50%

# optimizer = SGD(lr=learning_rate)
optimizer = Momemtum(lr=learning_rate)

for epoch in range(epoch_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    train_loss = network.loss(x_batch, t_batch)
    train_acc = network.accuracy(x_batch, t_batch)
    grads = network.gradient(x_batch, t_batch)

    optimizer.update(network.params, grads)

    if epoch % 50 == 0:
        test_loss = network.loss(x_test, t_test)
        test_acc = network.accuracy(x_test, t_test)
        print("epoch: %4d" % epoch, "train_loss: %6f" % train_loss, "train_acc: %6f" % train_acc,
              "test_loss: %6f" % test_loss, "test_acc: %6f" % test_acc)
    
