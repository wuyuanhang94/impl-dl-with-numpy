import numpy as np
from common import *
from layers import *
from dataset import load_mnist
from twolayernn import TwoLayerNet

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)

batch_size = 128
train_size = x_train.shape[0]
learning_rate = 1e-3
epoch_num = 2000


for epoch in range(epoch_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    grads = network.gradient(x_batch, t_batch)

    for key in ['W1', 'b1', 'W2', 'b2']:
        # print(grads[key].shape)
        # print(key)
        # print(grads[key].shape)
        # print(network.params[key].shape)
        network.params[key] -= learning_rate * grads[key]

    train_loss = network.loss(x_batch, t_batch)
    train_acc = network.accuracy(x_batch, t_batch)

    if epoch % 20 == 0:
        test_loss = network.loss(x_test, t_test)
        test_acc = network.accuracy(x_test, t_test)
        print("epoch: %4d" % epoch, "train_loss: %6f" % train_loss, "train_acc: %6f" % train_acc,
              "test_loss: %6f" % test_loss, "test_acc: %6f" % test_acc)
    
