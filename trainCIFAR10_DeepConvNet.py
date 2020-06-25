import numpy as np
from dataset.cifar10 import load_cifar10
from deepConvNet import DeepConvNet
from trainer import Trainer

(x_train, t_train), (x_test, t_test) = load_cifar10(flatten=False)

network = DeepConvNet()  
trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs=20, mini_batch_size=128,
                  optimizer='AdaGrad', optimizer_param={'lr':0.001})
trainer.train()
