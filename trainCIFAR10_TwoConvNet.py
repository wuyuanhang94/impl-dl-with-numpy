import numpy as np
from dataset.cifar10 import load_cifar10
from twoConvNet import ConvNet
from trainer import Trainer

(x_train, t_train), (x_test, t_test) = load_cifar10(flatten=False)
# print(x_train.shape)
network = ConvNet(input_dim=(3,32,32), 
                  conv_param1={'filter num': 64, 'filter size':3, 'pad':1, 'stride':1},
                  conv_param2={'filter num': 64, 'filter size':3, 'pad':1, 'stride':1},
                  output_size=10, weight_init_std=0.01, pretrained=False)
                        
trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs=20, mini_batch_size=128,
                  optimizer='AdaGrad', optimizer_param={'lr': 0.001})

trainer.train()
