import numpy as np
from dataset.cifar10 import load_cifar10
from simpleConvNet import ConvNet
from trainer import Trainer

(x_train, t_train), (x_test, t_test) = load_cifar10(flatten=False)

network = ConvNet(input_dim=(3,32,32), 
                  conv_param = {'filter_num': 50, 'filter_size': 5, 'pad': 0, 'stride': 1},
                  hidden_size=100, output_size=10, weight_init_std=0.01, pretrained=True)
                        
trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs=10, mini_batch_size=128,
                  optimizer='AdaGrad', optimizer_param={'lr': 0.005})

trainer.train()