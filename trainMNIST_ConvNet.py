import numpy as np
from dataset.mnist import load_mnist
from simpleConvNet import ConvNet
from trainer import Trainer

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)

network = ConvNet(input_dim=(1,28,28), 
                  conv_param = {'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                  hidden_size=100, output_size=10, weight_init_std=0.01)
                        
trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs=5, mini_batch_size=100,
                  optimizer='AdaGrad', optimizer_param={'lr': 0.001})

trainer.train()