import numpy as np
import pickle
from collections import OrderedDict
import os
import sys
sys.path.append('..')
from utils.common import *
from utils.layers import *
class DeepConvNet:
    # Conv -> ReLU -> Conv -> ReLU -> Pool ->
    # Conv -> ReLU -> Conv -> ReLU -> Pool ->
    # Conv -> ReLU -> Conv -> ReLU -> Pool ->
    # Affine -> ReLU -> Dropout -> Affine -> Dropout -> Softmax
    # batch_size 不用单独考虑
    # (3, 32, 32) -> 
    # (16, 32, 32) -> (16, 32, 32) -> (16, 16, 16)
    # (32, 16, 16) -> (32, 16, 16) -> (32, 8, 8)
    # (64, 8, 8) -> (64, 8, 8) -> (64, 4, 4) -> 64*4*4(flatten)
    # (64*4*4, hidden_size) -> (hidden_size, output_size)
    def __init__(self, input_dim=(3, 32, 32),
                 conv_param_1 = {'filter_num':16, 'filter_size':3, 'pad':1, 'stride':1},
                 conv_param_2 = {'filter_num':16, 'filter_size':3, 'pad':1, 'stride':1},
                 conv_param_3 = {'filter_num':32, 'filter_size':3, 'pad':1, 'stride':1},
                 conv_param_4 = {'filter_num':32, 'filter_size':3, 'pad':1, 'stride':1},
                 conv_param_5 = {'filter_num':64, 'filter_size':3, 'pad':1, 'stride':1},
                 conv_param_6 = {'filter_num':64, 'filter_size':3, 'pad':1, 'stride':1},
                 hidden_size=50, output_size=10):
        pre_node_nums = np.array([1*3*3, 16*3*3, 16*3*3, 32*3*3, 32*3*3, 64*3*3, 64*3*3, hidden_size])
        weight_init_scales = np.sqrt(2.0 / pre_node_nums)

        self.params = {}
        pre_channel_num = input_dim[0] # 记录上一层的输出个数 初始化为input channel
        for i, conv_param in enumerate([conv_param_1, conv_param_2, conv_param_3, conv_param_4, conv_param_5, conv_param_6]):
            self.params['W' + str(i+1)] = weight_init_scales[i] * \
                np.random.randn(conv_param['filter_num'], pre_channel_num, conv_param['filter_size'], conv_param['filter_size'])
            self.params['b' + str(i+1)] = np.zeros(conv_param['filter_num'])
            pre_channel_num = conv_param['filter_num']
        
        self.params['W7'] = weight_init_scales[6] * np.random.randn(conv_param_6['filter_num'] * (input_dim[1]//8) * (input_dim[2]//8), hidden_size)
        self.params['b7'] = np.zeros(hidden_size)
        self.params['W8'] = weight_init_scales[7] * np.random.randn(hidden_size, output_size)
        self.params['b8'] = np.zeros(output_size)

        self.layers = []
        # 所以他们才有block 相同结构组成的block
        self.layers.append(Convolution(self.params['W1'], self.params['b1'], conv_param_1['stride'], conv_param_1['pad']))
        self.layers.append(ReLU())
        self.layers.append(Convolution(self.params['W2'], self.params['b2'], conv_param_2['stride'], conv_param_2['pad']))
        self.layers.append(ReLU())
        self.layers.append(Pooling(pool_h=2, pool_w=2, stride=2))
        self.layers.append(Convolution(self.params['W3'], self.params['b3'], conv_param_3['stride'], conv_param_3['pad']))
        self.layers.append(ReLU())
        self.layers.append(Convolution(self.params['W4'], self.params['b4'], conv_param_4['stride'], conv_param_4['pad']))
        self.layers.append(ReLU())
        self.layers.append(Pooling(pool_h=2, pool_w=2, stride=2))
        self.layers.append(Convolution(self.params['W5'], self.params['b5'], conv_param_5['stride'], conv_param_5['pad']))
        self.layers.append(ReLU())
        self.layers.append(Convolution(self.params['W6'], self.params['b6'], conv_param_6['stride'], conv_param_6['pad']))
        self.layers.append(ReLU())
        self.layers.append(Pooling(pool_h=2, pool_w=2, stride=2))

        self.layers.append(Affine(self.params['W7'], self.params['b7']))
        self.layers.append(ReLU())
        # 先不实现Dropout
        # self.layers.append(Dropout(0.5))
        self.layers.append(Affine(self.params['W8'], self.params['b8']))
        # self.layers.append(Dropout(0.5))
        
        self.output = None
        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x, train_flg=False):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def loss(self, x, t):
        y = self.predict(x)
        self.output = y
        return self.lastLayer.forward(y, t)

    def accuracy(self, x, t):
        # loss->accuracy 以减少不必要的前向过程
        y = np.argmax(self.output, axis=1)
        t = np.argmax(t, axis=1)
        return 1.0 * np.sum(y == t) / x.shape[0]

    def gradient(self, x, t):
        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = self.layers.copy()
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        for i, layer_idx in enumerate([0, 2, 5, 7, 10, 12, 15, 17]):
            grads['W' + str(i+1)] = self.layers[layer_idx].dW
            grads['b' + str(i+1)] = self.layers[layer_idx].db

        return grads

    def saveParams(self):
        params = {}
        for key in self.params.keys():
            params[key] = self.params[key]
        with open('params.pkl', 'wb') as f:
            pickle.dump(params, f)

    def loadParams(self):
        params = None
        with open("params.pkl", 'rb') as f:
            params = pickle.load(f)
        for key in self.params.keys():
            self.params[key] = params[key]