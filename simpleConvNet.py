import pickle
import numpy as np
from layers import *
from optimizer import *
from common import *
from collections import OrderedDict

class ConvNet:
    # Conv -> ReLU -> Pool -> Affine -> ReLU -> Affine -> Softmax
    def __init__(self, input_dim=(1, 28, 28), 
                 conv_param={'filter num': 50, 'filter size':5, 'pad':0, 'stride':1},
                 hidden_size=100, output_size=50, weight_init_std=0.01, pretrained=False):
        filter_num = conv_param['filter_num']
        filter_size = conv_param['filter_size']
        filter_pad = conv_param['pad']
        filter_stride = conv_param['stride']
        input_size = input_dim[1]
        conv_output_size = (input_size - filter_size + 2*filter_pad) / filter_stride + 1
        pool_output_size = int(filter_num * (conv_output_size/2) * (conv_output_size/2))

        self.params = {'W1':None, 'b1':None,
                       'W2':None, 'b2':None,
                       'W3':None, 'b3':None}
        if not pretrained:
            self.params['W1'] = weight_init_std * np.random.randn(filter_num, input_dim[0], filter_size, filter_size) # Conv params
            self.params['b1'] = weight_init_std * np.random.randn(filter_num)
            self.params['W2'] = weight_init_std * np.random.randn(pool_output_size, hidden_size)
            self.params['b2'] = weight_init_std * np.random.randn(hidden_size)
            self.params['W3'] = weight_init_std * np.random.randn(hidden_size, output_size)
            self.params['b3'] = weight_init_std * np.random.randn(output_size)
        else:
            self.loadParams()

        self.layers = OrderedDict()
        self.layers['Conv'] = Convolution(W=self.params['W1'], b=self.params['b1'])
        self.layers['ReLu1'] = ReLU()
        self.layers['Pooling'] = Pooling(pool_h=2, pool_w=2, stride=2)
        self.layers['Affine1'] = Affine(W=self.params['W2'], b=self.params['b2'])
        self.layers['ReLU2'] = ReLU()
        self.layers['Affine2'] = Affine(W=self.params['W3'], b=self.params['b3'])
        
        if pretrained:
            for i, key in enumerate(['Conv', 'Affine1', 'Affine2']):
                self.layers[key].W = self.params['W' + str(i+1)]
                self.layers[key].b = self.params['b' + str(i+1)]
        
        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        
        return x
    
    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        return 1.0 * np.sum(y == t) / x.shape[0]

    def gradient(self, x, t):
        # 应该限制gradient的调用时序 作为backward 先call forward
        # self.loss(x, t)

        dout = 1 #对应着是loss 变化1 下面相关的变量变化多少
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()

        for layer in layers:
            dout = layer.backward(dout)
        
        grads = {}
        grads['W1'] = self.layers['Conv'].dW
        grads['b1'] = self.layers['Conv'].db
        grads['W2'] = self.layers['Affine1'].dW
        grads['b2'] = self.layers['Affine1'].db
        grads['W3'] = self.layers['Affine2'].dW
        grads['b3'] = self.layers['Affine2'].db

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

        # for i, key in enumerate(['Conv1', 'Affine1', 'Affine2']):
        #     self.layers[key].W = self.params['W' + str(i+1)]
        #     self.layers[key].b = self.params['b' + str(i+1)]
