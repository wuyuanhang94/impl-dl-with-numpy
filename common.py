import numpy as np
import copy
from layers import ReLU

def softmax(x):
    # x是一维的或者二维的
    # x: [input_size, ], [batch_size * input_size]
    # 除了这种升维 reshape也可以
    if x.ndim == 2:
        # 可用转置避免升维
        maxElem = np.max(x, axis=1)
        x -= np.expand_dims(maxElem, axis=1)
        x = np.exp(x)
        sum = np.sum(x, axis=1)
        return x / np.expand_dims(sum, axis=1)

    maxElem = np.max(x)
    x -= maxElem
    x = np.exp(x)
    return x / np.sum(x)

def crossEntropyLoss(y, t):
    # 这里的loss同样要考虑single case 和 batch case
    if t.ndim == 1:
        y = y.reshape(1, -1)
        t = t.reshape(1, -1)
    
    batch_size = y.shape[0]
    rowIdx = np.arange(batch_size)
    colIdx = np.argmax(t, axis=1)
    return -1/batch_size * np.sum(np.log(y[rowIdx, colIdx] + 1e-7))

if __name__ == "__main__":
    # 本地测试代码
    x = np.array([[-1, 2, -2], [0, .2, 3]])
    r1 = ReLU()
    print("ReLU(x):\n", r1.forward(x))
    print("softmax(x):\n", softmax(x))
    y = np.array([-1, 2, -3, 4, 5])
    print("softmax(y):\n", softmax(y))
