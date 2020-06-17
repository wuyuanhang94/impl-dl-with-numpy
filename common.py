import numpy as np

def ReLU(x):
    return np.maximum(0, x)

def softmax(x):
    # x是一维的或者二维的
    # x: [input_size, ], [batch_size * input_size]
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

if __name__ == "__main__":
    # 本地测试代码
    x = np.array([[-1, 2, -2], [0, .2, 3]])
    print("ReLU(x):\n", ReLU(x))
    print("softmax(x):\n", softmax(x))
    y = np.array([-1, 2, -3, 4, 5])
    print("softmax(y):\n", softmax(y))
