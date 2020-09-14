import numpy as np
import copy

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

def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_data.shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1

    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return col


def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]

if __name__ == "__main__":
    # 本地测试代码
    print("softmax(x):\n", softmax(x))
    y = np.array([-1, 2, -3, 4, 5])
    print("softmax(y):\n", softmax(y))
