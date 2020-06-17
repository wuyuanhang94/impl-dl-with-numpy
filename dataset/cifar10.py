# coding: utf-8
try:
    import urllib.request
except ImportError:
    raise ImportError('You should use Python 3.x')
import os.path
import gzip
import pickle
import os
import numpy as np
import tarfile
from scipy.misc import imsave

url_base = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'

dataset_dir = os.path.dirname(os.path.abspath(__file__))
save_file = dataset_dir + "/cifar10.pkl"

train_num = 50000
test_num = 10000
img_dim = (1, 32, 32)
img_size = 32*32

def download_cifar10():
    file_path = dataset_dir + "/" + "cifar-10-python.tar.gz"
    if os.path.exists(file_path):
        return    
    print("Downloading cifar10...")
    urllib.request.urlretrieve(url_base, file_path)
    print("Done")

def extract():
    file_path = dataset_dir + "/" + "cifar-10-python"
    if os.path.exists(file_path):
        return
    tar_path = dataset_dir + "/" + "cifar-10-python.tar.gz"
    tar = tarfile.open(tar_path)
    names= tar.getnames()
    for name in names:
        tar.extract(name, tar_path.split('.')[0])
    tar.close()

def unpickle():
    dir_path = os.path.join(dataset_dir, 'cifar-10-python/cifar-10-batches-py')
    train_dict = []
    test_dict = {}
    for f in os.listdir(dir_path):
        if 'data_batch' in f:
            with open(os.path.join(dir_path, f), 'rb') as fo:
                dict = pickle.load(fo, encoding='bytes')
                train_dict.append(dict)
        elif 'test_batch' in f:
            with open(os.path.join(dir_path, f), 'rb') as fo:
                test_dict = pickle.load(fo, encoding='bytes')
    return train_dict, test_dict

def init_cifar10():
    download_cifar10()
    extract()
    train_raw, test_raw = unpickle()

    dataset = {}
    dataset['x_train'] = []
    dataset['t_train'] = []
    dataset['x_test'] = []
    dataset['t_test'] = []
    for j, train in enumerate(train_raw):
        for i in range(10000):
            img = np.reshape(train[b'data'][i], (3, 32, 32))
            dataset['x_train'].append(img)
            dataset['t_train'].append(train[b'labels'][i])

    for i in range(0, 10000):
        img = np.reshape(test_raw[b'data'][i], (3, 32, 32))
        dataset['x_test'].append(img)
        dataset['t_test'].append(test_raw[b'labels'][i])
    
    dataset['x_train'] = np.array(dataset['x_train'])
    dataset['t_train'] = np.array(dataset['t_train'])
    dataset['x_test'] = np.array(dataset['x_test'])
    dataset['t_test'] = np.array(dataset['t_test'])

    with open(save_file, 'wb') as f:
        pickle.dump(dataset, f, -1)

def _change_one_hot_label(X):
    T = np.zeros((X.size, 10))
    for idx, row in enumerate(T):
        row[X[idx]] = 1

    return T

def load_cifar10(normalize=True, flatten=True, one_hot_label=True):
    if not os.path.exists(save_file):
        init_cifar10()

    with open(save_file, 'rb') as f:
        dataset = pickle.load(f)

    if normalize:
        for key in ('x_train', 'x_test'):
            dataset[key] = dataset[key].astype(np.float32)
            dataset[key] /= 255.0
    
    if one_hot_label:
        dataset['t_train'] = _change_one_hot_label(dataset['t_train'])
        dataset['t_test'] = _change_one_hot_label(dataset['t_test'])

    if flatten:
        for key in ('x_train', 'x_test'):
            row = dataset[key].shape[0]
            dataset[key] = dataset[key].reshape(-1, 1, 3*32*32)
    
    return (dataset['x_train'], dataset['t_train']), (dataset['x_test'], dataset['t_test'])

load_cifar10()