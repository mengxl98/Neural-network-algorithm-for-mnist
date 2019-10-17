import os
import struct
import numpy as np

def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,'%s-labels.idx1-ubyte' % kind)
    images_path = os.path.join(path,'%s-images.idx3-ubyte' % kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath,dtype=np.uint8).reshape(len(labels), 784)

    return images, labels

def _read(image,label):
    
    with gzip.open(label) as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        label = np.fromstring(flbl.read(), dtype=np.int8)
    with gzip.open(image, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        image = np.fromstring(fimg.read(), dtype=np.uint8).reshape(len(label), rows, cols)
    return image,label

def get_data():
    
    train_img,train_label = _read(
            'train-images-idx3-ubyte.gz', 
            'train-labels-idx1-ubyte.gz')

    
    test_img,test_label = _read(
            't10k-images-idx3-ubyte.gz', 
            't10k-labels-idx1-ubyte.gz')

    train_img_normlization = np.reshape(np.round(train_img/255),(60000,-1))
    test_img_normlization = np.reshape(np.round(test_img/255),(10000,-1))

    return [train_img_normlization,train_label,test_img_normlization,test_label]

