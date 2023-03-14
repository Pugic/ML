import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt  # pip install matplotlib

def load_mnist(path, kind='train'):
    labels_path = os.path.join(path, f'{kind}-labels-idx1-ubyte')
    images_path = os.path.join(path, f'{kind}-images-idx3-ubyte')
    with open(labels_path, 'rb') as lbpath:
        lbpath.read(8)
        labels = np.fromfile(lbpath, dtype=np.uint8)
    with open(images_path, 'rb') as imgpath:
        imgpath.read(16)
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
    return images, labels

def batch_generator(images, labels, batch_size):
    num_samples = images.shape[0]
    num_batches = num_samples // batch_size
    for i in range(num_batches):
        start = i * batch_size
        end = (i + 1) * batch_size
        batch_images = images[start:end].reshape(-1, 28, 28, 1)
        batch_labels = labels[start:end]
        yield torch.tensor(batch_images), torch.tensor(batch_labels)

def test_batch():
    path = './MNIST/raw/'
    X_train, y_train = load_mnist(path, kind='train')
    batch_size = 10
    train_generator = batch_generator(X_train, y_train, batch_size)

    for data in train_generator:    
        print(data)
        break

    X, y = data[0][0], data[1][0]
    print(data[1])

    plt.imshow(data[0][4].view(28,28))
    plt.show()

