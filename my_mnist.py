import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt  # pip install matplotlib

def load_mnist(path, kind='train'):
    labels_path = os.path.join(path, f'{kind}-labels-idx1-ubyte') 
    images_path = os.path.join(path, f'{kind}-images-idx3-ubyte')
    with open(labels_path, 'rb') as lbpath:
        lbpath.read(8) # чтение служебной информации
        labels = np.fromfile(lbpath, dtype=np.uint8)
    with open(images_path, 'rb') as imgpath:
        imgpath.read(16) # чтение служебной информации
        # считываем из памяти файл с картинками используя тип данных uint8 ( целые числа от 0 до 255), формируем двумерный массив 
        # в котором строк столько же сколько было считано lables, столбцов - 784, нормализуем данные делением на 255
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784) / 255 
    return images, labels



def batch_generator(images, labels, batch_size, shuffle = True):
    num_samples = images.shape[0] #считаем сколько строк в массиве, соответственно сколько картинок всего было получено
    num_batches = num_samples // batch_size # вычисляем сколько получится батчей заданного размера  
    for i in range(num_batches):
        start = i * batch_size
        end = (i + 1) * batch_size
        batch_images = images[start:end].reshape(-1, 28, 28, 1) # считываем batch_size строк массива
        batch_labels = labels[start:end]
        if shuffle == True:
            permutation = np.random.permutation(batch_size) # для перемешивания генерируем случайную перестановку 
            batch_images = batch_images[permutation] # используем полученную перестановку, чтобы сохранить соответствие между лейблами и картинками 
            batch_labels = batch_labels[permutation]
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
    print(data[0][1])
    plt.imshow(data[0][4].view(28,28))
    plt.show()

test_batch()


