import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt 
import random


def load_mnist(path, kind='train', shuffle = True):
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
    if shuffle == True:
            permutation = np.random.permutation(len(labels)) # для перемешивания генерируем случайную перестановку 
            images = images[permutation] # используем полученную перестановку, чтобы сохранить соответствие между лейблами и картинками 
            labels = labels[permutation]
    return images, labels


def noise_img(images, n_loc, n_scale):
    noise = np.random.normal(loc = n_loc, scale = 0.1, size = images.shape)
    new_imgs = np.clip(images + noise, 0, 1)
    return new_imgs

def rotate(imgs):
    angle = random.randint(0, 150) 
    for i in range(imgs.shape[0]):
        imgs[i] = np.rot90(imgs[i], int(angle/90))
    return imgs

def batch_generator(images, labels, batch_size):
    num_samples = images.shape[0] #считаем сколько строк в массиве, соответственно сколько картинок всего было получено
    num_batches = num_samples // batch_size # вычисляем сколько получится батчей заданного размера  
    for i in range(num_batches):
        start = i * batch_size
        end = (i + 1) * batch_size
        batch_images = images[start:end].reshape(-1, 28, 28, 1) # считываем batch_size строк массива
        batch_labels = labels[start:end]

        batch_images = noise_img(batch_images, 0.5, 0.1)
        batch_images = rotate(batch_images)

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
    print(data[0][1])
    print(data[1])
    plt.imshow(data[0][4].view(28,28))
    plt.show()

test_batch()



