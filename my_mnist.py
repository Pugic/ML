import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt 
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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

def rotate(img):
    angle = random.randint(0, 150) 
    img = np.rot90(img, int(angle/90))
    return img

def batch_generator(images, labels, batch_size):
    num_samples = images.shape[0] #считаем сколько строк в массиве, соответственно сколько картинок всего было получено
    num_batches = num_samples // batch_size # вычисляем сколько получится батчей заданного размера  
    for i in range(num_batches):
        start = i * batch_size
        end = (i + 1) * batch_size
        #batch_images = images[start:end].reshape(28, 28) # считываем batch_size строк массива
        batch_images = np.array([[noise_img(rotate(img.reshape(28,28)), 0.5, 0.1)] for img in images[start:end]])
        #print(batch_images[0])
        batch_labels = labels[start:end]
        #print(torch.Tensor(batch_images))
        yield torch.Tensor(batch_images), torch.tensor(batch_labels) 

def test_batch():
    path = './MNIST/raw/'
    X_train, y_train = load_mnist(path, kind='train')
    batch_size = 10
    train_generator = batch_generator(X_train, y_train, batch_size)
    #print(train_generator)

    for data in train_generator:    
        print(data)
        break

    X, y = data[0][0], data[1][0]
    #print("fffffff ",X)

    plt.imshow(data[0][4].view(28,28))
    plt.show()

#test_batch()

path = './MNIST/raw/'
X_train, y_train = load_mnist(path, kind='train')
batch_size = 10
train_generator = batch_generator(X_train, y_train, batch_size)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # первый сверточный слой
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        # второй сверточный слой
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        # слой максимального пулинга
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        # слой преобразования данных
        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        # выходной слой
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # применяем первый сверточный слой
        x = self.conv1(x)
        x = nn.functional.relu(x)
        # применяем второй сверточный слой
        x = self.conv2(x)
        x = nn.functional.relu(x)
        # применяем слой максимального пулинга
        x = self.maxpool(x)
        print("size of ", x.shape)
        # преобразуем данные перед подачей на полносвязный слой
        x = x.view(x.size(0), -1)
        print("size of ", x.shape)
        # применяем первый полносвязный слой
        x = self.fc1(x)
        x = nn.functional.relu(x)
        # выходной слой
        x = self.fc2(x)
        return x


net =  Net()

optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
criterion = nn.NLLLoss()

def train(model, optimizer, criterion, train_loader, num_epochs):
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            # обнуляем градиенты
            optimizer.zero_grad()
            
            # передаем данные через нейронную сеть
            outputs = model(inputs)
            # вычисляем функцию потерь
            loss = criterion(outputs, labels)
            # вычисляем градиенты
            loss.backward()
            # обновляем веса
            optimizer.step()
            # суммируем потери
            running_loss += loss.item()
            if i % 2000 == 1999:
                # выводим промежуточные результаты каждые 2000 мини-пакетов
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
    print('Finished Training')

train(net, optimizer, criterion, train_generator, 2)

