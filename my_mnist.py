import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt 
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Loader():
    def __init__(self, path):
        self.path = path # путь к данным

    def load_mnist(self, kind='train', shuffle = True):
        labels_path = os.path.join(self.path, f'{kind}-labels-idx1-ubyte') 
        images_path = os.path.join(self.path, f'{kind}-images-idx3-ubyte')
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


    def noise_img(self,images, n_loc, n_scale):
        noise = np.random.normal(loc = n_loc, scale = 0.1, size = images.shape)
        new_imgs = np.clip(images + noise, 0, 1)
        return new_imgs

    def rotate(self,img):
        angle = random.randint(0, 150) 
        img = np.rot90(img, int(angle/90))
        
        return img

    def batch_generator(self,images, labels, batch_size):
        num_samples = images.shape[0] #считаем сколько строк в массиве, соответственно сколько картинок всего было получено
        num_batches = num_samples // batch_size # вычисляем сколько получится батчей заданного размера  
        for i in range(num_batches):
            start = i * batch_size
            end = (i + 1) * batch_size
            # заполняем массив batch_images, изменяем размер картинок на массив 28*28, применяем вращение и зашумление 
            batch_images = np.array([[self.noise_img(self.rotate(img.reshape(28,28)), 0.5, 0.1)] for img in images[start:end]])
            batch_labels = labels[start:end]
            yield torch.Tensor(batch_images), torch.tensor(batch_labels) 

    # функция для проверки содержимого батчей 
    def test_batch(self, train_generator):
        #print(train_generator)

        for data in train_generator:    
            #print(data)
            break

        X, y = data[0][0], data[1][0]

        plt.imshow(data[0][4].view(28,28))
        plt.show()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # первый сверточный слой
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        # второй сверточный слой
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        # слой максимального пулинга
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        # слой преобразования данных
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        # выходной слой
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        #применяем первый сверточный слой
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.maxpool(x)
        #применяем второй сверточный слой
        x = self.conv2(x)
        x = nn.functional.relu(x)
        #применяем слой максимального пулинга
        x = self.maxpool(x)
        #преобразуем данные перед подачей на полносвязный слой
        x = x.view(x.size(0), -1)
        #применяем первый полносвязный слой
        x = self.fc1(x)
        x = nn.functional.relu(x)
        #выходной слой
        x = self.fc2(x)
        x = nn.functional.softmax(x, dim = 1)
        return x


def train(model, optimizer, criterion, train_loader, num_epochs):
    for epoch in range(num_epochs):
        running_loss = 0.0
        loss_sum = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            #обнуляем градиенты
            optimizer.zero_grad()
            
            #передаем данные через нейронную сеть
            outputs = model(inputs)
            #вычисляем функцию потерь
            loss = criterion(outputs, labels)
            #вычисляем градиенты
            loss.backward()
            #обновляем веса
            optimizer.step()
            #суммируем потери
            running_loss += loss.item()
            loss_sum += running_loss
            if i % 1000 == 999:
                #выводим промежуточные результаты каждые 2000 мини-пакетов
                print(running_loss)
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 1000))
                running_loss = 0.0
        print("loss for ", epoch," epoch is ", loss_sum / 6000)
    print('Finished Training')

if __name__ == "__main__" :
    loader = Loader('./MNIST/raw/')
    X_train, y_train = loader.load_mnist(kind='train')
    batch_size = 10
    train_generator = loader.batch_generator(X_train, y_train, batch_size)
    #loader.test_batch(train_generator)
    net =  Net()
    optimizer = optim.SGD(net.parameters(), lr=0.05, momentum=0.9)
    criterion = nn.NLLLoss()
    train(net, optimizer, criterion, train_generator, 10)




