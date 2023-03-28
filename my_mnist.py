import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt 
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math

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

    def rotate(self,image):
        angle = random.randint(0, 100) 
        angle=math.radians(angle)                              
        cos=math.cos(angle)
        sin=math.sin(angle)
        height=image.shape[0]                                   
        width=image.shape[1]                                    

        # вычисление новых размеров повернутой картинки 
        new_height  = round(abs(height*cos)+abs(width*sin))+1
        new_width  = round(abs(width*cos)+abs(height*sin))+1

        # создание массива нового размера 
        output=np.zeros((new_height,new_width, 1))

        # центр старой картинки 
        orig_centre_h   = round(((height+1)/2)-1)
        orig_centre_w   = round(((width+1)/2)-1) 

        # центр новой картинки
        new_centre_h= round(((new_height+1)/2)-1)        
        new_centre_w= round(((new_width+1)/2)-1)          

        for i in range(height):
            for j in range(width):
                #коородинаты пикселя относительно центра 
                y=height-1-i-orig_centre_h                   
                x=width-1-j-orig_centre_w                      

                #новые координаты
                new_y=round(-x*sin+y*cos)
                new_x=round(x*cos+y*sin)

                new_y=new_centre_h-new_y
                new_x=new_centre_w-new_x
                 
                if 0 <= new_x < new_width and 0 <= new_y < new_height and new_x>=0 and new_y>=0:
                    output[new_y,new_x]=image[i,j]                         
        output = cv2.resize(output,(28,28))
        
        return output
        

    def batch_generator(self,images, labels, batch_size):
        num_samples = images.shape[0] #считаем сколько строк в массиве, соответственно сколько картинок всего было получено
        num_batches = num_samples // batch_size # вычисляем сколько получится батчей заданного размера  
        for i in range(num_batches):
            start = i * batch_size
            end = (i + 1) * batch_size
            # заполняем массив batch_images, изменяем размер картинок на массив 28*28, применяем вращение и зашумление 
            batch_images = np.array([[self.noise_img(self.rotate(img.reshape(28,28)), 0.3, 0.1)] for img in images[start:end]])
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


def train(model, optimizer, criterion, num_epochs):
    for epoch in range(num_epochs):
        running_loss = 0.0
        loss_sum = 0.0
        train_generator = loader.batch_generator(X_train, y_train, batch_size)
        for i, (inputs, labels) in enumerate(train_generator):
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

def test(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    with torch.no_grad():
        for data, labels in test_loader:
            
            outputs = model(data)
            predicted = torch.argmax(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            true_positives += ((predicted == 1) & (labels == 1)).sum().item()
            false_positives += ((predicted == 1) & (labels == 0)).sum().item()
            false_negatives += ((predicted == 0) & (labels == 1)).sum().item()

    accuracy = correct / total
    recall = true_positives / (true_positives + false_negatives)
    precision = true_positives / (true_positives + false_positives)
    f1 = 2 * ((precision * recall) / (precision + recall))

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"F1 Score: {f1:.4f}")

    return accuracy, recall, precision, f1



if __name__ == "__main__" :
    loader = Loader('./MNIST/raw/')
    X_train, y_train = loader.load_mnist(kind='train')
    batch_size = 10
    train_generator = loader.batch_generator(X_train, y_train, batch_size)
    loader.test_batch(train_generator)
    net =  Net()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    #criterion = nn.NLLLoss()
    criterion = nn.CrossEntropyLoss()
    train(net, optimizer, criterion, 5)
    X_test, y_test = loader.load_mnist(kind='t10k')
    train_generator = loader.batch_generator(X_train, y_train, batch_size)
    test(net, train_generator)


    




