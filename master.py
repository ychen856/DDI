#reference
from torch.utils.data import random_split
from torchvision.datasets import CIFAR10
from torchvision.datasets import ImageNet
import torchvision.transforms as transforms
import torch.nn as nn
import torch
import sys
import socket
import pickle
import torch.utils.data as data_utils
import threading
import numpy as np
import os
import torchvision.datasets as datasets
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"
correctW1 = 0
correctW2 = 0
correctW3 = 0
correctW4 = 0

def worker_process(testloader, socket, worker):
    correct = 0
    iteration = 0
    for (inputs, labels) in testloader:
        iteration = iteration + 1
        inputs = inputs.cuda()
        labels = labels.cuda()

        newx = pickle.dumps(inputs)
        newxlen = len(newx)
        newxlen = newxlen.to_bytes(4, byteorder='big')
        socket.sendall(newxlen)
        socket.sendall(newx)

        data = socket.recv(1024)
        data = pickle.loads(data)
        if data == labels:
            correct = correct + 1

    inputs = 'end'
    newx = pickle.dumps(inputs)
    newxlen = len(newx)
    newxlen = newxlen.to_bytes(4, byteorder='big')
    socket.sendall(newxlen)
    socket.sendall(newx)

    if worker == 1:
        global correctW1
        correctW1 = correct
    elif worker == 2:
        global correctW2
        correctW2 = correct
    elif worker == 3:
        global correctW3
        correctW3 = correct
    elif worker == 4:
        global correctW4
        correctW4 = correct

def main():
    imageAmount = int(sys.argv[1])
    # creat connection
    SocketToW1 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    SocketToW1.bind(('10.7.48.14', 5038))
    #SocketToW1.bind(('10.7.48.142', 5038))
    SocketToW1.listen(1)

    print("wait W1")
    socktow1, w1address = SocketToW1.accept()
    print("con W1")

    SocketToW2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    SocketToW2.bind(('10.7.48.14', 5080))
    #SocketToW2.bind(('10.7.48.142', 5080))
    SocketToW2.listen(1)

    print("wait W2")
    socktow2, w2address = SocketToW2.accept()
    print("con W2")

    SocketToW3 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    SocketToW3.bind(('10.7.48.14', 5039))
    #SocketToW3.bind(('10.7.48.142', 5039))
    SocketToW3.listen(1)

    print("wait W3")
    socktow3, w3address = SocketToW3.accept()
    print("con W3")

    # SocketToW4 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # SocketToW4.bind(('10.7.48.14', 5070))
    # SocketToW4.bind(('10.7.48.142', 5070))
    # SocketToW4.listen(1)

    # print("wait W4")
    # socktow4, w4address = SocketToW4.accept()
    # print("con W4")
    ###################################
    # data set
    ###################################
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

    #trainset = CIFAR10("data/cifar10", transform=transform_train, train=True, download=True)
    #trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True, num_workers=1)

    #testset = CIFAR10("data/cifar10", transform=transform_test, train=False, download=True)
    testset = datasets.ImageFolder(
         #    "data/imagenet/train",
         "/home/train",
            transforms.Compose([
                transforms.Resize(80),
                transforms.RandomCrop(60, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
            ]))
    indices = torch.arange(imageAmount)
    while len(testset) < imageAmount:
        testset = torch.utils.data.ConcatDataset([testset, testset])
    testset_sliced = data_utils.Subset(testset, indices)

    #testset_slicedW1, testset_slicedW2, testset_slicedW3, testset_slicedW4 = random_split(testset_sliced, [int(imageAmount / 4),
    #                                                                                                       int(imageAmount / 4),
    #                                                                                                       int(imageAmount / 4),
    #                                                                                                       int(imageAmount / 4)])
    testset_slicedW1, testset_slicedW2, testset_slicedW3 = random_split(testset_sliced, [int(imageAmount / 3),
                                                                                                          int(imageAmount / 3),
                                                                                                          int(imageAmount / 3)])

    testloaderW1 = torch.utils.data.DataLoader(testset_slicedW1, batch_size=1, shuffle=False, num_workers=1)
    testloaderW2 = torch.utils.data.DataLoader(testset_slicedW2, batch_size=1, shuffle=False, num_workers=1)
    testloaderW3 = torch.utils.data.DataLoader(testset_slicedW3, batch_size=1, shuffle=False, num_workers=1)
    #testloaderW4 = torch.utils.data.DataLoader(testset_slicedW4, batch_size=1, shuffle=False, num_workers=1)

    ###################################
    # do test
    ###################################
    global correctW1
    global correctW2
    global correctW3
    global correctW4

    t1 = threading.Thread(target = worker_process, args=(testloaderW1, socktow1, 1))
    t2 = threading.Thread(target = worker_process, args=(testloaderW2, socktow2, 2))
    t3 = threading.Thread(target = worker_process, args=(testloaderW3, socktow3, 3))
    #t4 = threading.Thread(target = worker_process, args=(testloaderW4, socktow4, 4))

    t1.start()
    t2.start()
    t3.start()
    #t4.start()

    t1.join()
    t2.join()
    t3.join()
    #t4.join()

    print("correct w1", correctW1)
    print("correct w2", correctW2)
    print("correct w3", correctW3)
    #print("correct w4", correctW4)


    print('correct: ', correctW1 + correctW2 + correctW3 + correctW4)
    print('correct ratio: ', (correctW1 + correctW2 + correctW3 + correctW4) / imageAmount)

    SocketToW1.close()
    SocketToW2.close()
    SocketToW3.close()

if __name__ == '__main__':
    main()