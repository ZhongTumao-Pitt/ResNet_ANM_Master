import scipy.io as sio
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math
import time
import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

import argparse

import scipy.io as sio
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math
import time
import hdf5storage
import os

import datetime
import random
import timeit

from tqdm import tqdm
from models.model import *

parser = argparse.ArgumentParser(description='Resnet with anti-noise modules')
parser.add_argument('--model', type=int)
parser.add_argument('--lowSNR', type=int)
parser.add_argument('--upSNR', type=int)

argss = parser.parse_args()



os.environ["CUDA_VISIBLE_DEVICES"] = "0"
batch_size = 8
num_epochs = 100

SNR = np.arange(argss.lowSNR, argss.upSNR+1)
SNR = SNR.astype(int)
best_acc = np.zeros(len(SNR))
train_time = np.zeros(len(SNR))
#task 1: cwru data
#task 2: thu data
file_dir = './data/CWRUDATA/'
# 10 for task 1; 9 for task 2
numclasses = 10
# create a unique folder for saving results
nowTime = datetime.datetime.now().strftime("%Y%m%d%H%M%S")#current time
randomNum = random.randint(0,1000)#random number 1~1000
if randomNum < 10:
    randomNum = str(00) + str(randomNum)
elif randomNum < 100:
    randomNum = str(0) + str(randomNum)
uniqueNum = str(nowTime) + str(randomNum)
result_path = file_dir+"results/" + "model_"+str(argss.model)+"/"+uniqueNum + "/"
os.makedirs(result_path)

print("model type:"+str(argss.model)+"\t"+"result path:"+result_path)
print("low SNR:"+str(argss.lowSNR)+"\t"+"up SNR:\t"+str(argss.upSNR))

# start_time = timeit.default_timer()

for k in np.arange(len(SNR)):

    snr = str(SNR[k])

    if argss.model == 0:
        net = ResNet(BasicBlock, [1,1,1,1], num_classes=numclasses)
        print("Creating model:"+str(argss.model))
    elif argss.model == 1:
        net = ResNetANM_1(BasicBlock, [1,1,1,1], num_classes=numclasses)
        print("Creating model:"+str(argss.model))
    elif argss.model == 2:
        net = ResNetANM_2(BasicBlock, [1,1,1,1], num_classes=numclasses)
        print("Creating model:"+str(argss.model))
    elif argss.model == 3:
        net = ResNetANM_3(BasicBlock, [1,1,1,1], num_classes=numclasses)
        print("Creating model:"+str(argss.model))
    elif argss.model == 4:
        net = ResNetANM_4(BasicBlock, [1,1,1,1], num_classes=numclasses)
        print("Creating model:"+str(argss.model))
    elif argss.model == 5:
        net = ResNetANM_5(BasicBlock, [1,1,1,1], num_classes=numclasses)
        print("Creating model:"+str(argss.model))
    elif argss.model == 6:
        net = ResNetANM_6(BasicBlock, [1,1,1,1], num_classes=numclasses)
        print("Creating model:"+str(argss.model))
    elif argss.model == 7:
        net = ResNetANM_7(BasicBlock, [1,1,1,1], num_classes=numclasses)
        print("Creating model:"+str(argss.model))
    elif argss.model == 8:
        net = ResNetANM_8(BasicBlock, [1,1,1,1], num_classes=numclasses)
        print("Creating model:"+str(argss.model))
    elif argss.model == 9:
        net = ResNetANM_9(BasicBlock, [1,1,1,1], num_classes=numclasses)
        print("Creating model:"+str(argss.model))

    data = sio.loadmat(file_dir + str(snr)+'.mat')

    train_data = data['train_data']
    train_label = data['train_label']
    train_label = train_label-1

    num_train_instances = len(train_data)

    train_data = torch.from_numpy(train_data).type(torch.FloatTensor)
    train_label = torch.from_numpy(train_label).type(torch.LongTensor)
    train_data = train_data.view(num_train_instances, 1, -1)
    train_label = train_label.view(num_train_instances, 1)

    train_dataset = TensorDataset(train_data, train_label)
    train_data_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    test_data = data['test_data']
    test_label = data['test_label']
    test_label = test_label-1

    num_test_instances = len(test_data)

    test_data = torch.from_numpy(test_data).type(torch.FloatTensor)
    test_label = torch.from_numpy(test_label).type(torch.LongTensor)
    test_data = test_data.view(num_test_instances, 1, -1)
    test_label = test_label.view(num_test_instances, 1)

    test_dataset = TensorDataset(test_data, test_label)
    test_data_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    # vali
    vali_data = data['vali_data']
    vali_label = data['vali_label']
    vali_label = vali_label - 1

    num_vali_instances = len(vali_data)

    vali_data = torch.from_numpy(vali_data).type(torch.FloatTensor)
    vali_label = torch.from_numpy(vali_label).type(torch.LongTensor)
    vali_data = vali_data.view(num_vali_instances, 1, -1)
    vali_label = vali_label.view(num_vali_instances, 1)

    vali_dataset = TensorDataset(vali_data, vali_label)
    vali_data_loader = DataLoader(dataset=vali_dataset, batch_size=batch_size, shuffle=False)

    print([num_train_instances,num_vali_instances,num_test_instances])


    validation_label = []
    prediction_label = []

    net = net.cuda()
    print(net.shape)

    criterion = nn.CrossEntropyLoss(size_average=False).cuda()

    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40, 60, 80, 250, 300], gamma=0.5)
    # train_loss = np.zeros([num_epochs, 1])
    # test_loss = np.zeros([num_epochs, 1])
    train_acc = np.zeros([num_epochs, 1])
    test_acc = np.zeros([num_epochs, 1])
    vali_acc = np.zeros([num_epochs, 1])

    for epoch in range(num_epochs):
        print('SNR:',snr,'  Epoch:', epoch)
        net.train()
        scheduler.step()
        # loss_x = 0
        for i, (samples, labels) in enumerate(train_data_loader):
        # for (samples, labels) in tqdm(train_data_loader):
            samplesV = Variable(samples.cuda())
            labels = labels.squeeze()
            # print(labels)
            labelsV = Variable(labels.cuda())

            # Forward + Backward + Optimize
            optimizer.zero_grad()
            predict_label = net(samplesV)

           # predict_label = caspyra(samplesV)

            loss = criterion(predict_label[0], labelsV)
            # print(loss.item())

            # loss_x += loss.item()

            loss.backward()
            optimizer.step()

        # train_loss[epoch] = loss_x / num_train_instances

        net.eval()
        # loss_x = 0
        correct_train = 0
        for i, (samples, labels) in enumerate(train_data_loader):
            with torch.no_grad():
                samplesV = Variable(samples.cuda())
                labels = labels.squeeze()
                # print(labels)
                labelsV = Variable(labels.cuda())
                # labelsV = labelsV.view(-1)

                predict_label = net(samplesV)
                prediction = predict_label[0].data.max(1)[1]
                # print(prediction)
                correct_train += prediction.eq(labelsV.data.long()).sum()

                loss = criterion(predict_label[0], labelsV)
                # loss_x += loss.item()

        print("Training accuracy:", (100*float(correct_train)/num_train_instances))

        train_acc[epoch] = 100*float(correct_train)/num_train_instances
    #
    #
    #     loss_x = 0
        correct_test = 0
        prediction_label = []
        for i, (samples, labels) in enumerate(test_data_loader):
            with torch.no_grad():
                samplesV = Variable(samples.cuda())
                labels = labels.squeeze()
                labelsV = Variable(labels.cuda())
                # labelsV = labelsV.view(-1)

            predict_label = net(samplesV)
            prediction = predict_label[0].data.max(1)[1]

            prediction_label.append(prediction.cpu().numpy())
            correct_test += prediction.eq(labelsV.data.long()).sum()

        print("Test accuracy:", (100 * float(correct_test) / num_test_instances))

        test_acc[epoch] = 100 * float(correct_test) / num_test_instances

        # valiation set
        correct_vali = 0
        validation_label = []
        for i, (samples, labels) in enumerate(vali_data_loader):
            with torch.no_grad():
                samplesV = Variable(samples.cuda())
                labels = labels.squeeze()
                labelsV = Variable(labels.cuda())
                # labelsV = labelsV.view(-1)

            validate_label = net(samplesV)
            validation = validate_label[0].data.max(1)[1]

            validation_label.append(validation.cpu().numpy())
            correct_vali += validation.eq(labelsV.data.long()).sum()

        valiacc = str(100 * float(correct_vali) / num_vali_instances)[0:6]

        print("Vali accuracy:", (100 * float(correct_vali) / num_vali_instances))

        vali_acc[epoch] = 100 * float(correct_vali) / num_vali_instances


        sio.savemat(result_path + str(snr)+'.mat',\
                 {'train_accuracy': train_acc, 'test_accuracy':test_acc, 'vali_accuracy':vali_acc})

# end_time = timeit.default_timer()-start_time
#
# sio.savemat(result_path + 'training_time.mat', {'train_time': end_time})