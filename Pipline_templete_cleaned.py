#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import json
import tarfile
import zipfile
import nibabel as nib
from nilearn import plotting
from sklearn.preprocessing import normalize
import torch
from torch.utils.data.sampler import SubsetRandomSampler
import teamc_pipline
import psutil

print(psutil.virtual_memory())

# Hard coded parameters dict
#params = {'import_path':'','batch_size':25,'init_lr':0.01,'decay_freq':25,'data_length':3700,'record_name':'try','state_name':'MODEL_SAVE','epochs':5} # USE when you do first trainning
params = {'import_path':'MODEL_SAVE0','batch_size':25,'init_lr':0.1,'decay_freq':20,'data_length':3000,'record_name':'try','state_name':'MODEL_SAVE','epochs':5}

data_length = params['data_length']
batch_size = params['batch_size']
init_lr = params['init_lr']
decay_freq = params['decay_freq']
record_name = params['record_name']
state_name = params['state_name']
epochs=params['epochs']
import_path = params['import_path']



mapped_labels = pd.read_excel('NACC_LABELS_CLASSIFICATION_TASK_NEW_debug.xlsx')
name_touse = mapped_labels.Address_Name[:data_length+1]

X = teamc_pipline.data_mapper.data_mapping(['Sex_Bin','Age_Norm'],mapping_file_path='NACC_LABELS_CLASSIFICATION_TASK_NEW_debug.xlsx',data_path = '/work/03263/jcha9928/sharedirectory/nacc/',data_names = list(name_touse))
data_set, labels, features = X.execute(size=64)
x = torch.from_numpy(data_set).float()
y = torch.from_numpy(labels).long()
w = torch.from_numpy(features['Sex_Bin']).float()
z = torch.from_numpy(features['Age_Norm']).float()
print(labels)
print(y,w,z)

train_data = torch.utils.data.TensorDataset(x, y, w, z)

indices = list(range(len(train_data)))

# Train test val split
train_size = int(0.7 * len(train_data))
val_size = int(0.1 * len(train_data))
test_size = len(train_data)-train_size-val_size
train_indices, val_indices,test_indices = indices[:train_size], indices[train_size:train_size+val_size],indices[train_size+val_size:]
train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)
test_sampler =SubsetRandomSampler(test_indices)

# Set up DL models
from teamc_pipline import resnet152
from teamc_pipline import recorder
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial
import scipy
import tarfile
import nibabel as nib
import os
import zipfile
from nilearn import plotting
import time
from nilearn import image
import sys
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.utils.model_zoo as model_zoo

import_path = import_path
torch.set_num_threads(48)
print(psutil.virtual_memory())
if import_path == '':
    net = resnet152(pretrained=False, num_classes=2)
    model_num = 0
else:
   model_loader = torch.load(import_path)
   net = resnet152(pretrained=False, num_classes=5)
   net.load_state_dict(model_loader['state_dict'])
   model_num = model_loader['epoch']

train_loader = torch.utils.data.DataLoader(train_data,sampler=train_sampler, batch_size=batch_size, num_workers=0)
val_loader = torch.utils.data.DataLoader(train_data,sampler=val_sampler, batch_size=batch_size, num_workers=0)
test_loader = torch.utils.data.DataLoader(train_data,sampler=test_sampler ,batch_size=batch_size, num_workers=0)
print(psutil.virtual_memory())


#init_lr = 0.01 #SELECT INITIAL LR

criterion = nn.CrossEntropyLoss()

def adjust_lr(optimizer, epoch,decay_f=decay_freq):
    lr = init_lr * (0.1 ** (epoch // decay_f)) #select decrease function
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def weight_reset(m):
    if isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
        m.reset_parameters()


if import_path == '':
    optimizer = optim.SGD(net.parameters(), lr=init_lr, momentum=0.9, weight_decay=5e-4)
    net.apply(weight_reset)
else:
    optimizer = optim.SGD(net.parameters(), lr=init_lr, momentum=0.9, weight_decay=5e-4)
    optimizer.load_state_dict(model_loader['optimizer'])




#Train

net.train(True)
start_num = model_num
#epochs=5 #loop over the dataset multiple times

#set up recprder
recording_log = recorder(epochs,5)
for epoch in range(epochs):  

    train_class_total,train_class_correct = [0]*5,[0]*5
    val_class_total,val_class_correct = [0]*5,[0]*5

    adjust_lr(optimizer, epoch+model_num)
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
    alpha_rate = lr
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs
        inputs, labels, sex, age = data
        optimizer.zero_grad()
        print(psutil.virtual_memory())
        outputs = net(inputs, sex, age)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        
        print_cycle = 1
        running_loss += loss.item()
        if i % print_cycle == print_cycle-1:    # print every print_cycle mini-batches
             for param_group in optimizer.param_groups:
                print('[Epoch: %d, Mini Batch: %5d] loss: %.3f / Lr: %.5f'  %
                      (epoch + start_num  + 1, i + 1, running_loss / print_cycle, param_group['lr']))
                running_loss = 0.0        
        del loss
        del outputs
    
    # Get the trainning labels
    with torch.no_grad():
        for data in train_loader: #train loader
            images, labels, sex, age = data
            outputs = net(images, sex, age)
            _, predicted = torch.max(outputs.data, 1)
            c = (predicted == labels).squeeze()
            for i in range(len(labels)): #batchsize of train loader
                label = labels[i]
                train_class_correct[label] += c[i].item()
                train_class_total[label] += 1

    del outputs
    del predicted
    del c

    print(np.sum(train_class_total))
    print(train_class_total)
    print(train_class_correct)

    net.train(False) 
    with torch.no_grad():
        for data in val_loader: #train loader
            images, labels, sex, age = data
            outputs = net(images, sex, age)
            _, predicted = torch.max(outputs.data, 1)
            c = (predicted == labels).squeeze()
            for i in range(len(labels)): #batchsize of train loader
                label = labels[i]
                val_class_correct[label] += c[i].item()
                val_class_total[label] += 1
    del outputs
    del predicted
    del c

    net.train(True)
    recording_log.add_record(epoch,lr,train_class_total,train_class_correct,val_class_total,val_class_correct)
    out_file_name = record_name  + str(epoch+start_num+1) + '.csv'
    recording_log.out_file(start_num,path=out_file_name)
    try:
        os.remove(record_name+str(epoch+start_num)+'.csv')
    except:
        print('No such file')
    file_name = state_name + str(epoch+start_num)
    stats = {'epoch':epoch+start_num+1,'state_dict': net.state_dict(),
             'optimizer': optimizer.state_dict()}
    torch.save(stats, file_name)
                        
       

print('Finished Training')




