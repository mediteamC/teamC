#!/usr/bin/env python
# coding: utf-8

# In[2]:


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

print(torch.get_num_threads())
# In[3]:


import teamc_pipline
import psutil
print(psutil.virtual_memory())

# In[4]:


mapped_labels = pd.read_excel('NACC_LABELS_CLASSIFICATION_TASK_NEW.xlsx')
name_touse = mapped_labels.Address_Name[:50]


# In[5]:


X = teamc_pipline.data_mapper.data_mapping(['Sex_Bin','Age_Norm'],mapping_file_path='NACC_LABELS_CLASSIFICATION_TASK_NEW.xlsx',data_path = '/work/03263/jcha9928/sharedirectory/nacc/',data_names = list(name_touse))


# In[6]:


data_set, labels, features = X.execute(size=64)


# In[7]:

x = torch.from_numpy(data_set).float()
y = torch.from_numpy(labels).long()
w = torch.from_numpy(features['Sex_Bin']).float()
z = torch.from_numpy(features['Age_Norm']).float()
print(labels)
print(y,w,z)

# In[8]:

train_data = torch.utils.data.TensorDataset(x, y, w, z)
train_size = int(0.7 * len(train_data))
val_size = int(0.1 * len(train_data))
test_size = int(0.2 * len(train_data))
train_dataset,val_dataset, test_dataset = torch.utils.data.random_split(train_data, [train_size,val_size, test_size])


# In[9]:


from teamc_pipline import resnet152
from teamc_pipline import recorder


# In[10]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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


# In[11]:

torch.set_num_threads(48)
print(psutil.virtual_memory()) 
net = resnet152(pretrained=False, num_classes=5)


# In[12]:

batch_size = 5
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,shuffle=True, num_workers=0)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,shuffle=True, num_workers=0)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,shuffle=True, num_workers=0)
print(psutil.virtual_memory())


# In[13]:


init_lr = 0.01 #SELECT INITIAL LR

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=init_lr, momentum=0.9, weight_decay=5e-4)

def adjust_lr(optimizer, epoch):
    lr = init_lr * (0.1 ** (epoch // 3)) #select decrease function
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# In[14]:


def weight_reset(m):
    if isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
        m.reset_parameters()    
        
net.apply(weight_reset)


# In[15]:


#Train

net.train(True)

epochs=5 #loop over the dataset multiple times
aux_input = 1
#set up recprder
recording_log = recorder(epochs,5)
for epoch in range(epochs):  

    train_class_total,train_class_correct = [0]*5,[0]*5
    val_class_total,val_class_correct = [0]*5,[0]*5

    adjust_lr(optimizer, epoch)
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
    alpha_rate = lr
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs
        inputs, labels, sex, age = data
        #inputs, labels = data
        # zero the parameter gradients
        optimizer.zero_grad()
        print(psutil.virtual_memory())
        # forward + backward + optimize
        outputs = net(inputs, sex, age)
        #outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        
        print_cycle = 1
        running_loss += loss.item()
        if i % print_cycle == print_cycle-1:    # print every print_cycle mini-batches
             for param_group in optimizer.param_groups:
                print('[Epoch: %d, Mini Batch: %5d] loss: %.3f / Lr: %.5f'  %
                      (epoch + 1, i + 1, running_loss / print_cycle, param_group['lr']))
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
                for i in range(batch_size): #batchsize of train loader
                    label = labels[i]
                    train_class_correct[label] += c[i].item()
                    train_class_total[label] += 1

        del outputs
        del predicted
        del c
    print(np.sum(train_class_total))
    print(train_class_total) 
    with torch.no_grad():
        for data in val_loader: #train loader
            images, labels, sex, age = data
            outputs = net(images, sex, age)
            _, predicted = torch.max(outputs.data, 1)
            c = (predicted == labels).squeeze()
            for i in range(batch_size): #batchsize of train loader
                label = labels[i]
                val_class_correct[label] += c[i].item()
                val_class_total[label] += 1
    del outputs
    del predicted
    del c

    recording_log.add_record(epoch,lr,train_class_total,train_class_correct,val_class_total,val_class_correct)
    recording_log.out_file(path='try1.csv')
    torch.save(net.state_dict(), 'MODEL_SAVE')
                        
       

print('Finished Training')




