#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd


# In[24]:


class recorder(object):
    def __init__(self,total_epoch,label_num):
        self.label_num = label_num
        self.total_epoch = total_epoch
        self.inventory = np.zeros((total_epoch,4*label_num+1))
        self.names = ['Learning_rate']
        for i in range(label_num):
            self.names += ['Train_%d_Total' % i,'Train_%d_Correct' % i,'Val_%d_Total' %i,'Val_%d_Correct' %i]

    def add_record(self,epoch,learning_rate,train_class_total,train_class_correct,val_class_total,val_class_correct):
        self.inventory[epoch][0] = learning_rate
        count = 0
        for i in range(1,4*self.label_num+1,4):
            self.inventory[epoch][i] = train_class_total[count]
            self.inventory[epoch][i+1] = train_class_correct[count]
            self.inventory[epoch][i+2] = val_class_total[count]
            self.inventory[epoch][i+3] = val_class_correct[count]
            count += 1
    
    def out_file(self,start_num,path='Result.csv'):
        df = pd.DataFrame(self.inventory)
        df.columns = self.names
        df.index.names = ['epoch']
        df.index += start_num
        df.to_csv(path)

