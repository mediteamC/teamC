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
import torch.utils.data

class data_mapping(object):
    def __init__(self,feature_names,mapping_file_path='NACC_LABELS_CLASSIFICATION_TASK_NEW_debug.xlsx',data_path = 'Image/NACC/',data_names = []):
        self.data = pd.read_excel(mapping_file_path)
        self.feature_names = feature_names
        self.data_path = data_path
        self.data = self.data[self.data.Address_Name.isin(data_names)]
        self.data = self.data[['Address_Name','Label_ID','Diagnosis'] + self.feature_names]
        self.data['labels'] = self.data.apply(lambda x: self.get_label(x['Diagnosis']),axis = 1)
        data_path = os.path.join(data_path,'fs_t1_nacc.tar')
        self.archive = tarfile.open(data_path, 'r')
        self.image_names = data_names
        
    def assemble_features(self):
        if self.feature_names == []:
            return -1
        result = {}
        for f in self.feature_names:
            result[f] = np.array(getattr(self.data,f))
        return result
    #assenmble data from the tar file
    def assemble_data(self,size=10):
        new_size = 256-2*size
        self.X = np.zeros((len(self.data),1,new_size,new_size,new_size))
        invalid_address = []
        count = 0
        #print(len(self.image_names),len(self.data))
        for index,row in self.data.iterrows():
            if count % 100 == 0:
                print(count)
            member = row.Address_Name
            ylabel = row['labels']
            file_path = os.path.join(self.data_path,member)
            try:
                f = nib.load(file_path)
            except:
                try:
                    newpath = os.path.join('Image/NACC/',member)
                    self.archive.extract(member,path='Image/NACC/')
                    f = nib.load(newpath)
                except:
                    print('row:',row['Label_ID'],'has no valid Address')
                    invalid_address.append(row.Label_ID)
                    continue
            try:
                temp_img = f.get_data()
                temp_img = self.normlized_img(temp_img)
                temp_img = self.memory_reduction(temp_img,size)
               # print(self.data.iloc[count])
               # print(count)
                self.X[count][0] = temp_img
                count+=1
                os.remove(newpath)
            except  Exception as e:
                print(str(e))
                raise
        useful_data = self.data[~self.data.Label_ID.isin(invalid_address)]
        self.Y = np.array(useful_data.Diagnosis).reshape(len(self.data)-len(invalid_address),-1).ravel()
        self.X = self.X.reshape(-1,1,new_size,new_size,new_size)
        
    #helper function
    def get_label(self,tag):# For array output
        return [1 if i == tag else 0 for i in range(2)]
    
    def normlized_img(self,image):
        v_min = image.min()
        v_max = image.max()
        image = (image - v_min)/(v_max - v_min)
        return image
    
    def memory_reduction(self,image_data,size=10):
        return image_data[size:256-size,size:256-size,size:256-size]
    
    def execute(self,size=10):
        feature_set = self.assemble_features()
        self.assemble_data(size=size)
        self.archive.close()
        return self.X,self.Y,feature_set
