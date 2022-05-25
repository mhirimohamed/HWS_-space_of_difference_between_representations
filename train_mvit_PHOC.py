import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import numpy as np
import pandas as pd
#import pydicom as pcom
import os
import os
import random
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import math
import scipy.signal as ss
from matplotlib import pyplot as plt
import scipy.io
import representations as rep
import skimage.transform as sk
import skimage.util as sku
import scipy.spatial.distance as ds
import csv
from mvit import MobileViT

m = np.array(scipy.io.loadmat('/home/mmhiri/word_spotting_PBSC/IAM/data/train.mat')['train'])
img_data = np.array(m[:,1])
lab_data= np.array(m[:,0])
alpha=np.load('/home/mmhiri/word_spotting_PBSC/IAM/data/alpha.npy')
labels=np.load('/home/mmhiri/word_spotting_PBSC/IAM/data/labels/labels_PHOC.npy')

# generate PHOC labels ... only once
#Ydata=np.zeros((lab_data.shape[0],520),dtype='f')
#i=0
#for x in lab_data:
#      tmp=np.array(rep.rep_PHOC(x[0], alpha))
#      Ydata[i,:]=tmp
#      print(x[0],i)
#      i=i+1
#np.save('/home/mmhiri/word_spotting_PBSC/IAM/data/labels/labels_PHOC.npy',Ydata)

model = MobileViT(image_size = (40, 170), dims = [96, 96, 96], channels = [32, 32, 48, 48, 64, 64, 80, 80, 96, 96, 114, 114, 128, 24], num_classes =  520).cuda() 

img_data_train=img_data[:75000]
labels_train=labels[:75000,:]
img_data_val=img_data[75000:]
labels_val=labels[75000:,:]

del img_data, labels

threshold_valid = 100
lr_par = 1e-3

losses=[['loss_train', 'loss_validation']]

for ep in range(1000):

    optimizer = optim.Adam(model.parameters(), lr=lr_par, betas=(0.9, 0.999), weight_decay=1e-7)

    ## Train
    # Preparing the Data ... 
    model.train()
    s_train = 0; nb_train = 0
    Xdata = np.zeros((img_data_train.shape[0], 1, 40, 170), dtype='f')
    i = 0
    for x in img_data_train:
        x = sk.rotate(x, random.randint(-15, 15), preserve_range=True)
        x=np.clip(x,0,255)
        mn=13.330751; std=39.222755; x=(x-mn)/std
        Xdata[i, 0, :, :] = x
        i = i + 1
    Ydata=labels_train.copy()
    shuf = np.arange(Ydata.shape[0]); np.random.shuffle(shuf)
    Y = Ydata[shuf]; Xdata = Xdata[shuf]

    for i in range(int(Xdata.shape[0]/48)):
        input=Variable(torch.from_numpy(Xdata[i*48:(i+1)*48,:,:,:]).cuda())
        target= Variable(torch.from_numpy(Y[i*48:(i+1)*48,:]).cuda())
        output = model(input)
        loss = F.binary_cross_entropy_with_logits(output, target)
        s_train = s_train + loss.item(); nb_train=nb_train+1;
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    ## validation
    model.eval()
    s_val = 0; nb_val = 0
    Xdata = np.zeros((img_data_val.shape[0], 1, 40, 170), dtype='f')
    i = 0
    for x in img_data_val:
        mn = 13.330751;
        std = 39.222755;
        x = (x - mn) / std
        Xdata[i, 0, :, :] = x
        i = i + 1
    Y = labels_val.copy()

    for i in range(int(Xdata.shape[0] / 48)):
        input = Variable(torch.from_numpy(Xdata[i * 48:(i + 1) * 48, :, :, :]).cuda())
        target = Variable(torch.from_numpy(Y[i * 48:(i + 1) * 48, :]).cuda())
        output = model(input)
        loss = F.binary_cross_entropy_with_logits(output, target)
        s_val = s_val + loss.item();
        nb_val = nb_val + 1;

    if s_val / nb_val < threshold_valid:
        threshold_valid = s_val / nb_val
        losses.append([s_train/nb_train,s_val/nb_val])
        print('error loss:',s_train / nb_train,'val loss:', threshold_valid, lr_par)
        torch.save(model.state_dict(),'/home/mmhiri/word_spotting_PBSC/IAM/test/models/model_mvit_dict_PHOC.pth')
        torch.save(model,'/home/mmhiri/word_spotting_PBSC/IAM/test/models/model_mvit_PHOC.pth')
    else:
        lr_par = lr_par / 1.1
        losses.append([s_train/nb_train,s_val/nb_val])

    with open("/home/mmhiri/word_spotting_PBSC/IAM/test/models/losses_mvit_PHOC.csv", 'w') as f:
       writer = csv.writer(f, delimiter=',')
       writer.writerows(losses)

