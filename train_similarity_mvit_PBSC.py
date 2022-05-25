import os
from tabnanny import verbose
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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
import pickle
from sklearn.svm import LinearSVC

data_pos=np.float32(np.load('/home/mmhiri/word_spotting_PBSC/IAM/data/pos_similarity_data_mvit_PBCS.npy'))
data_pos_val=data_pos[int(0.9*data_pos.shape[0]):]
data_pos=data_pos[:int(0.9*data_pos.shape[0])]

data_neg=np.float32(np.load('/home/mmhiri/word_spotting_PBSC/IAM/data/neg_similarity_data_mvit_PBCS.npy'))  
data_neg_val=data_neg[int(0.9*data_neg.shape[0]):]
data_neg=data_neg[:int(0.9*data_neg.shape[0])]

data_pos_val=np.reshape(data_pos_val,(data_pos_val.shape[0],36,53))
data_pos_val=np.expand_dims(data_pos_val, 3)
data_pos_val=np.swapaxes(data_pos_val,1,2)

data_neg_val=np.reshape(data_neg_val,(data_neg_val.shape[0],36,53))
data_neg_val=np.expand_dims(data_neg_val, 3)
data_neg_val=np.swapaxes(data_neg_val,1,2)

class modele(nn.Module):

    def __init__(self):
        super(modele, self).__init__()

        self.l1=nn.Conv2d(53,64,(1,1))
        self.b1=nn.BatchNorm2d(64)

        self.l2=nn.Conv2d(64,4,(1,1))
        self.b2=nn.BatchNorm2d(4)

        self.l3 = nn.Linear(36*4, 64)
        self.b3 = nn.BatchNorm1d(64)

        self.l4 = nn.Linear(64, 1)

        self.relu=nn.ReLU()

    def forward(self, x):

        x=self.l1(x)
        x=self.b1(x)
        x=self.relu(x)

        x=self.l2(x)
        x=self.b2(x)
        x=self.relu(x)

        x=torch.flatten(x,start_dim=1)

        x=self.l3(x)
        x=self.b3(x)
        x=self.relu(x)

        return self.l4(x)

model= modele().cuda() 

threshold_valid = 100
lr_par = 1e-3

losses=[['loss_train', 'loss_validation']]

for ep in range(1000):

    optimizer = optim.Adam(model.parameters(), lr=lr_par, betas=(0.9, 0.999), weight_decay=1e-6)

    ## Train
    s_train = 0; nb_train = 0

    ind=[i for i in range(data_pos.shape[0])]
    random.shuffle(ind)
    data_poss=data_pos[ind]
    data_poss=np.reshape(data_poss,(data_poss.shape[0],36,53))
    data_poss=np.expand_dims(data_poss, 3)
    data_poss=np.swapaxes(data_poss,1,2)

    ind=[i for i in range(data_neg.shape[0])]
    random.shuffle(ind)
    data_negs=data_neg[ind]
    data_negs=np.reshape(data_negs,(data_negs.shape[0],36,53))
    data_negs=np.expand_dims(data_negs, 3)
    data_negs=np.swapaxes(data_negs,1,2)

    for i in range(int(data_poss.shape[0]/2024)):
        
        input=np.concatenate([data_poss[i*2024:(i+1)*2024], data_negs[i*2024:(i+1)*2024]],axis=0)
        input=torch.from_numpy(input).cuda()
        output = model(input)[:,0]
        output1,output2=output[:2024], output[2024:]
        target=torch.tensor([1]*2024).cuda()
        loss=F.margin_ranking_loss(output1, output2, target,margin=1,reduce='sum')
        s_train = s_train + loss.item(); nb_train=nb_train+1;
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    ## validation
    model.eval()
    s_val = 0; nb_val = 0

    for i in range(int(data_pos_val.shape[0]/2024)):

        input=np.concatenate([data_pos_val[i*2024:(i+1)*2024], data_neg_val[i*2024:(i+1)*2024]],axis=0)
        input=torch.from_numpy(input).cuda()
        output = model(input)[:,0]
        output1,output2=output[:2024], output[2024:]
        target=torch.tensor([1]*2024).cuda()
        loss=F.margin_ranking_loss(output1, output2, target,margin=1,reduce='sum')
        s_val = s_val + loss.item()
        nb_val = nb_val + 1;

    if s_val / nb_val < threshold_valid:
        threshold_valid = s_val / nb_val
        losses.append([s_train/nb_train,s_val/nb_val])
        print('error loss:',s_train / nb_train,',         val loss:', threshold_valid,',     ' ,lr_par)
        torch.save(model.state_dict(),'/home/mmhiri/word_spotting_PBSC/IAM/test/models/model_similarity_dict_mvit_PBSC.pth')
    else:
        lr_par = lr_par / 1.05
        losses.append([s_train/nb_train,s_val/nb_val])

    with open("/home/mmhiri/word_spotting_PBSC/IAM/test/models/losses_similarity_mvit_PBSC.csv", 'w') as f:
       writer = csv.writer(f, delimiter=',')
       writer.writerows(losses)

