import pandas as pd
import ujson
import glob
import os
import gzip
from tqdm import tqdm
pd.set_option('display.max_columns',None,'display.max_rows',1000)
from collections import Counter
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import datetime as dt
import pytz
import matplotlib.pyplot as plt

import sys

class Network(nn.Module):
    '''
    There are 3 actions available
    0 : Sell a position (-1)
    1 : Do nothing (0)
    2 : Buy a position (+1)
    '''
    def __init__(self,state_size,action_size,hidden=[260,260,130],drop_out=0.0):
        super().__init__()
        self.hidden = hidden
        self.drop_out = drop_out
        layerlist=[]
        layerlist.append(nn.BatchNorm1d(state_size))
        layerlist.append(nn.Linear(state_size, hidden[0]))
        layerlist.append(nn.Dropout(p=drop_out))
        layerlist.append(nn.ReLU(inplace=True))
        if len(hidden) > 1:
            for i in range(len(hidden[:-1])):
                layerlist.append(nn.Linear(hidden[i], hidden[i+1]))
                layerlist.append(nn.Dropout(p=drop_out))
                layerlist.append(nn.ReLU(inplace=True))
                
        layerlist.append(nn.Linear(hidden[-1], action_size))
        layerlist.append(nn.Softmax(dim=1))
        self.net = nn.Sequential(*layerlist)
        self.net.apply(init_weights)
        
    def forward(self, state):     
        x = np.array(state)
        x = torch.FloatTensor(x).to(device)
        x = self.net(x)
        return x
    
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
