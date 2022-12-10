import os
import sys
import time
import math
import glob
import random
import requests
import numpy as np
import logging

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import optuna
import plotly

from optuna.samplers import TPESampler
from optuna.samplers import NSGAIISampler
from optuna.samplers import CmaEsSampler

import nibabel as nib
from sklearn.model_selection import StratifiedKFold

import argparse
parser = argparse.ArgumentParser(description='PyTorch Hippocampal Sclerosis Classification Training')
parser.add_argument('--opt', type=str, choices=["Adam","SGD"], help='Optimizer')
parser.add_argument('--mom', type=float, help='Momentum for the optimizer')
parser.add_argument('--sch', type=str, choices=["CALR","SLR","RLRP"], help='Learning rate scheduler')
parser.add_argument('--mode', type=str, choices=["LAST_acc","BEST_acc","LAST_loss","BEST_loss","DUAL"], default="LAST_acc", help='Optimization Target')
parser.add_argument('--epoch', type=int, default=50, help='Train epoch per each Trial')
parser.add_argument('--ntrial', type=int, default=40, help='Trial iterations')
parser.add_argument('--prune', type=int, default=1, help='Whether to prune or not')
parser.add_argument('--pruner', type=str, help='Which pruner to use')
parser.add_argument('--debug', type=int, default=0, help='Debugging')
parser.add_argument('--batch', type=int, default=128, help='Batch_size')
parser.add_argument('--lrate', type=float, default=0.1, help='Learning Rate for Training')
parser.add_argument('--lreduce', type=float, default=0.1, help='Learning Rate Reduce')
parser.add_argument('--lrperiod', type=int, help='Learning Rate Reduce')
parser.add_argument('--wdecay', type=float, help='Learning Rate for Training')
parser.add_argument('--msg', type=str, help='Training Setting Message')
parser.add_argument('--seed', type=int, default = 1, help='Random seed')
parser.add_argument('--datatype', type=int, default = 0, help='Which Data to Load')
parser.add_argument('--controltype', type=str, help='HS or ADNI')
parser.add_argument('--imptnce', type=int, default = 1, help='Whether to save Importance figure')
parser.add_argument('--cma', type=int, default = 0, help='Random sampler')
parser.add_argument('--multiV', type=int, default = 0, help='multi_variate')
parser.add_argument('--ksize', type=int, default = 0, help='kernel_size')

opts = parser.parse_args()
opt = opts.opt
mom = opts.mom
sch = opts.sch
mode = opts.mode
prune = opts.prune
pruner = opts.pruner
imptnce = opts.imptnce
end_epoch = opts.epoch
datatype = opts.datatype
controltype = opts.controltype
ntrial = opts.ntrial
seed = opts.seed
msg = opts.msg
cma = opts.cma
multiV = opts.multiV

nb_KFold = 5
FOLD_SEED = 0
KFOLD = StratifiedKFold(n_splits=nb_KFold, random_state=FOLD_SEED, shuffle=True)
FOLD_SEED = FOLD_SEED + 1
device = 'cuda' if torch.cuda.is_available() else 'cpu'

start = time.time()
criterion = nn.CrossEntropyLoss()

def seed_set(rdmsd):
    random.seed(rdmsd)
    np.random.seed(rdmsd)
    torch.manual_seed(rdmsd)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed(rdmsd)
    else:
        print('[CUDA unavailable]'); 
        sys.exit()

# =============================================================================================================== #
# ================================================== Data Load ================================================== #
# =============================================================================================================== #


# Experiment Initialization
nb_KFold = 5
drop_rate = 0.5 ; weight_regularizer = 1e-5
imgRow = 160 ; imgCol = 200 ; imgDepth = 170

## Multi Mode
if(datatype == 60):  ## Multi_class_Oblcor 170
    dirDataSet = "../dataset/60_Obl_160_LPI_160EAM"
    imgCountNO_0 = 66; imgCountNO_7 = 66; imgCountNO_4 = 28
    imgCountYES_1 = 100; imgCountYES_2 = 60; imgCountYES_3 = 0
    imgCountYES = imgCountYES_1 + imgCountYES_2 + imgCountYES_3  
    imgCountNO = imgCountNO_0 + imgCountNO_4 + imgCountNO_7  
    
elif(datatype == 64): ## Multi Mode debug mode
    dirDataSet = "../dataset/64_debug"
    imgCountNO_0 = 10; imgCountNO_7 = 5; imgCountNO_4 = 5
    imgCountYES_1 = 10; imgCountYES_2 = 10; imgCountYES_3 = 0
    imgCountYES = imgCountYES_1 + imgCountYES_2 + imgCountYES_3  
    imgCountNO = imgCountNO_0 + imgCountNO_4 + imgCountNO_7

elif(datatype == 30): # ADNI1
    dirDataSet = "../dataset/30_ADNI1_670EA"
    imgCountNO_0 = 185  # CN
    imgCountNO_7 = 0
    imgCountNO_4 = 0 
    imgCountYES_1 = 160  # AD
    imgCountYES_2 = 325  # MCI
    imgCountYES_3 = 0
    imgCountYES = imgCountYES_1 + imgCountYES_2 + imgCountYES_3  
    imgCountNO = imgCountNO_0 + imgCountNO_4 + imgCountNO_7  

elif(datatype == 31): # ADNI1
    dirDataSet = "../dataset/31_ADNI1_Bal_480EA"
    imgCountNO_0 = 160  # CN
    imgCountNO_7 = 0
    imgCountNO_4 = 0 
    imgCountYES_1 = 160  # AD
    imgCountYES_2 = 160  # MCI
    imgCountYES_3 = 0
    imgCountYES = imgCountYES_1 + imgCountYES_2 + imgCountYES_3  
    imgCountNO = imgCountNO_0 + imgCountNO_4 + imgCountNO_7  

elif(datatype == 32): # ADNI1 debug
    dirDataSet = "../dataset/32_ADNI1_debug"
    imgCountNO_0 = 20
    imgCountNO_7 = 0
    imgCountNO_4 = 0 
    imgCountYES_1 = 20
    imgCountYES_2 = 10  # MCI
    imgCountYES_3 = 0
    imgCountYES = imgCountYES_1 + imgCountYES_2 + imgCountYES_3  
    imgCountNO = imgCountNO_0 + imgCountNO_4 + imgCountNO_7  
    
elif(datatype == 35): # ADNI1
    dirDataSet = "../dataset/35_ADNI12_BAL_260EA"
    imgCountNO_0 = 260  # CN
    imgCountNO_7 = 0
    imgCountNO_4 = 0 
    imgCountYES_1 = 260  # AD   
    imgCountYES_2 = 260  # MCI
    imgCountYES_3 = 0
    imgCountYES = imgCountYES_1 + imgCountYES_2 + imgCountYES_3  
    imgCountNO = imgCountNO_0 + imgCountNO_4 + imgCountNO_7  
    
if(controltype == "CLRM"): categories = ["C047_no","C1_left_yes","C2_right_yes"]
elif(controltype == "ADNI"): categories = ["ADNI_CN","ADNI_AD","ADNI_MCI"]

if('BN' in controltype): 
    imgCountYES_2 = 0
    categories = ["ADNI_CN","ADNI_AD"]
    
nb_classes = len(categories)
imgCount = imgCountNO + imgCountYES


def source_load(categories,dirDataSet):
    inputX, inputY, listFileName, niiInfo = data_single(categories,dirDataSet)    
    Y_vector = []
    for i in range(imgCountNO_4): Y_vector.append(4)
    for i in range(imgCountNO_7): Y_vector.append(7)
    for i in range(imgCountNO_0): Y_vector.append(0)
    for i in range(imgCountYES_1): Y_vector.append(1)
    for i in range(imgCountYES_2): Y_vector.append(2)
    for i in range(imgCountYES_3): Y_vector.append(3)            
    return inputX, inputY, Y_vector, listFileName, niiInfo

def data_single(categories, dirDataSet):
    inputX = []; inputY = []; listFileName = []
    for idx, f in enumerate(categories):        
        label = idx
        image_dir = dirDataSet + "/" + f
        for (imagePath, dir, files) in sorted(os.walk(image_dir)):
            ## read 3D nii file ###
            if(imagePath == image_dir):                
                # image reading #
                files = sorted(glob.glob(imagePath + "/*.nii.gz"))
                listFileName += (files)
    
            for i, fname in enumerate(files):
                if(i==0): niiInfo = nib.load(fname)
                img = nib.load(fname).get_fdata()
                inputX.append(img)
                inputY.append(label)
    inputX = torch.tensor(inputX)
    inputY = torch.tensor(inputY)
    listFileName = np.array(listFileName,dtype=object)
    return inputX, inputY, listFileName, niiInfo

def CV_data_load(inputX,inputY,train_index,val_index,AUG,batch_size): # switch : train/eval
    ### Dividing data into folds
    x_train = inputX[train_index]
    x_val = inputX[val_index]
    y_train = inputY[train_index]
    y_val = inputY[val_index]

    if(AUG=='hflip'):    
        temp_data = torch.flip(x_train,[1,]) # 0:batch  1:horizontal  2:forward&backward   3:vertical
        
        if not('ADNI' in controltype):
            temp_label =  torch.zeros(len(y_train))
            for idx in range(len(y_train)):
                if(y_train[idx] == 1) : temp_label[idx]= 2
                elif(y_train[idx] == 2) : temp_label[idx]= 1 
                else : temp_label[idx]= y_train[idx]
            temp_label = temp_label.long()
        else: temp_label = y_train
            
        x_train = torch.cat([x_train,temp_data],dim=0)
        y_train = torch.cat([y_train,temp_label],dim=0)

    train_data = torch.utils.data.TensorDataset(x_train, y_train)
    val_data = torch.utils.data.TensorDataset(x_val, y_val)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size =batch_size, shuffle = True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size = batch_size, shuffle = True)
    return train_loader, val_loader
# =============================================================================================================== #
# ==================================================== Model ==================================================== #
# =============================================================================================================== #

class Torch3D(nn.Module):

    def __init__(self,ksize,num_classes=nb_classes):
        super().__init__()
        self.ksize = ksize
        if(self.ksize==3): self.linear=48000
        elif(self.ksize==4): self.linear=35640

        NumFilter = [5,10,20,40,60] # Number of Convolutional Filters to use
        NumDense = [64,64,3]
        kernel_size = [self.ksize,self.ksize,self.ksize] # Convolution Kernel Size
        stride_size = (1,1,1) # Convolution Stride Size
        pad_size = [1,1,1] # Convolution Zero Padding Size
        pool_size = [2,2,2]
        
        self.extractor = nn.Sequential(
            nn.Conv3d(1, NumFilter[0], bias=True, kernel_size=kernel_size, stride=stride_size, padding = pad_size),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2),

            nn.Conv3d(NumFilter[0],NumFilter[1], bias=True, kernel_size=kernel_size, stride=stride_size, padding = pad_size),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2),

            nn.Conv3d(NumFilter[1],NumFilter[2], bias=True, kernel_size=kernel_size, stride=stride_size, padding = pad_size),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2),

            nn.Conv3d(NumFilter[2],NumFilter[3], bias=True, kernel_size=kernel_size, stride=stride_size, padding = pad_size),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2)
            )
        
        self.classifier = nn.Sequential(
            nn.Linear(self.linear, 64, bias=True),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=drop_rate),

            nn.Linear(64, 64, bias=True),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=drop_rate),

            nn.Linear(64, num_classes, bias=True)
            )
        
    def forward(self, x):
        x = self.extractor(x)
        x = x.view(-1,self.linear)
        x = self.classifier(x)
        
        return x


def HSCNN(ksize):
    return Torch3D(ksize)


class TorchAD(nn.Module):

    def __init__(self,ksize,num_classes=nb_classes):
        super().__init__()
        self.ksize = ksize
        if(self.ksize==3): self.linear=9000
        elif(self.ksize==4): self.linear=4800
        elif(self.ksize==5): self.linear=2160
            
        NumFilter = [5,10,20,40,60] # Number of Convolutional Filters to use
        kernel_size = [self.ksize,self.ksize,self.ksize] # Convolution Kernel Size
        stride_size = (1,1,1) # Convolution Stride Size
        pad_size = [1,1,1] # Convolution Zero Padding Size
        pool_size = [2,2,2]
        
        self.extractor = nn.Sequential(
            nn.Conv3d(1, NumFilter[0], bias=True, kernel_size=kernel_size, stride=stride_size, padding = pad_size),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2),

            nn.Conv3d(NumFilter[0],NumFilter[1], bias=True, kernel_size=kernel_size, stride=stride_size, padding = pad_size),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2),

            nn.Conv3d(NumFilter[1],NumFilter[2], bias=True, kernel_size=kernel_size, stride=stride_size, padding = pad_size),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2),

            nn.Conv3d(NumFilter[2],NumFilter[3], bias=True, kernel_size=kernel_size, stride=stride_size, padding = pad_size),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2),
            
            nn.Conv3d(NumFilter[3],NumFilter[4], bias=True, kernel_size=kernel_size, stride=stride_size, padding = pad_size),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2)
            )
        
        self.classifier = nn.Sequential(
            nn.Linear(self.linear, 64, bias=True),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=drop_rate),

            nn.Linear(64, 64, bias=True),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=drop_rate),

            nn.Linear(64, 64, bias=True),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=drop_rate),

            nn.Linear(64, num_classes, bias=True)
            )
        
    def forward(self, x):
        x = self.extractor(x)
        x = x.view(-1,self.linear)
        x = self.classifier(x)        
        return x


def ADNICNN(ksize):
    return TorchAD(ksize)

# =============================================================================================================== #
# ==================================================== Record =================================================== #
# =============================================================================================================== #

def format_time(seconds):
    days = int(seconds / 3600/24) ; seconds = seconds - days*3600*24 ; hours = int(seconds / 3600)
    seconds = seconds - hours*3600 ; minutes = int(seconds / 60) ; seconds = seconds - minutes*60
    secondsf = int(seconds) ; seconds = seconds - secondsf ; millis = int(seconds*1000)
    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

def study_record(study):    
    log_path = './log'
    dir_path = log_path+'/%d%d'%(end_epoch,ntrial)
    if(not(os.path.isdir(log_path))):os.mkdir(log_path)
    if(not(os.path.isdir(dir_path))):os.mkdir(dir_path)

    Summary = log_path+'/study_%s.txt' %(mode)
    Sum = open(Summary,'a')
    parser = str(opts)
    Sum.write(parser[9:])
    best_trials = '\n' + str(study.best_trials)
    Sum.write(best_trials)
    fin = time.time() - start
    time_elapse = '\nTime elapsed : '+format_time(fin)+'\n\n'
    Sum.write(time_elapse)
    Sum.close()
        
    if(imptnce):
        # Put the 'orca.AppImgae' at somewhere in your PATH
        plotly.io.orca.config.executable = '/home/dokim/orca.AppImage'
        if(mode != "DUAL"):
            fig = optuna.visualization.plot_param_importances(study,target_name=mode)
            fig.write_image(dir_path+'/Importance_%s_Optim%s_Sched%s_Epoch%d_Ntrial%d_Pruner%s_%s.png'%(mode,opt,sch,end_epoch,ntrial,pruner,msg))
            fig = optuna.visualization.plot_parallel_coordinate(study)
            fig.write_image(dir_path+'/Parallel_%s_Optim%s_Sched%s_Epoch%d_Ntrial%d_Pruner%s_%s.png'%(mode,opt,sch,end_epoch,ntrial,pruner,msg))
        else:
            fig = optuna.visualization.plot_param_importances(study,target_name="Accuracy",target=lambda t: t.values[0])
            fig.write_image(dir_path+'/Importance_%s_Acc_Optim%s_Sched%s_Epoch%d_Ntrial%d_Pruner%s_%s.png'%(mode,opt,sch,end_epoch,ntrial,pruner,msg))
            fig = optuna.visualization.plot_param_importances(study,target_name="Loss",target=lambda t: t.values[1])
            fig.write_image(dir_path+'/Importance_%s_Lss_Optim%s_Sched%s_Epoch%d_Ntrial%d_Pruner%s_%s.png'%(mode,opt,sch,end_epoch,ntrial,pruner,msg))


def best_last(_list, trial,option):
    tensor = torch.tensor(_list,dtype=float)
    tensor = tensor.view(nb_KFold,-1)
    mean = torch.mean(tensor,dim=0)
    if(option): argidx = torch.argmax(mean)
    else : argidx = torch.argmin(mean)
    best = mean[argidx]
    last = mean[-1]
    return best, last