import os
import time
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
#import torch.backends.cudnn as cudnn
from torchsummary import summary

import math
import gzip
import glob
import random
import numpy as np
import nibabel as nib
from sklearn.model_selection import StratifiedKFold
from torch.autograd import Variable
import scipy
import scipy.io as io

from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score, recall_score

from Args.argument import get_args
args = get_args()

# Single Model Training
AUG = args.AUG
SETT = args.SETT
TALK = args.TALK
TRIAL = args.TRIAL
MODEL = args.MODEL
BATCH = args.BATCH
DATATYPE = args.DATATYPE
ENDEPOCH = args.ENDEPOCH
KERNEL_SEED = args.KERNEL_SEED
CONTROLTYPE = args.CONTROLTYPE

# Ensemble 
EnsMODE = args.EnsMODE # 'AVR' or 'VOT'
K = args.K

# LRP
RULE = args.RULE
LRPMSG = args.LRPMSG
PERCENT = args.PERCENT

# Model Search
mg = args.mgpu
debug = args.debug
drop_ = args.drop
step = args.step
lr = args.lr

# Seed
torch.manual_seed(KERNEL_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(KERNEL_SEED)
random.seed(KERNEL_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(KERNEL_SEED)
else:
    print('[CUDA unavailable]')
    sys.exit()
start = time.time()

tmp_percent = 0.05
temp_Loss = nn.MSELoss()


# Experiment Initialization
nb_KFold = 5 ; epochs = 300 ; step_size = step ; iters=100
drop_rate = 0.5 ; learning_rate = lr ; weight_regularizer = 1e-5
imgRow = 160 ; imgCol = 200 ; imgDepth = 170

FOLD_SEED = 0
KFOLD = StratifiedKFold(n_splits=nb_KFold, random_state=FOLD_SEED, shuffle=True)
FOLD_SEED = FOLD_SEED + 1

decay_rate = learning_rate / step_size
criterion = nn.CrossEntropyLoss()

if(EnsMODE=='AVR')or(EnsMODE=='VOT'):
    graph_path = '/graph/T%d%s' %(TRIAL,EnsMODE)
    KERNELS = K
    ENSEMBLE_NUM = len(KERNELS) # 5 or 10 or 15
else:
    graph_path = '/graph/T%dK%d' %(TRIAL,KERNEL_SEED)


if(not(os.path.isdir(os.getcwd()+graph_path))):
    os.mkdir(os.getcwd()+graph_path)

if(debug): # debug mode
#    epochs = 3 ; step_size = 2
    epochs = 5 ; step_size = 2

## Multi Mode
if(DATATYPE == 60):  ## Multi_class_Oblcor 170
    dirDataSet = "../dataset/60_Obl_160_LPI_160EAM"
    if(CONTROLTYPE=='CCLR')or(CONTROLTYPE=="CLRM"):
        imgCountNO_0 = 66
        imgCountNO_7 = 66
        imgCountNO_4 = 28
    elif(CONTROLTYPE=='CZLR'):
        imgCountNO_0 = 160
        imgCountNO_7 = 0
        imgCountNO_4 = 0
    elif(CONTROLTYPE=='CCM'):
        imgCountNO_0 = 66
        imgCountNO_7 = 66
        imgCountNO_4 = 28       
    imgCountYES_1 = 100
    imgCountYES_2 = 60
    imgCountYES_3 = 0
    imgCountYES = imgCountYES_1 + imgCountYES_2 + imgCountYES_3  
    imgCountNO = imgCountNO_0 + imgCountNO_4 + imgCountNO_7  
    


elif(DATATYPE == 61):  ## Multi_class_Oblcor 170
    dirDataSet = "../dataset/61_Ax_160_LPI_160EAM"
    if(CONTROLTYPE=='CCLR')or(CONTROLTYPE=="CLRM"):
        imgCountNO_0 = 66
        imgCountNO_7 = 66
        imgCountNO_4 = 28
    elif(CONTROLTYPE=='CZLR'):
        imgCountNO_0 = 160
        imgCountNO_7 = 0
        imgCountNO_4 = 0 
    imgCountYES_1 = 100
    imgCountYES_2 = 60
    imgCountYES_3 = 0
    imgCountYES = imgCountYES_1 + imgCountYES_2 + imgCountYES_3  
    imgCountNO = imgCountNO_0 + imgCountNO_4 + imgCountNO_7  

elif(DATATYPE == 62):  ## Multi_class_Oblcor 170
    dirDataSet = "../dataset/62_Sum_320_LPI_320EAM"
    if(CONTROLTYPE=='CCLR')or(CONTROLTYPE=="CLRM"):
        imgCountNO_0 = 132
        imgCountNO_7 = 132
        imgCountNO_4 = 56
    elif(CONTROLTYPE=='CZLR'):
        imgCountNO_0 = 160
        imgCountNO_7 = 0
        imgCountNO_4 = 0 
    imgCountYES_1 = 200
    imgCountYES_2 = 120
    imgCountYES_3 = 0
    imgCountYES = imgCountYES_1 + imgCountYES_2 + imgCountYES_3  
    imgCountNO = imgCountNO_0 + imgCountNO_4 + imgCountNO_7  

elif(DATATYPE == 63):  ## Multi_class_Oblcor 170
    dirDataSet = "../dataset/63_Obl_Ax_160_LPI_fusion"
    if(CONTROLTYPE=='CCLR')or(CONTROLTYPE=="CLRM"):
        imgCountNO_0 = 66
        imgCountNO_7 = 66
        imgCountNO_4 = 28
    elif(CONTROLTYPE=='CZLR'):
        imgCountNO_0 = 160
        imgCountNO_7 = 0
        imgCountNO_4 = 0 
    imgCountYES_1 = 100
    imgCountYES_2 = 60
    imgCountYES_3 = 0
    imgCountYES = imgCountYES_1 + imgCountYES_2 + imgCountYES_3  
    imgCountNO = imgCountNO_0 + imgCountNO_4 + imgCountNO_7  
    
elif(DATATYPE == 67):  ## Multi_class_Oblcor 170
    dirDataSet = "../dataset/67_transformer_Obl"
    imgCountNO_0 = 66
    imgCountNO_7 = 66
    imgCountNO_4 = 28
    imgCountYES_1 = 100
    imgCountYES_2 = 60
    imgCountYES_3 = 0
    imgCountYES = imgCountYES_1 + imgCountYES_2 + imgCountYES_3  
    imgCountNO = imgCountNO_0 + imgCountNO_4 + imgCountNO_7   
    imgRow = 160 ; imgCol = 200 ; imgDepth = 168

    
elif(DATATYPE == 40):  ## Multi_class_Oblcor 170
    dirDataSet = "../dataset/40_new_data_Obl_194EA"
    if(CONTROLTYPE=='CCLR')or(CONTROLTYPE=="CLRM"):
        imgCountNO_0 = 83
        imgCountNO_7 = 83
        imgCountNO_4 = 28
    imgCountYES_1 = 110
    imgCountYES_2 = 84
    imgCountYES_3 = 0
    imgCountYES = imgCountYES_1 + imgCountYES_2 + imgCountYES_3  
    imgCountNO = imgCountNO_0 + imgCountNO_4 + imgCountNO_7  

elif(DATATYPE == 42):  ## Multi_class_Oblcor 170
    dirDataSet = "../dataset/42_new_data_Ax_216EA"
    if(CONTROLTYPE=='CCLR')or(CONTROLTYPE=="CLRM"):
        imgCountNO_0 = 94
        imgCountNO_7 = 94
        imgCountNO_4 = 28
    imgCountYES_1 = 120
    imgCountYES_2 = 96
    imgCountYES_3 = 0
    imgCountYES = imgCountYES_1 + imgCountYES_2 + imgCountYES_3  
    imgCountNO = imgCountNO_0 + imgCountNO_4 + imgCountNO_7  


    
elif(DATATYPE == 64): ## Multi Mode debug mode
    dirDataSet = "../dataset/64_CCLR_CZLR_debug_20EAM"
    if(CONTROLTYPE=='CCLR')or(CONTROLTYPE=='CLRM'):
        imgCountNO_0 = 10
        imgCountNO_7 = 5
        imgCountNO_4 = 5
    elif(CONTROLTYPE=='CZLR'):
        imgCountNO_0 = 20
        imgCountNO_7 = 0
        imgCountNO_4 = 0
    imgCountYES_1 = 11
    imgCountYES_2 = 9
    imgCountYES_3 = 0
    imgCountYES = imgCountYES_1 + imgCountYES_2 + imgCountYES_3  
    imgCountNO = imgCountNO_0 + imgCountNO_4 + imgCountNO_7  

tstDataSet = "../dataset/75_add_new4_from71"
tstCountNO_0 =  168 ; tstCountNO_7 = 69 ; tstCountNO_4 = 15
tstCountYES_1 = 25 ; tstCountYES_2 = 25 ; tstCountYES_3 = 0
tstCountYES = tstCountYES_1 + tstCountYES_2 + tstCountYES_3  
tstCountNO = tstCountNO_0 + tstCountNO_4 + tstCountNO_7  

'''
imgCountNO_0 = tstCountNO_0
imgCountNO_4 = tstCountNO_4
imgCountNO_7 = tstCountNO_7 
imgCountYES_1 = tstCountYES_1
imgCountYES_2 = tstCountYES_2
imgCountYES = imgCountYES_1 + imgCountYES_2 + imgCountYES_3  
imgCountNO = imgCountNO_0 + imgCountNO_4 + imgCountNO_7  
'''

if(CONTROLTYPE == 'CLRM'):
    categories = ["C047_no","C1_left_yes","C2_right_yes"]

nb_classes = len(categories)
imgCount = imgCountNO + imgCountYES
tstCount = tstCountNO + tstCountYES