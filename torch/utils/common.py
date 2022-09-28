import os
import time
import sys
sys.path.insert(0,'..')

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
OPT = args.OPT
SCH = args.SCH
MOM = args.MOM
SETT = args.SETT
TALK = args.TALK
TRIAL = args.TRIAL
MODEL = args.MODEL
BATCH = args.BATCH
DATATYPE = args.DATATYPE
ENDEPOCH = args.ENDEPOCH
FULEPOCH = args.FULEPOCH
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
lr = args.lr
lreduce = args.lreduce
lrperiod = args.lrperiod
wdecay = args.wdecay
ksize = args.ksize
start = time.time()

# Experiment Initialization
nb_KFold = 5 ; iters=100
drop_rate = 0.5 ; learning_rate = lr ; weight_regularizer = 1e-5
imgRow = 160 ; imgCol = 200 ; imgDepth = 170
device = 'cuda' if torch.cuda.is_available() else 'cpu'

FOLD_SEED = 0
KFOLD = StratifiedKFold(n_splits=nb_KFold, random_state=FOLD_SEED, shuffle=True)
FOLD_SEED = FOLD_SEED + 1


#decay_rate = learning_rate / step_size
criterion = nn.CrossEntropyLoss()

if(EnsMODE=='AVR')or(EnsMODE=='VOT'):
    graph_path = '/graph/T%d%s_K' %(TRIAL,EnsMODE) + str(K)
    KERNELS = K
    ENSEMBLE_NUM = len(KERNELS) # 5 or 10 or 15
else:
    graph_path = '/graph/T%dK%d' %(TRIAL,KERNEL_SEED)


if(not(os.path.isdir(os.getcwd()+graph_path))):
    os.mkdir(os.getcwd()+graph_path)

    
## Multi Mode
if(DATATYPE == 60):  ## Multi_class_Oblcor 170
    dirDataSet = "../dataset/60_Obl_160_LPI_160EAM"
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
    imgCountNO_0 = 66
    imgCountNO_7 = 66
    imgCountNO_4 = 28
    imgCountYES_1 = 100
    imgCountYES_2 = 60
    imgCountYES_3 = 0
    imgCountYES = imgCountYES_1 + imgCountYES_2 + imgCountYES_3  
    imgCountNO = imgCountNO_0 + imgCountNO_4 + imgCountNO_7  

elif(DATATYPE == 62):  ## Multi_class_Oblcor 170
    dirDataSet = "../dataset/62_Sum_320_LPI_320EAM"
    imgCountNO_0 = 132
    imgCountNO_7 = 132
    imgCountNO_4 = 56
    imgCountYES_1 = 200
    imgCountYES_2 = 120
    imgCountYES_3 = 0
    imgCountYES = imgCountYES_1 + imgCountYES_2 + imgCountYES_3  
    imgCountNO = imgCountNO_0 + imgCountNO_4 + imgCountNO_7  

elif(DATATYPE == 64): ## Multi Mode debug mode
    dirDataSet = "../dataset/64_debug"
    if(CONTROLTYPE=='CCLR')or(CONTROLTYPE=='CLRM'):
        imgCountNO_0 = 10
        imgCountNO_7 = 5
        imgCountNO_4 = 5
    elif(CONTROLTYPE=='CZLR'):
        imgCountNO_0 = 20
        imgCountNO_7 = 0
        imgCountNO_4 = 0
    imgCountYES_1 = 10
    imgCountYES_2 = 10
    imgCountYES_3 = 0
    imgCountYES = imgCountYES_1 + imgCountYES_2 + imgCountYES_3  
    imgCountNO = imgCountNO_0 + imgCountNO_4 + imgCountNO_7  
    
tstDataSet = "../dataset/75_add_new4_from71"
tstCountNO_0 =  168 ; tstCountNO_7 = 69 ; tstCountNO_4 = 15
tstCountYES_1 = 25 ; tstCountYES_2 = 25 ; tstCountYES_3 = 0
tstCountYES = tstCountYES_1 + tstCountYES_2 + tstCountYES_3  
tstCountNO = tstCountNO_0 + tstCountNO_4 + tstCountNO_7  

if(CONTROLTYPE == "CLRM"): categories = ["C047_no","C1_left_yes","C2_right_yes"]
elif(CONTROLTYPE == "ADNI"): categories = ["ADNI_CN","ADNI_AD","ADNI_MCI"]

nb_classes = len(categories)
imgCount = imgCountNO + imgCountYES
tstCount = tstCountNO + tstCountYES

if(debug):
    ENDEPOCH=4
    FULEPOCH=2
    BATCH = 4
