import gc
import time
import random
import pickle
import os, glob
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf

import keras
from keras import optimizers
from keras import initializers
from keras import regularizers
from keras.regularizers import l2

import keras.backend
import keras.layers
from keras.layers import Conv2D, MaxPooling2D, Conv3D, MaxPool3D, GlobalAvgPool3D
from keras.layers import Input,Dense, Dropout, Activation, BatchNormalization, Flatten
from keras.layers import Average,PReLU, LeakyReLU

import keras.models
from keras.models import load_model
from keras.models import Sequential
from keras.models import Model
from keras.models import model_from_json

import keras.utils
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint
from keras.preprocessing import image

import scipy.misc
import scipy.io as io
import nibabel as nib
import gzip
import math

# Confusion Matrix
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()

from Args.argument import get_args

args = get_args()
AUG = args.AUG
SETT = args.SETT
TRIAL = args.TRIAL
BATCH = args.BATCH
DATATYPE = args.DATATYPE
CONVOLUTION_TYPE = args.MODEL
KERNEL_SEED = args.KERNEL_SEED
CONTROLTYPE  =  args.CONTROLTYPE #CZ or C7 or CC or CCM or CCL
LRPMSG = args.LRPMSG
PERCENT = args.PERCENT
FOLD_SEED = args.FOLD_SEED
LOSS1 = args.LOSS1
LOSS2 = args.LOSS2
BALANCE = args.BALANCE
TESTEPOCH = args.TEST_EPOCH
fc1 = args.FC1
fc2 = args.FC2
MODE = args.EnsMODE
ruleLRP = args.RULE

# Seed
np.random.seed(KERNEL_SEED)
random.seed(KERNEL_SEED)
        
#operation option
DEBUG = True
STUDYMODE = True    #True is saving weight
ENSEMBLE_MODEL = False

#global Param
TRAIN_MAX = 1
nb_batchSize = BATCH  #default 8 
nb_KFold = 5    #default 5
verboseType = 1 # 0 = silent, 1 = progress bar, 2 = one line epoch.
droprate=0.5
Param_regularizers = 1e-5       #default 1e-5
Param_regularizersDense = 1e-5   #default 1e-5

MULTI_CHECK = (CONTROLTYPE=='ZLRM')or(CONTROLTYPE=='CLRM')
LEARNING_MODE = 'FAST' # default MID, or FAST, FINETUNE
learning_rate = 1e-2   # MID : 1e-5  FAST : 1e-2  FINETUNE : learning_rate setting is "1e-3"
loop_epochs = 4 
nb_epochs = 75
MODELNUMBERING = 20
modelNum = (DATATYPE*100) + MODELNUMBERING
decay_rate = learning_rate / nb_epochs
adam=optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=decay_rate, amsgrad=False)

# Ensemble
ENSEMBLE_MODE = MODE  #AVR or VOT(average, voting)
ENSEMBLE_NUM = 5 # 5 or 10 or 15
IMAGE_CLASS = 1 # T2_AX, T2_OBLCOR, Flair_OBLCOR
ENSEMBLE_IMG = 1

# LRP
squareValue = 0.1
backgraoundPixeloffset = 15

if(TRIAL==9):
    loop_epochs = 1
    nb_epochs = 3

##====================== DATATYPE  ========================##

# 01 : T2_oblcor_voxel3m, 02 : T2_ax_voxel3m, 03 : flair_voxel3m
# 11 : T2_oblcor_voxel1m, 12 : T2_ax_voxel1m, 13 : flair_voxel1m
# 14 : T2_ax_voxel1m_135EA, 24 : T2_ax_voxel1m_240EA
# 22 : T2_ax resize MNI space, 25 : T2_ax_left_HS
# 32 : T2_ax Sub-Cortical Structure 
# 42 : T2_only Hippocampal position 
# 9x ~ : 2D image


#default imgCountNO, imgCountYES
imgCountNO = 140; imgCountYES= 140

## ========================================================[1](19.09.05)

if(DATATYPE == 11):
    dirDataSet = "./dataset/11_T2_oblcor_v1m"
elif(DATATYPE == 12):
    dirDataSet = "./dataset/12_T2_ax_v1m"
elif(DATATYPE == 13):
    dirDataSet = "./dataset/13_flair_v1m"

## for Ax 240EA
elif(DATATYPE == 14): # resize Same demension 197x233x189
    dirDataSet = "./dataset/22_T2_ax_240EA"
    imgCountNO = 240;    imgCountYES= 240 
elif(DATATYPE == 15): # resize Same demension 197x233x189
    dirDataSet = "./dataset/22_T2_ax_240EA"
    imgCountNO = 240;    imgCountYES= 240 

## Original Data Set
elif(DATATYPE == 21): ##resize MNI space 160x200x170(before 197x233x189)
    dirDataSet = "./dataset/21_T2_oblcor_185EA"
    imgCountNO = 185;    imgCountYES= 185
elif(DATATYPE == 22): ##resize MNI space 160x200x170(before 197x233x189)
    dirDataSet = "./dataset/22_T2_ax_240EA"
    imgCountNO = 240;    imgCountYES= 240 
elif(DATATYPE == 23): ##resize MNI space 160x200x170(before 197x233x189)
    dirDataSet = "./dataset/23_flair_180EA"
    imgCountNO = 180;    imgCountYES= 180

    
## Multi Mode
elif(DATATYPE == 30): ## Ax_multi class mode
    dirDataSet = "./dataset/30_CCM_CCL_CAT_ax_148EAM"
    imgCountNO_0 = 61
    imgCountNO_7 = 62
    imgCountNO_4 = 25
    imgCountYES_1 = 93
    imgCountYES_2 = 55
    imgCountYES_3 = 0
    imgCountYES = imgCountYES_1 + imgCountYES_2 + imgCountYES_3  
    imgCountNO = imgCountNO_0 + imgCountNO_4 + imgCountNO_7  

## Multi Mode
elif(DATATYPE == 31): ## Ax_multi class mode
    dirDataSet = "./dataset/31_CCM_CCL_CAT_oblcor_148EAM"
    imgCountNO_0 = 61
    imgCountNO_7 = 62
    imgCountNO_4 = 25
    imgCountYES_1 = 93
    imgCountYES_2 = 55
    imgCountYES_3 = 0
    imgCountYES = imgCountYES_1 + imgCountYES_2 + imgCountYES_3  
    imgCountNO = imgCountNO_0 + imgCountNO_4 + imgCountNO_7  
    imgCountYES = imgCountYES_1 + imgCountYES_2 + imgCountYES_3  

## Debug Mode
elif(DATATYPE == 39): ## Ax_multi class mode
    dirDataSet = "./dataset/39_Debug"
    imgCountNO_0 = 7
    imgCountNO_7 = 7
    imgCountNO_4 = 6
    imgCountYES_1 = 10
    imgCountYES_2 = 10
    imgCountYES_3 = 0
    imgCountYES = imgCountYES_1 + imgCountYES_2 + imgCountYES_3  
    imgCountNO = imgCountNO_0 + imgCountNO_4 + imgCountNO_7  
    imgCountYES = imgCountYES_1 + imgCountYES_2 + imgCountYES_3  

## Multi Mode
elif(DATATYPE == 40): ## Obl_New_Data
    dirDataSet = "./dataset/40_new_data_Obl_198EA"
    if(CONTROLTYPE=='ZLRM'):
        imgCountNO_0 = 198 
        imgCountNO_7 = 0 # 103
        imgCountNO_4 = 0 # 247
    elif(CONTROLTYPE=='CLRM'):
        imgCountNO_0 = 85 
        imgCountNO_7 = 85 # 103
        imgCountNO_4 = 28 # 247
    imgCountYES_1 = 114
    imgCountYES_2 = 84
    imgCountYES_3 = 0
    imgCountYES = imgCountYES_1 + imgCountYES_2 + imgCountYES_3  
    imgCountNO = imgCountNO_0 + imgCountNO_4 + imgCountNO_7  
    
elif(DATATYPE == 41): ## Obl_New_Data
    dirDataSet = "./dataset/41_new_data_Obl_SAME_183EA"
    if(CONTROLTYPE=='ZLRM'):
        imgCountNO_0 = 183 
        imgCountNO_7 = 0
        imgCountNO_4 = 0
    elif(CONTROLTYPE=='CLRM'):
        imgCountNO_0 = 81 
        imgCountNO_7 = 81
        imgCountNO_4 = 21
    imgCountYES_1 = 103
    imgCountYES_2 = 80
    imgCountYES_3 = 0
    imgCountYES = imgCountYES_1 + imgCountYES_2 + imgCountYES_3  
    imgCountNO = imgCountNO_0 + imgCountNO_4 + imgCountNO_7  
    
elif(DATATYPE == 42): ## Obl_New_Data
    dirDataSet = "./dataset/42_new_data_Ax_220EA"
    if(CONTROLTYPE=='ZLRM'):
        imgCountNO_0 = 220 
        imgCountNO_7 = 0
        imgCountNO_4 = 0
    elif(CONTROLTYPE=='CLRM'):
        imgCountNO_0 = 96 
        imgCountNO_7 = 96
        imgCountNO_4 = 28
    imgCountYES_1 = 124
    imgCountYES_2 = 96
    imgCountYES_3 = 0
    imgCountYES = imgCountYES_1 + imgCountYES_2 + imgCountYES_3  
    imgCountNO = imgCountNO_0 + imgCountNO_4 + imgCountNO_7  

elif(DATATYPE == 43): ## Obl_New_Data
    dirDataSet = "./dataset/43_new_data_Ax_SAME_183EA"
    if(CONTROLTYPE=='ZLRM'):
        imgCountNO_0 = 183 
        imgCountNO_7 = 0
        imgCountNO_4 = 0
    elif(CONTROLTYPE=='CLRM'):
        imgCountNO_0 = 81 
        imgCountNO_7 = 81
        imgCountNO_4 = 21
    imgCountYES_1 = 103
    imgCountYES_2 = 80
    imgCountYES_3 = 0
    imgCountYES = imgCountYES_1 + imgCountYES_2 + imgCountYES_3  
    imgCountNO = imgCountNO_0 + imgCountNO_4 + imgCountNO_7  
    
elif(DATATYPE == 45): ## Obl_New_Data
    dirDataSet = "./dataset/45_FIV_Ax_202EA"
    if(CONTROLTYPE=='ZLRM'):
        imgCountNO_0 = 202
        imgCountNO_7 = 0
        imgCountNO_4 = 0
    elif(CONTROLTYPE=='CLRM'):
        imgCountNO_0 = 88 
        imgCountNO_7 = 88
        imgCountNO_4 = 26
    imgCountYES_1 = 112
    imgCountYES_2 = 90
    imgCountYES_3 = 0
    imgCountYES = imgCountYES_1 + imgCountYES_2 + imgCountYES_3  
    imgCountNO = imgCountNO_0 + imgCountNO_4 + imgCountNO_7  
    
elif(DATATYPE == 46): ## Obl_New_Data_MNI5
    dirDataSet = "./dataset/46_FIV_Obl_202EA"
    if(CONTROLTYPE=='ZLRM'):
        imgCountNO_0 = 202
        imgCountNO_7 = 0
        imgCountNO_4 = 0
    elif(CONTROLTYPE=='CLRM'):
        imgCountNO_0 = 88
        imgCountNO_7 = 88
        imgCountNO_4 = 26
    imgCountYES_1 = 112
    imgCountYES_2 = 90
    imgCountYES_3 = 0
    imgCountYES = imgCountYES_1 + imgCountYES_2 + imgCountYES_3  
    imgCountNO = imgCountNO_0 + imgCountNO_4 + imgCountNO_7  
    
elif(DATATYPE == 47): ## Ax_New_Data_MNI5
    dirDataSet = "./dataset/47_FIV_Ax_202EA"
    if(CONTROLTYPE=='ZLRM'):
        imgCountNO_0 = 202
        imgCountNO_7 = 0
        imgCountNO_4 = 0
    elif(CONTROLTYPE=='CLRM'):
        imgCountNO_0 = 88
        imgCountNO_7 = 88
        imgCountNO_4 = 26
    imgCountYES_1 = 112
    imgCountYES_2 = 90
    imgCountYES_3 = 0
    imgCountYES = imgCountYES_1 + imgCountYES_2 + imgCountYES_3  
    imgCountNO = imgCountNO_0 + imgCountNO_4 + imgCountNO_7  

## Multi Mode
elif(DATATYPE == 51): ## Ax_multi class mode
    dirDataSet = "./dataset/51_CCM_CCL_CAT_T2_ax_230EAM"
    imgCountNO_0 = 98
    imgCountNO_7 = 99
    imgCountNO_4 = 33
    imgCountYES_1 = 128
    imgCountYES_2 = 90
    imgCountYES_3 = 12
    imgCountYES = imgCountYES_1 + imgCountYES_2 + imgCountYES_3  
    imgCountNO = imgCountNO_0 + imgCountNO_4 + imgCountNO_7  

    imgCountYES = imgCountYES_1 + imgCountYES_2 + imgCountYES_3  
elif(DATATYPE == 52): ## Obl_multi class mode
    dirDataSet = "./dataset/52_CCM_CCL_CAT_oblcor_160EAM"
    imgCountNO_0 = 67
    imgCountNO_7 = 68
    imgCountNO_4 = 25
    imgCountYES_1 = 93
    imgCountYES_2 = 55
    imgCountYES_3 = 12
    imgCountYES = imgCountYES_1 + imgCountYES_2 + imgCountYES_3  
    imgCountNO = imgCountNO_0 + imgCountNO_4 + imgCountNO_7  
## conCATenate
elif(DATATYPE == 53):
    dirDataSet = "./dataset/53_CC_ax_240EA"
    imgCountNO = 240;    imgCountYES = 240
elif(DATATYPE == 54):
    dirDataSet = "./dataset/54_CC_oblcor_170EA"
    imgCountNO = 170;    imgCountYES = 170
elif(DATATYPE == 55):
    dirDataSet = "./dataset/55_CCM_CCL_flair_161EAM"
    imgCountNO = 161;    imgCountYES = 161
    imgCountYES_1 = 85
    imgCountYES_2 = 65
    imgCountYES_3 = 11
    imgCountYES = imgCountYES_1 + imgCountYES_2 + imgCountYES_3  

elif(DATATYPE == 56):  ## Ax_multi_class mode 160
    dirDataSet = "./dataset/56_CCM_CCL_CAT_ax_160EAM"
    imgCountNO_0 = 67
    imgCountNO_7 = 68
    imgCountNO_4 = 25
    imgCountYES_1 = 93
    imgCountYES_2 = 55
    imgCountYES_3 = 12
    imgCountYES = imgCountYES_1 + imgCountYES_2 + imgCountYES_3  
    imgCountNO = imgCountNO_0 + imgCountNO_4 + imgCountNO_7  
    
elif(DATATYPE == 57):  ## Multi_class_Oblcor 170
    dirDataSet = "./dataset/57_CCM_CCL_oblcor_170EAM"
    imgCountNO_0 = 71
    imgCountNO_7 = 71
    imgCountNO_4 = 28
    imgCountYES_1 = 98
    imgCountYES_2 = 60
    imgCountYES_3 = 12
    imgCountYES = imgCountYES_1 + imgCountYES_2 + imgCountYES_3  
    imgCountNO = imgCountNO_0 + imgCountNO_4 + imgCountNO_7  

elif(DATATYPE == 58):  ## Multi_class_Oblcor 170
    dirDataSet = "./dataset/58_CCLR_CZLR_oblcor_160EAM"
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
    
elif(DATATYPE == 59):  ## Multi_class_Oblcor 170
    dirDataSet = "./dataset/59_CCLR_ax_218EAM"
    imgCountNO_0 = 92
    imgCountNO_7 = 93
    imgCountNO_4 = 33
    imgCountYES_1 = 128
    imgCountYES_2 = 90
    imgCountYES_3 = 0
    imgCountYES = imgCountYES_1 + imgCountYES_2 + imgCountYES_3  
    imgCountNO = imgCountNO_0 + imgCountNO_4 + imgCountNO_7  


elif(DATATYPE == 60):  ## Multi_class_Oblcor 170
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
    dirDataSet = "./dataset/61_Ax_160_LPI_160EAM"
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
    
elif(DATATYPE == 62): # CAT Oblcor debug mode
    dirDataSet = "./dataset/62_CC_debug_5EA"
    imgCountNO = 5 ; imgCountYES = 5
    
elif(DATATYPE == 63): ## Multi Mode debug mode
    dirDataSet = "./dataset/63_CCM__debug_20EAM"
    imgCountNO_0 = 10
    imgCountNO_7 = 5
    imgCountNO_4 = 5
    imgCountYES_1 = 10
    imgCountYES_2 = 5
    imgCountYES_3 = 5
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
    imgCountYES_1 = 10
    imgCountYES_2 = 10
    imgCountYES_3 = 0
    imgCountYES = imgCountYES_1 + imgCountYES_2 + imgCountYES_3  
    imgCountNO = imgCountNO_0 + imgCountNO_4 + imgCountNO_7  

elif(DATATYPE == 65): ## Multi Mode debug mode
    dirDataSet = "./dataset/65_Obl_160_LPI_no4_160EAM"
    if(CONTROLTYPE=='CCLR')or(CONTROLTYPE=='CLRM'):
        imgCountNO_0 = 80
        imgCountNO_7 = 80
        imgCountNO_4 = 0
    imgCountYES_1 = 100
    imgCountYES_2 = 60
    imgCountYES_3 = 0
    imgCountYES = imgCountYES_1 + imgCountYES_2 + imgCountYES_3  
    imgCountNO = imgCountNO_0 + imgCountNO_4 + imgCountNO_7  
    
elif(DATATYPE == 66): ## Multi Mode debug mode
    dirDataSet = "./dataset/66_Obl_160_LPI_more4"
    if(CONTROLTYPE=='CCLR')or(CONTROLTYPE=='CLRM'):
        imgCountNO_0 = 63
        imgCountNO_7 = 63
        imgCountNO_4 = 34
    imgCountYES_1 = 100
    imgCountYES_2 = 60
    imgCountYES_3 = 0
    imgCountYES = imgCountYES_1 + imgCountYES_2 + imgCountYES_3  
    imgCountNO = imgCountNO_0 + imgCountNO_4 + imgCountNO_7  

elif(DATATYPE == 70): ## Multi Mode debug mode
    dirDataSet = "./dataset/70_test"
    imgCountNO_0 =  182
    imgCountNO_7 = 83
    imgCountNO_4 = 6
    imgCountYES_1 = 25
    imgCountYES_2 = 25
    imgCountYES_3 = 0
    imgCountYES = imgCountYES_1 + imgCountYES_2 + imgCountYES_3  
    imgCountNO = imgCountNO_0 + imgCountNO_4 + imgCountNO_7  
elif(DATATYPE == 71): ## Multi Mode debug mode
    dirDataSet = "./dataset/71_test_no4"
    imgCountNO_0 =  168
    imgCountNO_7 = 69
    imgCountNO_4 = 0
    imgCountYES_1 = 25
    imgCountYES_2 = 25
    imgCountYES_3 = 0
    imgCountYES = imgCountYES_1 + imgCountYES_2 + imgCountYES_3  
    imgCountNO = imgCountNO_0 + imgCountNO_4 + imgCountNO_7  
elif(DATATYPE == 72): ## Multi Mode debug mode
    dirDataSet = "./dataset/72_test_only_new4"
    imgCountNO_0 =  182
    imgCountNO_7 = 83
    imgCountNO_4 = 11
    imgCountYES_1 = 25
    imgCountYES_2 = 25
    imgCountYES_3 = 0
    imgCountYES = imgCountYES_1 + imgCountYES_2 + imgCountYES_3  
    imgCountNO = imgCountNO_0 + imgCountNO_4 + imgCountNO_7  
elif(DATATYPE == 73): ## Multi Mode debug mode
    dirDataSet = "./dataset/73_add_new4"
    imgCountNO_0 =  182
    imgCountNO_7 = 83
    imgCountNO_4 = 17
    imgCountYES_1 = 25
    imgCountYES_2 = 25
    imgCountYES_3 = 0
    imgCountYES = imgCountYES_1 + imgCountYES_2 + imgCountYES_3  
    imgCountNO = imgCountNO_0 + imgCountNO_4 + imgCountNO_7  
elif(DATATYPE == 74): ## Multi Mode debug mode
    dirDataSet = "./dataset/74_add_only_new4_from71"
    imgCountNO_0 =  168
    imgCountNO_7 = 69
    imgCountNO_4 = 9
    imgCountYES_1 = 25
    imgCountYES_2 = 25
    imgCountYES_3 = 0
    imgCountYES = imgCountYES_1 + imgCountYES_2 + imgCountYES_3  
    imgCountNO = imgCountNO_0 + imgCountNO_4 + imgCountNO_7  
elif(DATATYPE == 75): ## Multi Mode debug mode
    dirDataSet = "./dataset/75_add_new4_from71"
    imgCountNO_0 =  168
    imgCountNO_7 = 69
    imgCountNO_4 = 15
    imgCountYES_1 = 25
    imgCountYES_2 = 25
    imgCountYES_3 = 0
    imgCountYES = imgCountYES_1 + imgCountYES_2 + imgCountYES_3  
    imgCountNO = imgCountNO_0 + imgCountNO_4 + imgCountNO_7  
elif(DATATYPE == 78): ## Multi Mode debug mode
    dirDataSet = "./dataset/78_check"
    imgCountNO_0 =  182
    imgCountNO_7 = 30
    imgCountNO_4 = 3
    imgCountYES_1 = 25
    imgCountYES_2 = 25
    imgCountYES_3 = 0
    imgCountYES = imgCountYES_1 + imgCountYES_2 + imgCountYES_3  
    imgCountNO = imgCountNO_0 + imgCountNO_4 + imgCountNO_7  
elif(DATATYPE == 80): ## LRP_CHECK
    dirDataSet = "./dataset/80_LRP_cv"
    imgCountNO_0 =  10
    imgCountNO_7 = 0
    imgCountNO_4 = 0
    imgCountYES_1 = 11
    imgCountYES_2 = 19
    imgCountYES_3 = 0
    imgCountYES = imgCountYES_1 + imgCountYES_2 + imgCountYES_3  
    imgCountNO = imgCountNO_0 + imgCountNO_4 + imgCountNO_7  
elif(DATATYPE == 81): ## LRP_CHECK
    dirDataSet = "./dataset/81_LRP_test"
    imgCountNO_0 =  6
    imgCountNO_7 = 0
    imgCountNO_4 = 0
    imgCountYES_1 = 10
    imgCountYES_2 = 5
    imgCountYES_3 = 0
    imgCountYES = imgCountYES_1 + imgCountYES_2 + imgCountYES_3  
    imgCountNO = imgCountNO_0 + imgCountNO_4 + imgCountNO_7  
    
    
if(DATATYPE <= 10):
    imgRow = 65
    imgCol = 77
    imgDepth = 63
elif((DATATYPE >= 11) & (DATATYPE <= 81)) :
    if(SETT=='FIV'):
        imgRow = 290
        imgCol = 370
        imgDepth = 320
        imgRow = 160
        imgCol = 200
        imgDepth = 170
    else:
        imgRow = 160
        imgCol = 200
        imgDepth = 170
        imgDepthCAT = 340
        imgDepthCAT2 = 510
elif(DATATYPE <= 90):
    imgRow = 197
    imgCol = 233
    imgDepth = 189
else:
    imgRow = 160
    imgCol = 210
    imgDepth = 1
    

if(DATATYPE >= 90):
    # categories = ["Cat","Dog","Ship","Airplane","Frog"]
    categories = ["W11_NG","W11_OK"]
else:
    if(CONTROLTYPE  == 'C7'):
        categories = ["C7_no","C123_yes"]
    elif(CONTROLTYPE  == 'CC'):
        categories = ["C047_no","C123_yes"]
    elif(CONTROLTYPE  == 'CCM'):
        if(30<=TRIAL<40):
            categories = ["C047_no","C1_left_yes","C2_right_yes"]
        else:
            categories = ["C047_no","C1_left_yes","C2_right_yes","C3_bi_yes"]
    elif(CONTROLTYPE == 'ZLRM'):
        categories = ["C0_no","C1_left_yes","C2_right_yes"]
    elif(CONTROLTYPE == 'CLRM'):
        categories = ["C047_no","C1_left_yes","C2_right_yes"]
    elif(CONTROLTYPE  == 'CZM'):
        categories = ["C0_no","C1_left_yes","C2_right_yes","C3_bi_yes"]
    elif(CONTROLTYPE  == 'CCL'):
        categories = ["C047_no","C1_left_yes","C2_right_yes","C3_bi_yes"]
    elif(CONTROLTYPE == 'CCLR'):
        categories = ["C047_no","C1_left_yes","C2_right_yes"]
    elif(CONTROLTYPE == 'CZLR'):
        categories = ["C0_no","C1_left_yes","C2_right_yes"]
    elif(CONTROLTYPE == 'PRED'):
        categories = ["C047_no","C1_left_yes","C2_right_yes","Unknown"]
        

    else: #CZ
        categories = ["C0_no","C123_yes"]      
        
nb_classes = len(categories)
imgCount = imgCountNO + imgCountYES

plt.rc('font',size=15)
plt.rc('axes',titlesize=15)
plt.rc('axes',labelsize=15)
plt.rc('xtick',labelsize=15)
plt.rc('ytick',labelsize=15)
plt.rc('legend',fontsize=13)


def average(values):
    if len(values) == 0:
      return None
    return sum(values, 0.0) / len(values)

def timeCheck():
    now = time.localtime()
    currentTime = "%04d-%02d-%02d %02d:%02d:%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
    return currentTime

def debugMessage():
    if DEBUG:
        # Define a function, a class or do some crazy stuff
        print('The source is in debug mode')
    else:
        print('The source is not in debug mode')

    if STUDYMODE:
        print('The source is in Training mode')
    else:
        print('The source is in Prediction mode')
    return


 