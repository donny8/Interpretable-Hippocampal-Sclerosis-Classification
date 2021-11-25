import os
import time
import glob
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import scipy.misc
import scipy.io as io
from sklearn.manifold import TSNE
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from Args.argument import get_args

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F



args = get_args()
KERNEL_SEED = args.seed
layer_of_interest=args.interest
perp = args.perp
epsi = args.epsi
step = args.step
select = args.select
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

l_width = 0.2
p_size = 50
nb_classes = 3


num1=0 ; num2=0 ; num3=50
SETT = 'FUL' ; AUG = 'hflip' ; CONTROLTYPE = 'CLRM'
TRIAL = 27 ; loadmodelNum = 6020 ; FOLD_SEED = 1 ; iters = 100 ; DATATYPE = 60 ; drop_rate = 0.5
imgCountNO_0 =  168 ; imgCountNO_7 = 69 ; imgCountNO_4 = 15
imgCountYES_1 = 25 ; imgCountYES_2 = 25 ; imgCountYES_3 = 0
imgRow = 160 ; imgCol = 200 ; imgDepth = 170
categories = ["C047_no","C1_left_yes","C2_right_yes"]
pca_dim = 200
df_color = 0
mini = 0.1



if(select=='UMAP'):
    import umap

print('\n\nKernel seed : %d\nlayer of interest : %d\nselect : %s\n' %(KERNEL_SEED, layer_of_interest, select))
    
def rotate(angle):
     ax.view_init(azim=angle)


def data_single(categories,dirDataSet):
    inputX = []; inputY = []; listFileName=[]
    for idx, f in enumerate(categories):        
        label = [ 0 for i in range(len(categories)) ]
        label[idx] = 1
        image_dir = dirDataSet + "/" + f
        print(image_dir)
        for (imagePath, dir, files) in sorted(os.walk(image_dir)):
            if(imagePath == image_dir):                
                if(imagePath == image_dir):
                    files = sorted(glob.glob(imagePath + "/*.nii.gz"))
                    listFileName += files
                for i, fname in enumerate(files):
                    img = nib.load(fname).get_fdata()
                    inputX.append(img)
                    inputY.append(label)
    return inputX, inputY, listFileName
        
    

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


                


def figure_single(SETT, MODE, color_arr, idxMulti,intermediates_tsne, layer_path, select,KERNEL_SEED,pca_dim,df_color,CM0,CM1,CM2,DM0,DM1,DM2):
    if(df_color):
        if(SETT=='M')or(SETT=='P'):
#            colors = ["#e50000","#ff6f52","#030aa7", "#4984b8", "#0a5f38", '#65ab7c']
            colors = [DM0,CM0,DM1,CM1,DM2,CM2]
            colors = [DM0,DM1,DM2,CM0,CM1,CM2]
    else:
        if(SETT=='M')or(SETT=='P'):
            colors = ["#d62728", "#1f77b4", '#2ca02c']

    plt.figure(figsize=(10, 10))
    for color_idx in colors:
        if(MODE=='TRAIN')or(MODE=='T&TM')or(MODE=='T&TB'):
            idx = color_arr==color_idx
        elif(MODE=='TEST'):
            idx = color_arr[idxMulti]==color_idx

        if(df_color):
            if(color_idx==DM0):
                label_col = 'Train no HS'
            elif(color_idx==CM0):
                label_col = 'Test no HS'
                
            elif(color_idx==DM1):
                if(SETT=='M')or(SETT=='P'):
                    label_col = 'Train LHS'
            elif(color_idx==CM1):
                if(SETT=='M')or(SETT=='P'):
                    label_col = 'Test LHS'
                    
            elif(color_idx==DM2):
                label_col = 'Train RHS'
            elif(color_idx==CM2):
                label_col = 'Test RHS'
                    

        if(MODE=='TRAIN')or(MODE=='T&TM')or(MODE=='T&TB'):
            if(color_idx==CM0)or(color_idx==CM1)or(color_idx==CM2): # TEST
#                plt.scatter(x = intermediates_tsne[:,0][idx], y=intermediates_tsne[:,1][idx], color= color_idx, label = label_col) #1 
#                plt.scatter(x = intermediates_tsne[:,0][idx], y=intermediates_tsne[:,1][idx], color= color_idx, label = label_col, edgecolor='black', linewidth=0.5) #2
#                plt.scatter(x = intermediates_tsne[:,0][idx], y=intermediates_tsne[:,1][idx], color= color_idx, label = label_col, marker='^' ) #3
                plt.scatter(x = intermediates_tsne[:,0][idx], y=intermediates_tsne[:,1][idx], color= color_idx, label = label_col, marker='^', edgecolor='black', linewidth=l_width, s=p_size) #4
            elif(color_idx==DM0)or(color_idx==DM1)or(color_idx==DM2): # TRAIN
                plt.scatter(x = intermediates_tsne[:,0][idx], y=intermediates_tsne[:,1][idx], color= color_idx, label = label_col, s=30)
#                plt.scatter(x = intermediates_tsne[:,0][idx], y=intermediates_tsne[:,1][idx], color= color_idx, label = label_col, edgecolor='black', linewidth=0.5) #0
            
        elif(MODE=='TEST'):
            plt.scatter(x = intermediates_tsne[idxMulti][:,0][idx], y=intermediates_tsne[idxMulti][:,1][idx], color= color_idx, label=label_col)
    if(SETT=='P'):
        saveFileName = layer_path + '/%s_%s_K%d_%s%d.png' %(select,MODE,KERNEL_SEED,SETT,pca_dim)
    else:
        saveFileName = layer_path + '/%s_%s_K%d_%s.png' %(select,MODE,KERNEL_SEED,SETT)
    plt.axis('off')
    plt.legend(prop={'size': 15})
    plt.savefig(saveFileName)
            

def traintest(switch,idxMulti,perp,epsi,step,layer_path,intermediate_tensor_function, pca_dim,df_color):
    categories = ["C047_no","C1_left_yes","C2_right_yes"]
    if(switch):
        dirDataSet = "../dataset/75_add_new4_from71"
        MODE='TEST'
        zerocut=86
    else:
        dirDataSet = "../dataset/60_Obl_160_LPI_160EAM"
        MODE='TRAIN'
        zerocut=94

    inputX, inputY, listFile = data_single(categories,dirDataSet)
    inputX = inputX[zerocut:]
    inputY = inputY[zerocut:]
    inputX = np.array(inputX)
    inputX = inputX.reshape(-1,160,200,170,1)

    intermediates = []
    color_binary = []
    color_multi = []
    for i in range(len(inputX)):
        print('\r iteration : %d' %(i), end="")
        output_class = np.argmax(inputY[i])
        if(output_class == 0):
            color_multi.append("#d62728")
            color_binary.append("#d62728")
        elif(output_class == 1):
            color_multi.append("#1f77b4")
            color_binary.append("#1f77b4")
        elif(output_class == 2):
            color_multi.append('#2ca02c')
            color_binary.append("#1f77b4")

        intermediate_tensor = intermediate_tensor_function([inputX[i].reshape(1,160,200,170,1)])[0][0]
        intermediates.append(intermediate_tensor)

    if(select=='tSNE'):
        tsne = TSNE(n_components=2, random_state=1, perplexity = perp, learning_rate=epsi,n_iter=step)
        intermediates_tsne = tsne.fit_transform(intermediates)
        if(layer_of_interest==8):
            pca = PCA(n_components = pca_dim, random_state=1)
            Principals = pca.fit_transform(intermediates)
            intermediates_PCA = tsne.fit_transform(Principals)
            intermediates_PCA = np.array(intermediates_PCA)

    elif(select=='UMAP'):
        reducer = umap.UMAP(n_components=2, random_state=KERNEL_SEED, n_neighbors=perp)
        intermediates_tsne = reducer.fit_transform(intermediates)
        if(layer_of_interest==8):
            pca = PCA(n_components = 200, random_state=1)
            Principals = pca.fit_transform(intermediates)
            intermediates_PCA = reducer.fit_transform(Principals)
            intermediates_PCA = np.array(intermediates_PCA)

    
    idxMulti = idxMulti-zerocut
    idxMulti=np.array(sorted(idxMulti))
    
    intermediates_tsne = np.array(intermediates_tsne)
    color_multi = np.array(color_multi)
    color_binary = np.array(color_binary)    
        
    figure_single('M', MODE, color_multi, idxMulti, intermediates_tsne, layer_path, select, KERNEL_SEED,pca_dim,df_color)
    figure_single('B', MODE, color_binary, idxMulti, intermediates_tsne, layer_path, select, KERNEL_SEED,pca_dim,df_color)
    if(layer_of_interest==8):
        figure_single('P', MODE, color_multi, idxMulti, intermediates_PCA, layer_path, select, KERNEL_SEED,pca_dim,df_color)
    




def painter(x,y,label,MODE,KERNEL_SEED,perp,epsi,step,MSG):
    plt.figure(figsize=(8, 8))
    color = []
    for i, k in enumerate(label):
        output_class = k
        if(output_class == 0):
            color.append("#d62728")
        elif(output_class == 1):
            color.append("#1f77b4")
        elif(output_class == 2):
            color.append('#2ca02c')
    color = np.array(color)
    for color_idx in ["#d62728", "#1f77b4", '#2ca02c']:
        idx = color==color_idx
        if(color_idx=="#d62728"):
            label_col = 'Control'
        elif(color_idx=="#1f77b4"):
            label_col = 'LHS'
        elif(color_idx=='#2ca02c'):
            label_col = 'RHS'

        plt.scatter(x[idx],y[idx],c=color_idx, label=label_col)
    saveName = './%s/K%dP%dE%dS%d/layer%d/%s3D_%s_K%d_%s.' %(select,KERNEL_SEED,perp,epsi,step,layer_of_interest,select,MODE,KERNEL_SEED,MSG)
#        plt.xlim([-400,400])
#        plt.ylim([-400,400])
    plt.axis('off')
    plt.legend(prop={'size': 15})
    plt.savefig(saveName)
        



def loader(SETT,TRIAL,AUG,KERNEL_SEED,LRMB):
    cwd = os.getcwd()
    if(LRMB!='M'):
        yT = io.loadmat(cwd+'/[%s%d%s%d]Y_true_%s.mat' %(SETT,TRIAL,AUG,KERNEL_SEED,LRMB))
        yP = io.loadmat(cwd+'/[%s%d%s%d]yPrediction_%s.mat' %(SETT,TRIAL,AUG,KERNEL_SEED,LRMB))
        yL = io.loadmat(cwd+'/[%s%d%s%d]yLabelPrediction_%s.mat' %(SETT,TRIAL,AUG,KERNEL_SEED,LRMB))

        Y_true = np.transpose(yT['mydata'])
        yPrediction = np.transpose(yP['mydata'])
        yLabelPrediction = np.transpose(yL['mydata'])
    
        return Y_true, yPrediction, yLabelPrediction
    else: 
        yT = io.loadmat(cwd+'/[%s%d%s%d]Y_true_%s.mat' %(SETT,TRIAL,AUG,KERNEL_SEED,LRMB))
        yL = io.loadmat(cwd+'/[%s%d%s%d]yLabelPrediction_%s.mat' %(SETT,TRIAL,AUG,KERNEL_SEED,LRMB))

        Y_true = np.transpose(yT['mydata'])
        yLabelPrediction = np.transpose(yL['mydata'])
    
        return Y_true, yLabelPrediction
    

    
def random_sample(idx4,idx7,idx0,idx1,idx2,num1,num2,num3,Y_true,yPrediction,yLabelPrediction,expr):
    
    if(expr=='M'):
        if(num1!=0):
            idxM4 = np.random.choice(idx4, num1//2, replace=False)
            true_NO4 = Y_true[idxM4]
            label_NO4 = yLabelPrediction[idxM4]
        if(num2!=0):
            idxM7 = np.random.choice(idx7, num2//2, replace=False)
            true_NO7 = Y_true[idxM7]
            label_NO7 = yLabelPrediction[idxM7]
            
        idxM0 = np.random.choice(idx0, num3//2, replace=False)
        idxM1 = idx1
        idxM2 = idx2
        
        true_NO0 = Y_true[idxM0]
        true_YES1 = Y_true[idxM1]
        true_YES2 = Y_true[idxM2]
        
        if(num1==0)and(num2==0):
            true = np.vstack([true_NO0,true_YES1, true_YES2])
        else:
            true = np.vstack([true_NO4,true_NO7,true_NO0,true_YES1, true_YES2])

        label_NO0 = yLabelPrediction[idxM0]
        label_YES1 = yLabelPrediction[idxM1]
        label_YES2 = yLabelPrediction[idxM2]
        
        if(num1==0)and(num2==0):
            label = np.vstack([label_NO0, label_YES1, label_YES2])
        else:
            label = np.vstack([label_NO4,label_NO7,label_NO0, label_YES1, label_YES2])

        idxM0 = np.reshape(idxM0,(-1,))
        idxM1 = np.reshape(idxM1,(-1,))
        idxM2 = np.reshape(idxM2,(-1,))
        if(num1==0)and(num2==0):
            idxs = np.concatenate([idxM0,idxM1,idxM2])
        else:
            idxM4 = np.reshape(idxM4,(-1,))
            idxM7 = np.reshape(idxM7,(-1,))
            idxs = np.concatenate([idxM4,idxM7,idxM0,idxM1,idxM2])

        
        return true, label, idxs
    
    elif(expr=='B'):
        if(num1!=0):
            idxB4 = np.random.choice(idx4, num1, replace=False)
            true_NO4 = Y_true[idxB4]
            label_NO4 = yLabelPrediction[idxB4]
            pred_NO4 = yPrediction[idxB4]
        if(num2!=0):
            idxB7 = np.random.choice(idx7, num2, replace=False)
            true_NO7 = Y_true[idxB7]
            label_NO7 = yLabelPrediction[idxB7]
            pred_NO7 = yPrediction[idxB7]
            
        idxB0 = np.random.choice(idx0, num3, replace=False)
        idxBY = np.hstack([idx1,idx2])
        
        true_NO0 = Y_true[idxB0]
        true_YES = Y_true[idxBY]
        
        if(num1==0)and(num2==0):
            true = np.vstack([true_NO0, true_YES])
        else:
            true = np.vstack([true_NO4,true_NO7,true_NO0, true_YES])
        
        label_NO0 = yLabelPrediction[idxB0]
        label_YES1 = yLabelPrediction[idxBY]
        if(num1==0)and(num2==0):
            label = np.vstack([label_NO0, label_YES1])
        else:
            label = np.vstack([label_NO4,label_NO7,label_NO0, label_YES1])

        pred_NO0 = yPrediction[idxB0]
        pred_YES = yPrediction[idxBY]
        if(num1==0)and(num2==0):
            pred = np.vstack([pred_NO0,pred_YES])
        else:
            pred = np.vstack([pred_NO4,pred_NO7,pred_NO0,pred_YES])

        idxB0 = np.reshape(idxB0,(-1,))
        idxBY = np.reshape(idxBY,(-1,))

        if(num1==0)and(num2==0):
            idxs = np.concatenate([idxB0,idxBY])
        else:
            idxB4 = np.reshape(idxB4,(-1,))
            idxB7 = np.reshape(idxB7,(-1,))
            idxs = np.concatenate([idxB4,idxB7,idxB0,idxBY])

        return true, label, pred, idxs


def balance(CONTROLTYPE,SETT,TRIAL,AUG,KERNEL_SEED,iters, C4,C7,C0,C1,C2,num1,num2,num3, RR):
    np.random.seed(KERNEL_SEED)
    Y_true_M,                yLabelPrediction_M = loader(SETT,TRIAL,AUG, KERNEL_SEED,"M")
    Y_true_B, yPrediction_B, yLabelPrediction_B = loader(SETT,TRIAL,AUG, KERNEL_SEED,"B")
    yPrediction_M = []
    
    idx4 = np.arange(C4)
    idx7 = np.arange(C7) + C4
    idx0 = np.arange(C0) + C4 + C7
    idx1 = np.arange(C1) + C4 + C7 + C0
    idx2 = np.arange(C2) + C4 + C7 + C0 + C1
    right_len = len(idx2)
    
    multi_set = []  ;  multi_perf = []
    binary_set = [] ;  binary_perf = []
    tnn_set = [] ; fpn_set = []
    tpl_set = [] ; fnl_set = []
    tpr_set = [] ; fnr_set = []
    
    roc = []
    pr = []
    sens = []
    spec = []
    b_debug = []
    f1_scores = []

    multi4 = 0; multi7 = 0; multi0 = 0
    binary4 = 0; binary7 = 0; binary0 = 0
    
    bestM=0.0 ; bestB=0.0
    iM=0 ; iB=0


    for i in range(iters):
        M_true, M_label, M_idx         = random_sample(idx4,idx7,idx0,idx1,idx2,num1,num2,num3, Y_true_M, yPrediction_M, yLabelPrediction_M,'M' )
        B_true, B_label, B_pred, B_idx = random_sample(idx4,idx7,idx0,idx1,idx2,num1,num2,num3, Y_true_B, yPrediction_B, yLabelPrediction_B,'B' )
        
        multi_class_confusion = confusion_matrix(M_true, M_label)
        tntp = np.diag(multi_class_confusion)
        correct = np.sum(tntp)
        acc = correct/len(M_true)
        multi_set.append([tntp[0],tntp[1],tntp[2]])
        multi_perf.append(acc)
        
        
        if(acc>bestM):
            bestM = acc
            iM=i
            multi_index = M_idx
        
        tn, fp, fn, tp = confusion_matrix(B_true, B_label).ravel()
        acc = (tn+tp)/(tn+fp+fn+tp)
        auc = roc_auc_score(B_true, B_pred)
        binary_set.append([tn,fp,fn,tp])
        binary_perf.append([acc,auc])
        sens.append(tp/(tp+fn))
        spec.append(tn/(tn+fp))
        f1 = f1_score(B_true, B_label)
        f1_scores.append(f1)
        
        if(acc>bestB):
            bestB = acc
            iB=i
            binary_index = B_idx

        num_yes = 2 * right_len
        BN = B_label[:num_yes]
        BL = B_label[num_yes:num_yes+right_len]
        BR = B_label[num_yes+right_len:]

        tnn = len(BN[BN==0])
        fpn = len(BN[BN==1])

        tpl = len(BL[BL==1])
        fnl = len(BL[BL==0])

        tpr = len(BR[BR==1])
        fnr = len(BR[BR==0])
    
        tnn_set.append(tnn)
        fpn_set.append(fpn)
        tpl_set.append(tpl)
        fnl_set.append(fnl)
        tpr_set.append(tpr)
        fnr_set.append(fnr)
        
        M_check = M_true==M_label
        B_check = B_true==B_label
        b_debug.append(B_check[50:])
        M_0=M_check[(num1+num2)//2:(num1+num2+num3)//2]
        B_0=B_check[num1+num2:num1+num2+num3]
        multi0+=len(M_0[M_0==True]);binary0+=len(B_0[B_0==True])
        if(num1!=0)and(num2!=0):
            M_7=M_check[num1//2:(num1+num2)//2] ; B_7=B_check[num1:num1+num2]
            multi7+=len(M_7[M_7==True]);binary7+=len(B_7[B_7==True])
            M_4=M_check[:num1//2] ; B_4=B_check[:num1]
            multi4 += len(M_4[M_4==True]); binary4 += len(B_4[B_4==True])
        
        precisions, recalls, thresholds = precision_recall_curve(B_true, B_pred)
        pr.append([precisions,recalls,thresholds])
        fpr, true_pr, thresholds2 = roc_curve(B_true, B_pred) 
        roc.append([fpr,true_pr,thresholds2])
        
    
        
    ms = multi_set
    bs = binary_set
    sens = np.array(sens)
    spec = np.array(spec)
    f1_scores = np.array(f1_scores)
    
    multi_set = np.round(np.average(multi_set,axis=0),1)
    binary_set = np.round(np.average(binary_set,axis=0),1)
    multi_perf = np.round(np.average(multi_perf,axis=0),3)
    binary_perf = np.round(np.average(binary_perf,axis=0),3)
    
    tnn = np.round(np.average(tnn_set,axis=0))
    fpn = np.round(np.average(fpn_set,axis=0))
    tpl = np.round(np.average(tpl_set,axis=0))
    fnl = np.round(np.average(fnl_set,axis=0))
    tpr = np.round(np.average(tpr_set,axis=0))
    fnr = np.round(np.average(fnr_set,axis=0))
        
    RR.write('\n\n ======== [BALANCE] ========')
    
    RR.write('\n\n[MULTI]')
    RR.write('\nNo Left Right\n')
    RR.write(str(multi_set))
    RR.write('\nacc : ')
    RR.write(str(multi_perf))
    if(num1!=0)and(num2!=0):
        RR.write('\ntrue four : %0.1f   acc : %0.3f' %(np.round(multi4/iters),multi4/(iters*num1//2)))
        RR.write('\ntrue seven : %0.1f   acc : %0.3f' %(np.round(multi7/iters),multi7/(iters*num2//2)))
    RR.write('\ntrue zero : %0.1f   acc : %0.3f' %(np.round(multi0/iters),multi0/(iters*num3//2)))
    
    RR.write('\n\n[BINARY]')
    RR.write('\ntn fp fn tp\n')
    RR.write(str(binary_set))
    RR.write('\nacc : ')
    test = (binary_set[0] + binary_set[3]) / np.sum(binary_set)
    test = np.round(test,3)
    RR.write(str(test))
    RR.write('\nauc :')
    RR.write(str(binary_perf[1]))
    if(num1!=0)and(num2!=0):
        RR.write('\ntrue four : %0.1f   acc : %0.3f' %(np.round(binary4/iters),binary4/(iters*num1)))
        RR.write('\ntrue seven : %0.1f   acc : %0.3f' %(np.round(binary7/iters),binary7/(iters*num2)))
    RR.write('\ntrue zero : %0.1f   acc : %0.3f' %(np.round(binary0/iters),binary0/(iters*num3)))
    
    RR.write('\n\nTrue Negative NO : %s' %(str(tnn)))
    RR.write('\nFalse Positive NO : %s' %(str(fpn)))
    RR.write('\nNO : %s' %(str(np.round(tnn/(tnn+fpn),3))))

    RR.write('\n\nTrue Positive Left : %s' %(str(tpl)))
    RR.write('\nFalse Negative Left : %s' %(str(fnl)))
    RR.write('\nLeft : %s' %(str(np.round(tpl/(tpl+fnl),3))))

    RR.write('\n\nTrue Positive Right : %s' %(str(tpr)))
    RR.write('\nFalse Negative Right : %s' %(str(fnr)))
    RR.write('\nRight : %s' %(str(np.round(tpr/(tpr+fnr),3))))
    
    print('Multi :',bestM,'i :',iM)
    print('Binary :',bestB,'i :',iB)
    RR.write('\nbest Multi acc : %.3f' %(bestM))

    return ms, bs, pr, roc, sens, spec, f1_scores, b_debug,multi_index,binary_index
