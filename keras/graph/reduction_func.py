import os
import glob
import scipy.misc
import scipy.io as io
import numpy as np
import pandas as pd
import nibabel as nib

import umap
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from keras.models import model_from_json
from sklearn.metrics import confusion_matrix
import keras.backend.tensorflow_backend as K

from Args.dim_reduc_arg import get_args
args = get_args()
AUG = args.AUG
SETT = args.SETT
PERP = args.PERP
EPSI = args.EPSI
STEP = args.STEP
TRIAL = args.TRIAL
SELECT = args.SELECT
KERNEL_SEED = args.SEED
CONTROLTYPE = args.CONTROLTYPE
layer_of_interest=args.INTEREST

loadmodelNum = 6020 ; FOLD_SEED = 1 ; 
imgCountNO_0 =  168 ; imgCountNO_7 = 69 ; imgCountNO_4 = 15
imgCountYES_1 = 25 ; imgCountYES_2 = 25 ; imgCountYES_3 = 0
categories = ["C047_no","C1_left_yes","C2_right_yes"]
iters = 100; mini = 0.1
num1=0 ; num2=0 ; num3=50
l_width = 0.2; p_size = 50
learning_rate = 1e-2; nb_epochs = 50
decay_rate = learning_rate / nb_epochs

# Colors
CM0 = '#fea993'; CM1 = '#a2cffe'; CM2 = '#9be5aa'    # TEST
DM0 = '#e50000'; DM1 = '#030aa7'; DM2 = '#0a5f38'    # TRAIN

dir_path = './%s' %(SELECT)
kernel_path = dir_path + '/K%dP%dE%dS%d' %(KERNEL_SEED,PERP,EPSI,STEP)
layer_path = kernel_path + '/layer%d' %(layer_of_interest)
print('\n\n\nKernel seed : %d\nlayer of interest : %d\nselect : %s\n\n\n' %(KERNEL_SEED, layer_of_interest, SELECT))
    
def rotate(angle):
     ax.view_init(azim=angle)

def DR_expr_sett():
    if(not(os.path.isdir(dir_path))):
            os.mkdir(dir_path)
    if(not(os.path.isdir(kernel_path))):
            os.mkdir(kernel_path)        
    if(not(os.path.isdir(layer_path))):
            os.mkdir(layer_path)

    modelHS = model_load(FOLD_SEED,KERNEL_SEED)
    return modelHS


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
        
    
def model_load(FOLD_SEED,KERNEL_SEED):
    ## model load ##
    modelName = '../saveModel/[%s%d%s]modelHS%s_%d{F%dK%d}.json' %(SETT,TRIAL,AUG,CONTROLTYPE,loadmodelNum,FOLD_SEED,KERNEL_SEED)
    if os.path.isfile(modelName):
        json_file = open(modelName, "r") 
        loaded_model_json = json_file.read() 
        json_file.close() 
        model = model_from_json(loaded_model_json)        
    else:
        print('\n!!!warning!!! \n load model file not exist!!')

    weightFileName = '../saveModel/[%s%d%s]HS%s_%d{F%dK%d}[0](best).h5' %(SETT,TRIAL,AUG,CONTROLTYPE,loadmodelNum,FOLD_SEED,KERNEL_SEED)
    print('weightFileName %s' %(weightFileName))
    if os.path.isfile(weightFileName):
        model.load_weights(weightFileName)
    else:
        print('\n!!!warning!!! \n load Weight file not exist!!')
    
    return model


def last_conv_features(modelHS):
    zerocut=86
    dirDataSet = "../../dataset/75_add_new4_from71"
    
    idxMulti = balance(CONTROLTYPE,SETT,TRIAL,AUG,KERNEL_SEED,iters, imgCountNO_4, imgCountNO_7,imgCountNO_0,imgCountYES_1,imgCountYES_2,num1,num2,num3)        
    idxMulti = idxMulti-zerocut
    idxMulti=np.array(sorted(idxMulti))

    inputX, inputY, listFile1 = data_single(categories,dirDataSet)
    inputX = inputX[zerocut:] ; inputY = inputY[zerocut:] ; listFile1 = listFile1[zerocut:]
    inputX = np.array(inputX) ; inputY = np.array(inputY) ; listFile1 = np.array(listFile1)
    inputX = inputX[idxMulti] ; inputY = inputY[idxMulti] ; listFile1 = listFile1[idxMulti]
    inputX = inputX.reshape(-1,160,200,170,1)

    intermediates = [] ; color_multi = []
    intermediate_tensor_function = K.function([modelHS.layers[0].input],[modelHS.layers[layer_of_interest].output])

    for i in range(len(idxMulti)):
        output_class = np.argmax(inputY[i])
        if(output_class == 0):
            color_multi.append(CM0)
        elif(output_class == 1):
            color_multi.append(CM1)
        elif(output_class == 2):
            color_multi.append(CM2)            
        intermediate_tensor = intermediate_tensor_function([inputX[i].reshape(1,160,200,170,1)])[0][0]
        intermediates.append(intermediate_tensor)

    zerocut=94
    dirDataSet = "../../dataset/60_Obl_160_LPI_160EAM"
    
    inputX, inputY, listFile2 = data_single(categories,dirDataSet)
    inputX = inputX[zerocut:] ; inputY = inputY[zerocut:] ; listFile2 = listFile2[zerocut:]
    inputX = np.array(inputX) ; listFile2 = np.array(listFile2)
    inputX = inputX.reshape(-1,160,200,170,1)
    listFile = np.hstack([listFile1,listFile2])

    for i in range(len(inputX)):
        output_class = np.argmax(inputY[i])
        if(output_class == 0):
            color_multi.append(DM0)
        elif(output_class == 1):
            color_multi.append(DM1)
        elif(output_class == 2):
            color_multi.append(DM2)
        intermediate_tensor = intermediate_tensor_function([inputX[i].reshape(1,160,200,170,1)])[0][0]
        intermediates.append(intermediate_tensor)    
    color_multi = np.array(color_multi)

    listName = []
    for i in range(len(listFile)):
        listName.append(listFile[i][-19:-14])
    listName = np.array(listName)

    return intermediates, color_multi, listName, idxMulti



            

def loader(SETT,TRIAL,AUG,KERNEL_SEED,LRMB):
    yT = io.loadmat(os.getcwd()+'/T%d/[%s%d%s%d]Y_true_%s.mat' %(TRIAL,SETT,TRIAL,AUG,KERNEL_SEED,LRMB))
    yL = io.loadmat(os.getcwd()+'/T%d/[%s%d%s%d]yLabelPrediction_%s.mat' %(TRIAL,SETT,TRIAL,AUG,KERNEL_SEED,LRMB))
    Y_true = np.transpose(yT['mydata'])
    yLabelPrediction = np.transpose(yL['mydata'])
    return Y_true, yLabelPrediction

    
def random_sample(idx4,idx7,idx0,idx1,idx2,num1,num2,num3,Y_true,yPrediction,yLabelPrediction,expr):    
    if(num1!=0):
        idxM4 = np.random.choice(idx4, num1//2, replace=False)
        true_NO4 = Y_true[idxM4]
        label_NO4 = yLabelPrediction[idxM4]
    if(num2!=0):
        idxM7 = np.random.choice(idx7, num2//2, replace=False)
        true_NO7 = Y_true[idxM7]
        label_NO7 = yLabelPrediction[idxM7]
        
    idxM0 = np.random.choice(idx0, num3//2, replace=False)
    idxM1 = idx1; idxM2 = idx2
    
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
    

def balance(CONTROLTYPE,SETT,TRIAL,AUG,KERNEL_SEED,iters, C4,C7,C0,C1,C2,num1,num2,num3):
    np.random.seed(KERNEL_SEED)
    Y_true_M,                yLabelPrediction_M = loader(SETT,TRIAL,AUG, KERNEL_SEED,"M")
    yPrediction_M = []
    
    idx4 = np.arange(C4)
    idx7 = np.arange(C7) + C4
    idx0 = np.arange(C0) + C4 + C7
    idx1 = np.arange(C1) + C4 + C7 + C0
    idx2 = np.arange(C2) + C4 + C7 + C0 + C1    

    bestM=0.0; iM=0 
    for i in range(iters):
        M_true, M_label, M_idx         = random_sample(idx4,idx7,idx0,idx1,idx2,num1,num2,num3, Y_true_M, yPrediction_M, yLabelPrediction_M,'M' )        
        multi_class_confusion = confusion_matrix(M_true, M_label)
        tntp = np.diag(multi_class_confusion)
        correct = np.sum(tntp)
        acc = correct/len(M_true)        
        if(acc>bestM):
            bestM = acc
            iM=i;  multi_index = M_idx
                    
    return multi_index

def excel_save(listName, intermediates_tsne, color_multi):
    dict_ = {'File':listName,'X':intermediates_tsne[:,0],'Y':intermediates_tsne[:,1],'color':color_multi}
    excel = pd.DataFrame(dict_)
    excel.to_csv(layer_path+'/%s_%d_result.csv' %(SELECT,KERNEL_SEED))

def figure_2D(color_multi, idxMulti, intermediates_tsne):
    colors = [DM0,DM1,DM2,CM0,CM1,CM2]

    plt.figure(figsize=(10, 10))
    for color_idx in colors:
        idx = color_multi==color_idx

        if(color_idx==DM0):
            label_col = 'Train No HS'
        elif(color_idx==CM0):
            label_col = 'Test No HS'
            
        elif(color_idx==DM1):
            label_col = 'Train Left HS'
        elif(color_idx==CM1):
            label_col = 'Test Left HS'
                
        elif(color_idx==DM2):
            label_col = 'Train Right HS'
        elif(color_idx==CM2):
            label_col = 'Test Right HS'
                    

        if(color_idx==CM0)or(color_idx==CM1)or(color_idx==CM2): # TEST
            plt.scatter(x = intermediates_tsne[:,0][idx], y=intermediates_tsne[:,1][idx], color= color_idx, label = label_col, marker='^', edgecolor='black', linewidth=l_width, s=p_size) #4
        elif(color_idx==DM0)or(color_idx==DM1)or(color_idx==DM2): # TRAIN
            plt.scatter(x = intermediates_tsne[:,0][idx], y=intermediates_tsne[:,1][idx], color= color_idx, label = label_col, s=30)
            
    saveFileName = layer_path + '/%s_K%d.png' %(SELECT,KERNEL_SEED)
    plt.axis('off')
    plt.legend(prop={'size': 15})
    plt.savefig(saveFileName)

def figure_3D(color_multi,intermediates_tsne):
    print('\n=========================================== \n    3D_projection_initiate    \n===========================================')
    zero = [0] * 66 ; one = [1] * 100 ; two = [2] * 60
    zero2 = [0] * 25 ; one2 = [1] * 25 ; two2 = [2] * 25
    label = zero + one + two
    label2 = zero2 + one2 + two2
    label3 = label2 + label
    label = np.array(label3)

    fig = plt.figure(figsize=(15,15))
    color_list = [DM0,DM1,DM2,CM0,CM1,CM2]
    ax = fig.add_subplot(111, projection='3d')

    for color_idx in color_list:
        idx = color_multi==color_idx
        if(color_idx==DM0):
            label_col = 'Train No HS'
        elif(color_idx==CM0):
            label_col = 'Test No HS'
                
        elif(color_idx==DM1):
            label_col = 'Train Left HS'
        elif(color_idx==CM1):
            label_col = 'Test Left HS'
                    
        elif(color_idx==DM2):
            label_col = 'Train Right HS'
        elif(color_idx==CM2):
            label_col = 'Test Right HS'
        
        if(color_idx==CM0)or(color_idx==CM1)or(color_idx==CM2): # TEST
            ax.scatter(intermediates_tsne[:,0][idx], intermediates_tsne[:,1][idx], intermediates_tsne[:,2][idx], zdir='z', s=p_size, c=color_idx,label=label_col, depthshade=False, marker='^' , edgecolor='black', linewidth=l_width) #4
        elif(color_idx==DM0)or(color_idx==DM1)or(color_idx==DM2): # TRAIN
            ax.scatter(intermediates_tsne[:,0][idx], intermediates_tsne[:,1][idx], intermediates_tsne[:,2][idx], zdir='z', s=30, c=color_idx,label=label_col, depthshade=False)

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.legend(prop={'size': 15})
    ax.figure.savefig(layer_path+'/%s3D_stop_K%d_%d.png'%(SELECT,KERNEL_SEED,layer_of_interest))

    def rotate(angle):
        ax.view_init(azim=angle)

    angle = 3
    ani = animation.FuncAnimation(fig, rotate, frames=np.arange(0, 360, angle), interval=50)
    ani.save(layer_path+'/%s3D_K%d_%d.gif' %(SELECT,KERNEL_SEED, layer_of_interest), writer=animation.PillowWriter(fps=20))