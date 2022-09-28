from utils.common import *
from utils.model import *
from utils.util import *
from utils.util_lrp import *
import gc
import math
import gzip


foldNum = 0 
police=[]; inputX = []; inputY=[] ;listFileName = []

if(SETT=='FUL'): 
    dirDataSet = tstDataSet
#inputX, inputY, Y_vector, listFileName, niiInfo = source_load(categories, dirDataSet)
inputX, inputY, Y_vector, listFileName, niiInfo = source_load(categories, dirDataSet)

mkdir('./3D_output')    
dir_path = os.getcwd()+'/3D_output/%s_T%d_%d%s' %(CONTROLTYPE,TRIAL,DATATYPE,LRPMSG)
if(not(os.path.isdir(dir_path))):
        os.mkdir(dir_path)
        
LRPort = dir_path + '/[%s%d%s]LRP_Reports.txt' %(SETT,PERCENT, AUG)
print(LRPort)
LR = open(LRPort,'a')
        
NO = 0 ; NO_true = 0
LEFT = 0 ; LEFT_true = 0
RIGHT = 0 ; RIGHT_true = 0

NO_avg = torch.zeros((imgRow,imgCol,imgDepth))
LEFT_avg = torch.zeros((imgRow,imgCol,imgDepth))
RIGHT_avg = torch.zeros((imgRow,imgCol,imgDepth))

for train, validation in KFOLD.split(inputX,Y_vector):

    inputFoldX = inputX[validation].float()
    inputFoldY = inputY[validation]
    listFileNameFold = listFileName[validation]
    print('fold inputX shape', inputFoldX.shape)
    
    lrp_path = dir_path + '/%s_FOLD_%d' %(RULE,foldNum)
    if(not(os.path.isdir(lrp_path))):
        os.mkdir(lrp_path)

    ## model load ##
    if(MODEL=='3D_5124'): 
        model = HSCNN(ksize=4)
        if (device == 'cuda'): 
            model = torch.nn.DataParallel(model)
        
    if(SETT=='SIG'):
        model_path = os.getcwd()+'/saveModel/[%s%d%s]HS%s_D%d{F%dK%d}[%d](best).pt'%(SETT,TRIAL,AUG,CONTROLTYPE,DATATYPE,FOLD_SEED,KERNEL_SEED,foldNum)
    elif(SETT=='FUL'):
        model_path = os.getcwd()+'/saveModel/[%s%d%s]HS%s_D%d{F%dK%d}[best].pt'%(SETT,TRIAL,AUG,CONTROLTYPE,DATATYPE,FOLD_SEED,KERNEL_SEED)
    print(model_path)
    if os.path.isfile(model_path):
        model.load_state_dict(torch.load(model_path))
    else:
        print('\n!!!warning!!! \n load Model file not exist!!')
    model.cuda()
    model.eval()
    
    inputFoldX = inputFoldX.view(-1,1,imgRow,imgCol,imgDepth)
    inputFoldX = inputFoldX.cuda()
    
    inn_model = InnvestigateModel(model, lrp_exponent=1, method="e-rule", beta=0.5, epsilon=1e-6)
    inn_model = inn_model.cuda()
    
    backgraoundPixeloffset = 15
    squareValue = 0.1
    
    inputStartIdx = 0
    inputEndIdx = 1


    for loopCnt in range(len(inputFoldX)):
        prob_Sum = 0 ; cnt_Sum = 0
            
        prob = model(inputFoldX[loopCnt:loopCnt+1])
        prob = prob.view(nb_classes)
        prob = nn.Softmax(dim=0)(prob)

        for neuronCnt in range(nb_classes):
            score, result = inn_model.innvestigate(in_tensor=inputFoldX[inputStartIdx:inputEndIdx], rel_for_class=neuronCnt)
            result = result.view(imgRow,imgCol,imgDepth)
            if(RULE!='lrp.alpha_1_beta_0'):
                result = np.where((result<0),0,result)
            nomalizeLRP = (result - np.min(result))
            nomalizeLRP = nomalizeLRP - nomalizeLRP[backgraoundPixeloffset][backgraoundPixeloffset][backgraoundPixeloffset] #backGround Sub.
            nomalizeLRP = np.where((nomalizeLRP < 0), 0 ,nomalizeLRP) #backGround to "zero"
    
            # distribution 0~1 #
            nomalizeLRP = nomalizeLRP - np.min(nomalizeLRP) 
            nomalizeLRPori = 100*nomalizeLRP / np.max(nomalizeLRP)
            # cutoff by percent #
            nomalizeLRP = np.where((nomalizeLRPori < PERCENT), 0 ,nomalizeLRPori)
            check = nancheck(nomalizeLRPori)    
                
            target = inputFoldY[loopCnt:loopCnt+1]
            if(neuronCnt == target):
                best = torch.argmax(prob,dim=0)
                print('nancheck : %d  C%d prob : %.3f  best : %d' %(check,neuronCnt,prob[neuronCnt].item(),best.item()) )
                print('neuron:',neuronCnt)
                if(neuronCnt==0):
                    NO, NO_true, NO_avg = counter(check,nomalizeLRP,best,NO,NO_true,NO_avg,neuronCnt)
                        
                elif(neuronCnt==1):
                    print('Left:',LEFT)
                    LEFT, LEFT_true, LEFT_avg = counter(check,nomalizeLRP,best,LEFT,LEFT_true,LEFT_avg,neuronCnt)
                        
                elif(neuronCnt==2):
                    RIGHT, RIGHT_true, RIGHT_avg = counter(check,nomalizeLRP,best,RIGHT,RIGHT_true,RIGHT_avg,neuronCnt)
                                
    
            ##  File name patient number ##
            if(neuronCnt==0):
                print('filename',listFileNameFold[inputStartIdx])
                patient = listFileNameFold[inputStartIdx].split('.nii')
                y = patient[0][-12:]
                print('y',y)
                LR.write('\n\n%s :' %(y))
                    
            if(nancheck(nomalizeLRPori)):
                police.append(y)                        
                        
            ni_img = nib.Nifti1Image(nomalizeLRP, niiInfo.affine, niiInfo.header)
            SaveFileName = lrp_path + '/%s_%s_C%d[%d].nii' %(y,RULE,neuronCnt,prob[neuronCnt]*100)
            GzSaveFileName = lrp_path + '/%s_%s_C%d[%d].nii.gz' %(y,RULE,neuronCnt,prob[neuronCnt]*100)
            LRPsave(ni_img, SaveFileName, GzSaveFileName)
            
            del nomalizeLRP
            del nomalizeLRPori
            
            if(neuronCnt==0):
                LR.write('\nNO : ')       
            elif(neuronCnt==1):
                LR.write('\nLEFT : ')
            elif(neuronCnt==2):
                LR.write('\nRIGHT : ')

            temp_prob = prob[neuronCnt]
            LR.write('%.3f'%(temp_prob))  

        inputStartIdx += 1
        inputEndIdx += 1
        print('LRP START,END', inputStartIdx,inputEndIdx)
        target.cpu()
    
    del model
    del ni_img
    del inn_model
    del inputFoldX
    
    foldNum += 1
    torch.cuda.empty_cache()       
    

NO_avg_Name = dir_path+'/%s_LRP_NO_avg.nii' %(RULE)
GZ_NO_avg_Name = dir_path+'/%s_LRP_NO_avg.nii.gz' %(RULE)
averageLRP(NO_avg,NO_true,niiInfo,NO_avg_Name,GZ_NO_avg_Name)


LEFT_avg_Name = dir_path+'/%s_LRP_LEFT_avg.nii' %(RULE)
GZ_LEFT_avg_Name = dir_path+'/%s_LRP_LEFT_avg.nii.gz' %(RULE)
averageLRP(LEFT_avg,LEFT_true,niiInfo,LEFT_avg_Name,GZ_LEFT_avg_Name)


RIGHT_avg_Name = dir_path+'/%s_LRP_RIGHT_avg.nii' %(RULE)
GZ_RIGHT_avg_Name = dir_path+'/%s_LRP_RIGHT_avg.nii.gz' %(RULE)
averageLRP(RIGHT_avg,RIGHT_true,niiInfo,RIGHT_avg_Name,GZ_RIGHT_avg_Name)


print('\nDebug!')
print('NO :', NO)
print('NO_true :',NO_true)
print('NO_acc :', NO_true/NO)
print('\nLEFT :', LEFT)
print('LEFT_true :',LEFT_true)
print('LEFT_acc :', LEFT_true/LEFT)
print('\nRIGHT :', RIGHT)
print('RIGHT_true :',RIGHT_true)
print('RIGHT_acc :', RIGHT_true/RIGHT)
print('\n Police :', police)

LR.write('\n ===================================')

temp = '\n\nNO : ' + str(NO)
LR.write(temp)
temp = '\nNO_true : ' + str(NO_true)
LR.write(temp)
temp = '\nNO_acc :' + str(np.around(NO_true/NO,3))
LR.write(temp)

temp = '\n\nLEFT : ' + str(LEFT)
LR.write(temp)
temp = '\nLEFT_true : ' + str(LEFT_true)
LR.write(temp)
temp = '\nLEFT_acc :' + str(np.around(LEFT_true/LEFT,3))
LR.write(temp)

temp = '\n\nRIGHT : ' + str(RIGHT)
LR.write(temp)
temp = '\nRIGHT_true : ' + str(RIGHT_true)
LR.write(temp)
temp = '\nRIGHT_acc :' + str(np.around(RIGHT_true/RIGHT,3))
LR.write(temp)

print(timer(start))
LR.close()
