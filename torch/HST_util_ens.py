from HST_common import *
from HST_model import *
from HST_util import *

def log_ensemble():
    storage = 'log/ensembleStorage.txt'
    fwSt=open(storage,'a')
    fileName = 'log/[%s%d%s]aLog_%s[%s]K%s.txt' %(SETT,TRIAL,AUG,CONTROLTYPE,EnsMODE,str(K))
    fw = open(fileName,'a')
    fwSt.write('\n\nTRIAL %d  SETT %s  DATATYPE %d  CONTROLTYPE %s  EnsMODE %s Kernels[%s]\n' %(TRIAL, SETT, DATATYPE, CONTROLTYPE, EnsMODE,str(K)))
    return fw, fwSt

if(SETT=='FUL'):
    iC = tstCount
    iCN0 = tstCountNO_0   ;  iCN4 = tstCountNO_4  ;  iCN7 = tstCountNO_7
    iCY1 = tstCountYES_1 ;  iCY2 = tstCountYES_2
elif(SETT=='SIG')or(SETT=='MM'):
    iC = imgCount
    iCN0 = imgCountNO_0   ;  iCN4 = imgCountNO_4  ;  iCN7 = imgCountNO_7
    iCY1 = imgCountYES_1 ;  iCY2 = imgCountYES_2


def MAIN_ENSB(inputX,inputY,Y_vector,fw,fwSt):
    ensb = CLASS_ENSB(iC, iCN4, iCN7, iCN0, iCY1,iCY2)
    if(SETT=='SIG')or(SETT=='MM'):
        for train_index, val_index in KFOLD.split(inputX,Y_vector):
            ensb.GET_ENSB(train_index,val_index)
            ensb.MODL_ENSB()    
            ensb.PRE_ENSB(inputX,inputY)   
        yIdxPrediction, yLabelPrediction, yPredictionB, yLabelPredictionB,accuracy,accuracyB = ensb.SET_ENSB()
        acc_roc(accuracy,accuracyB,yIdxPrediction,yLabelPrediction,yPredictionB,yLabelPredictionB,fw,fwSt,iters,graph_path,imgCountNO_4,imgCountNO_7,imgCountNO_0,imgCountYES_1,imgCountYES_2)
    elif(SETT=='FUL'):
        tst_index = np.arange(len(inputY))
        ensb.GET_ENSB(tst_index,tst_index)
        ensb.MODL_ENSB()    
        ensb.PRE_ENSB(inputX,inputY)    
        yIdxPrediction, yLabelPrediction, yPredictionB, yLabelPredictionB,accuracy,accuracyB = ensb.SET_ENSB()
        acc_roc(accuracy,accuracyB,yIdxPrediction,yLabelPrediction,yPredictionB,yLabelPredictionB,fw,fwSt,iters,graph_path,tstCountNO_4,tstCountNO_7,tstCountNO_0,tstCountYES_1,tstCountYES_2)
        
    ensb.RESULT_ENSB(fw)

class CLASS_ENSB :
    def __init__(self,iC, iCN4, iCN7, iCN0, iCY1, iCY2):
        self.yPredictionB = []       # Prediction scores for Binary
        self.yIdxPrediction = []     # Data Order Restoration
        self.yLabelPrediction = []   # Predicted Label for Multi
        self.yLabelPredictionB = []  

        self.foldNum = 0
        self.accuracy = [] 
        self.accuracyB = []
        
        self.iC = iC     ; self.iCN4 = iCN4 ; self.iCN7 = iCN7
        self.iCN0 = iCN0 ; self.iCY1 = iCY1 ; self.iCY2 = iCY2

    def GET_ENSB(self, train_index,val_index):
        self.train_index = train_index
        self.val_index = val_index
            
    def SET_ENSB(self):
        return self.yIdxPrediction, self.yLabelPrediction, self.yPredictionB, self.yLabelPredictionB, self.accuracy, self.accuracyB
    
    def MODL_ENSB(self):
        self.ensembleModel = []      # Models to ensemble
        for kerCnt in range(ENSEMBLE_NUM):

            ## model load ##
            if(MODEL=='3D_5124'): 

                if(CONTROLTYPE=='CLRM'): temp_net = HSCNN(ksize)
                elif('ADNI' in CONTROLTYPE): temp_net = ADNICNN(ksize)
                if (device == 'cuda')and(ksize==4): temp_net = torch.nn.DataParallel(temp_net)
            if(SETT=='SIG')or(SETT=='MM'):
                model_path = os.getcwd()+'/saveModel/[%s%d%s]HS%s_D%d{F%dK%d}[%d](best).pt'%(SETT,TRIAL,AUG,CONTROLTYPE,DATATYPE,FOLD_SEED,KERNELS[kerCnt],self.foldNum)
            elif(SETT=='FUL'):
                    model_path = os.getcwd()+'/saveModel/[%s%d%s]HS%s_D%d{F%dK%d}[best].pt'%(SETT,TRIAL,AUG,CONTROLTYPE,DATATYPE,FOLD_SEED,KERNELS[kerCnt])
            print(model_path)
            if os.path.isfile(model_path):
                temp_net.load_state_dict(torch.load(model_path))
            else:
                print('\n!!!warning!!! \n load Model file not exist!!')
            self.ensembleModel.append(temp_net)

        del temp_net
        self.foldNum += 1

    def PRE_ENSB(self,inputX,inputY): 
        self.eLabels = []            # Label Prediction fo r Voting ensemble
        self.eLabelsB = []           
        self.ePredictions = []       # Score Prediction for Average Ensemble
        self.ePredictionsB = []

        for Num in range(len(self.val_index)):
            self.yIdxPrediction.append(self.val_index[Num])

        modeIdx = 0 ; indexModel = 0
        for cntModel in self.ensembleModel:
            if(indexModel % ENSEMBLE_NUM == 0):
                print('Modality Idx', modeIdx)
                modeIdx = modeIdx + 1
            print('indexModel %d ENSEMBLE_NUM %d' %(indexModel+1,ENSEMBLE_NUM))
            indexModel = indexModel +1
            if(SETT=='SIG')or(SETT=='MM'):
                _, val_loader = CV_data_load(inputX,inputY,self.train_index,self.val_index,AUG,False)
            elif(SETT=='FUL'):
                _, val_loader = FUL_data_load(inputX,inputY,inputX,inputY,AUG,False)
            total = 0 ; totalB = 0 ; correct = 0 ; correctB = 0
            cntModel.to(device)
            cntModel.eval()

            with torch.no_grad():
                for batch_index, (images, labels) in enumerate(val_loader):
                    images = images.view(-1,1,imgRow,imgCol,imgDepth)
                    images = images.float()
                    images = images.to(device)
                    labels = labels.to(device)
                    output = F.softmax(cntModel(images),dim=1)  

                    labelsB = labels.detach().clone()
                    labelsB[labelsB!=0] = 1
                    labelsB = labelsB.to(device)

                    # Multi-to-Binary
                    out_yes = output[:,1:3].sum(dim=1)
                    out_no = output[:,0]
                    outputB = torch.stack([out_no,out_yes],dim=1)
                    yPredB = outputB[:,1]

                    _, pred = output.max(1)
                    correct += pred.eq(labels).sum().item()
                    total += labels.size(0)

                    _, predB = outputB.max(1) # dim=1 => Compare with!
                    correctB += predB.eq(labelsB).sum().item()
                    totalB += labelsB.size(0)

                    output = output.detach().cpu().numpy()
                    outputB = outputB.detach().cpu().numpy()
                    pred = pred.detach().cpu().numpy()
                    predB = predB.detach().cpu().numpy()
                    yPredB = yPredB.detach().cpu().numpy()
                    

                    if(batch_index==0):
                        multi_result = output.copy()
                        binary_result = outputB.copy() 
                        yLP = pred.copy()
                        yLPB = predB.copy()
                        yPB = yPredB.copy()
                    else :
                        multi_result = np.concatenate([multi_result,output],axis=0)
                        binary_result = np.concatenate([binary_result,outputB],axis=0)
                        yLP = np.concatenate([yLP,pred],axis=0)
                        yLPB = np.concatenate([yLPB,predB],axis=0)
                        yPB = np.concatenate([yPB,yPredB],axis=0)

            del images

            multi_acc = 100.*correct/total
            binary_acc = 100.*correctB/totalB
            k_accuracy = '%.2f' %(multi_acc)
            k_accuracyB = '%.2f' %(binary_acc)
            self.accuracy.append(k_accuracy)
            self.accuracyB.append(k_accuracyB)
            print('Best epoch wise  Accuracy: {}'.format(self.accuracy))

            self.eLabels.append(yLP)
            self.ePredictions.append(multi_result)
            self.eLabelsB.append(yLPB)
            self.ePredictionsB.append(binary_result)

        if(EnsMODE == 'AVR'):
            self.AVR_ENSB()             
        elif(EnsMODE == 'VOT'):
            self.VOT_ENSB()            
  
        del self.ensembleModel
        del self.train_index
        del self.val_index
    
    def AVR_ENSB(self):
        ePredictions = self.ePredictions
        ePredictionsB = self.ePredictionsB
        
        #sum across ensemble members
        ePredictions = np.array(ePredictions)
        ePredictionsB = np.array(ePredictionsB)
        ePredictions = np.reshape(ePredictions,(ENSEMBLE_NUM,-1,nb_classes))
        ePredictionsB = np.reshape(ePredictionsB,(ENSEMBLE_NUM,-1,2))

        avr = np.average(ePredictions, axis=0)
        avrB = np.average(ePredictionsB, axis=0)
        # argmax across classes
        result = np.argmax(avr, axis=1)
        resultB = np.argmax(avrB, axis=1)

        for Num in range(len(self.val_index)):
            self.yPredictionB.append(avrB[Num][1])
            self.yLabelPrediction.append(result[Num])
            self.yLabelPredictionB.append(resultB[Num])

        del avr, avrB
        del result, resultB
        del ePredictions, self.ePredictions
        del ePredictionsB, self.ePredictionsB
        
    def VOT_ENSB(self):
        eLabels = self.eLabels
        eLabelsB = self.eLabelsB
        
        # Multi-class Classification
        eLabels = np.array(eLabels)               
        eLabels = np.reshape(eLabels,(ENSEMBLE_NUM,len(self.val_index)))  # (ensemble_models,val_len)
        eLabels = np.transpose(eLabels, (1, 0))    # (val_len,ensemble_models)
        eLabels = scipy.stats.mode(eLabels, axis=1)[0] # (val_len,1)
        eLabels = np.squeeze(eLabels) # (val_len,)
        for Num in range(len(self.val_index)):
            self.yLabelPrediction.append(eLabels[Num])

        # Binary Classification
        eLabelsB = np.array(eLabelsB)
        eLabelsB = np.reshape(eLabelsB,(ENSEMBLE_NUM,len(self.val_index)))
        eLabelsB = np.transpose(eLabelsB, (1, 0))
        vottingArrayB = eLabelsB
        eLabelsB = scipy.stats.mode(eLabelsB, axis=1)[0]
        eLabelsB = np.squeeze(eLabelsB)
        for Num in range(len(self.val_index)):
            self.yLabelPredictionB.append(eLabelsB[Num])
            vottingPrediction = 0
            for cnt in range(len(self.ensembleModel)):
                vottingPrediction = vottingPrediction + vottingArrayB[Num][cnt]
            self.yPredictionB.append(vottingPrediction/len(self.ensembleModel))
        
        del eLabels, self.eLabels
        del eLabelsB, self.eLabelsB
            
    def RESULT_ENSB(self,fw):

        yLabelPrediction = np.array(self.yLabelPrediction)
        yLabelPredictionB = np.array(self.yLabelPredictionB)
        yIdxPrediction = np.array(self.yIdxPrediction)
        yPredictionB = np.array(self.yPredictionB)
            

        Y_vector = []  
        for i in range(self.iCN4):
            Y_vector.append(4)
        for i in range(self.iCN7):
            Y_vector.append(7)
        for i in range(self.iCN0):
            Y_vector.append(0)
        for i in range(self.iCY1):
            Y_vector.append(1)
        for i in range(self.iCY2):
            Y_vector.append(2)
                
        tempMatrix = [0]*self.iC ; tempMatrixP1 = [0]*self.iC ; tempMatrixP2 = [0]*self.iC

        for idx in range(self.iC):
            tempMatrix[yIdxPrediction[idx]] = yLabelPrediction[idx]
            tempMatrixP1[yIdxPrediction[idx]] = yPredictionB[idx]
            tempMatrixP2[yIdxPrediction[idx]] = yLabelPredictionB[idx]

        tempMatrixP1 = np.array(tempMatrixP1)
        yLabelPrediction = tempMatrix  ;  yPredictionB = tempMatrixP1  ;  yLabelPredictionB = tempMatrixP2

        self.yPredictionB = yPredictionB 
        self.yIdxPrediction = yIdxPrediction
        self.yLabelPrediction = yLabelPrediction
        self.yLabelPredictionB = yLabelPredictionB
        
        self.eval_class(Y_vector,fw)

    def eval_class(self,Y_vector,fw):

        
        # Specific Evaluation for each class
        LEFT_NO=0; LEFT_TRUE=0 ; LEFT_RIGHT=0
        RIGHT_NO=0; RIGHT_LEFT=0 ; RIGHT_TRUE=0
        NO_TRUE=0 ; NO_LEFT=0 ; NO_RIGHT=0
        FOUR_TRUE=0 ; FOUR_LEFT=0 ; FOUR_RIGHT=0
        SEVEN_TRUE=0 ; SEVEN_LEFT=0 ; SEVEN_RIGHT=0

        LEFT_TRUE_B=0 ; LEFT_FALSE_B=0
        RIGHT_TRUE_B=0 ; RIGHT_FALSE_B=0
        NO_TRUE_B=0 ; NO_FALSE_B=0
        FOUR_TRUE_B=0 ; FOUR_FALSE_B=0
        SEVEN_TRUE_B=0 ; SEVEN_FALSE_B=0 
    

        for idx in range(len(self.yLabelPrediction)):
            Yidx = Y_vector[idx]
            yLPidx = self.yLabelPrediction[idx]
            if(Yidx==0):
                if(yLPidx==0):
                    NO_TRUE+=1
                elif(yLPidx==1):
                    NO_LEFT+=1
                elif(yLPidx==2):
                    NO_RIGHT+=1
            elif(Yidx==4):
                if(yLPidx==0):
                    FOUR_TRUE+=1
                elif(yLPidx==1):
                    FOUR_LEFT+=1
                elif(yLPidx==2):
                    FOUR_RIGHT+=1
            elif(Yidx==7):
                if(yLPidx==0):
                    SEVEN_TRUE+=1
                elif(yLPidx==1):
                    SEVEN_LEFT+=1
                elif(yLPidx==2):
                    SEVEN_RIGHT+=1
            elif(Yidx==1):
                if(yLPidx==1):
                    LEFT_TRUE+=1
                elif(yLPidx==0):
                    LEFT_NO+=1
                elif(yLPidx==2):
                    LEFT_RIGHT+=1
            elif(Yidx==2):
                if(yLPidx==2):
                    RIGHT_TRUE+=1
                elif(yLPidx==0):
                    RIGHT_NO+=1
                elif(yLPidx==1):
                    RIGHT_LEFT+=1


        NO_TRUE_B = NO_TRUE ; NO_FALSE_B = NO_LEFT + NO_RIGHT
        FOUR_TRUE_B = FOUR_TRUE ; FOUR_FALSE_B = FOUR_LEFT + FOUR_RIGHT
        SEVEN_TRUE_B = SEVEN_TRUE ; SEVEN_FALSE_B = SEVEN_LEFT + SEVEN_RIGHT
        LEFT_TRUE_B = LEFT_TRUE + LEFT_RIGHT ; LEFT_FALSE_B =  LEFT_NO
        RIGHT_TRUE_B = RIGHT_TRUE + RIGHT_LEFT ; RIGHT_FALSE_B = RIGHT_NO
        
        fw.write('\n\n\n\n [Multi Class Errors]')
        fw.write('\n\n NO : %0.3f (%d/%d/%d)' %(np.round(NO_TRUE/self.iCN0,3),NO_TRUE,NO_LEFT, NO_RIGHT))
        if(self.iCN4!=0):
            fw.write('\n\n FOUR : %0.3f (%d/%d/%d)' %(np.round(FOUR_TRUE/self.iCN4,3),FOUR_TRUE,FOUR_LEFT, FOUR_RIGHT))
        fw.write('\n\n SEVEN : %0.3f (%d/%d/%d)' %(np.round(SEVEN_TRUE/self.iCN7,3),SEVEN_TRUE,SEVEN_LEFT, SEVEN_RIGHT))
        fw.write('\n\n LEFT : %0.3f (%d/%d/%d)' %(np.round(LEFT_TRUE/self.iCY1,3),LEFT_NO, LEFT_TRUE,LEFT_RIGHT))
        fw.write('\n\n RIGHT : %0.3f (%d/%d/%d)' %(np.round(RIGHT_TRUE/self.iCY2,3),RIGHT_NO, RIGHT_LEFT, RIGHT_TRUE))

        fw.write('\n\n [Binary Class Errors]')
        fw.write('\n\n NO : %0.3f (%d/%d)' %(np.round(NO_TRUE_B/self.iCN0,3),NO_TRUE_B,NO_FALSE_B))
        if(self.iCN4!=0):
            fw.write('\n\n FOUR : %0.3f (%d/%d)' %(np.round(FOUR_TRUE_B/self.iCN4,3),FOUR_TRUE_B,FOUR_FALSE_B))
        fw.write('\n\n SEVEN : %0.3f (%d/%d)' %(np.round(SEVEN_TRUE_B/self.iCN7,3),SEVEN_TRUE_B,SEVEN_FALSE_B))
        fw.write('\n\n LEFT : %0.3f (%d/%d)' %(np.round(LEFT_TRUE_B/self.iCY1,3),LEFT_FALSE_B,LEFT_TRUE_B))
        fw.write('\n\n RIGHT : %0.3f (%d/%d)' %(np.round(RIGHT_TRUE_B/self.iCY2,3),RIGHT_FALSE_B,RIGHT_TRUE_B))