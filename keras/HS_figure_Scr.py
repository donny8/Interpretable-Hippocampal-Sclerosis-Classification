from HS_common import *

class historyInfoHS:

    def __init__(self):
        self.accuracy = []
        self.accuracyFloat = []
        self.val_loss = [[0]*nb_epochs for i in range(nb_KFold)]
        self.val_acc = [[0]*nb_epochs for i in range(nb_KFold)]
        self.train_loss = [[0]*nb_epochs for i in range(nb_KFold)]
        self.train_acc = [[0]*nb_epochs for i in range(nb_KFold)]        

        
class mul_bin_pred:

    def __init__(self):
        self.p = []
        self.pp = []
        self.yPrediction = []
        self.yLabelPrediction = []
        self.yIdxPrediction = []
        self.yPrediction_B = []
        self.yLabelPrediction_B = []

class err_per_class:

    def __init__(self):
        self.LEFT_NO=0; self.LEFT_TRUE=0 ; self.LEFT_RIGHT=0
        self.RIGHT_NO=0; self.RIGHT_LEFT=0 ; self.RIGHT_TRUE=0
        self.NO_TRUE=0 ; self.NO_LEFT=0 ; self.NO_RIGHT=0
        self.FOUR_TRUE=0 ; self.FOUR_LEFT=0 ; self.FOUR_RIGHT=0
        self.SEVEN_TRUE=0 ; self.SEVEN_LEFT=0 ; self.SEVEN_RIGHT=0

class LRP_result:

    def __init__(self):
        self.NO = 0; self.NO_true = 0
        self.NO_avg = np.zeros((imgRow,imgCol,imgDepth,1))

        self.LEFT = 0; self.LEFT_true = 0
        self.LEFT_avg = np.zeros((imgRow,imgCol,imgDepth,1))

        self.RIGHT = 0; self.RIGHT_true = 0
        self.RIGHT_avg = np.zeros((imgRow,imgCol,imgDepth,1))


def historyMeanHST(historyTrain,Totalaccuracymodel,fw,RR,modelNum,trainNum,foldNum):
    avr_val_loss = np.mean(historyTrain.val_loss, axis = 0)
    avr_val_acc = np.mean(historyTrain.val_acc, axis = 0)
    avr_train_loss = np.mean(historyTrain.train_loss, axis = 0)
    avr_train_acc = np.mean(historyTrain.train_acc, axis = 0)
    val_acc = historyTrain.val_acc
    val_acc = np.array(val_acc)

    avr_val_acc = np.array(avr_val_acc)
    maxIdx = np.argmax(avr_val_acc, axis=0)
    arrayMaxCV = []
    for loopNum in range(nb_KFold):
        tempString =  '%.2f' %(val_acc[loopNum][maxIdx])
        arrayMaxCV.append(tempString)

    print(FOLD_SEED, 'K-fold Max CV Accuracy: {}'.format(arrayMaxCV))
    Totalaccuracymodel.append(max(avr_val_acc))

    temp_log = 'maxIDx: %d, maxValue %0.3f\n' %(maxIdx,max(avr_val_acc))
    fw.write(temp_log)
    RR.write(temp_log)
    temp_log = '%d, fold Max CV Accuracy: ' %(FOLD_SEED)    
    fw.write(temp_log)
    RR.write(temp_log)
    temp_log = '%s\n\n' %(arrayMaxCV)    
    fw.write(temp_log)
    RR.write(temp_log)

    return Totalaccuracymodel

def logDataSetHS(modelNum,trainNum,foldNum,train,validation,fw):
    temp_log = '***   dataset [%d](%d)_%d   ***\n' %(modelNum,FOLD_SEED,foldNum+1)
    fw.write(temp_log)
    train_log = 'train idx :{}\n'.format(train)                                      
    fw.write(train_log)
    validation_log = 'validation idx :{}\n'.format(validation)
    fw.write(validation_log)

    return


def calcPredictionHS(resultPredict,Y,validation,fw):
    accuracy = 0
    for Num in range(len(validation)):
        idxLabel = np.argmax(Y[validation][Num], axis=0)
        idxPredict = np.argmax(resultPredict[Num], axis=0)
        if(idxLabel == idxPredict):
            flagResult = 'P'
            accuracy = accuracy + 1
        else:
            flagResult = 'F'         
        temp_log = 'PD %d th : %d %c' %(Num+1,validation[Num],flagResult)
        fw.write(temp_log)
        temp_log = '%s\n' %(np.around(resultPredict[Num],decimals=1))    
        fw.write(temp_log)
        
        tempfloat =  accuracy / len(validation)
    return tempfloat



def result1Darray(resultPredict,Y):
    arr1D = []
    for Num in range((int)(len(Y)/nb_KFold)):
        # idxLabel = np.argmax(Y[validation][Num], axis=0)
        arr1D.append(np.argmax(resultPredict[Num], axis=0))
    return arr1D
