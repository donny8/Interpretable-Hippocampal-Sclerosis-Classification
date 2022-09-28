from utils.util import *
from utils.util_ens import *
from utils.common import *

def main():
    fw, fwSt = log_ensemble()
    ensb = CLASS_ENSB(iC, iCN4, iCN7, iCN0, iCY1,iCY2)

    if(SETT=='SIG')or(SETT=='MM'):
        inputX, inputY, Y_vector,_,_ = source_load(categories, dirDataSet)
        for train_index, val_index in KFOLD.split(inputX,Y_vector):
            ensb.GET_ENSB(train_index,val_index)
            ensb.MODL_ENSB()    
            ensb.PRE_ENSB(inputX,inputY)   
        yIdxPrediction, yLabelPrediction, yPredictionB, yLabelPredictionB,accuracy,accuracyB = ensb.SET_ENSB()
        acc_roc(accuracy,accuracyB,yIdxPrediction,yLabelPrediction,yPredictionB,yLabelPredictionB,fw,fwSt,iters,graph_path,imgCountNO_4,imgCountNO_7,imgCountNO_0,imgCountYES_1,imgCountYES_2)

    elif(SETT=='FUL'):
        Y_vector = []
        inputX, inputY, _, _ = data_single(categories,tstDataSet)
        tst_index = np.arange(len(inputY))
        ensb.GET_ENSB(tst_index,tst_index)
        ensb.MODL_ENSB()    
        ensb.PRE_ENSB(inputX,inputY)    
        yIdxPrediction, yLabelPrediction, yPredictionB, yLabelPredictionB,accuracy,accuracyB = ensb.SET_ENSB()
        acc_roc(accuracy,accuracyB,yIdxPrediction,yLabelPrediction,yPredictionB,yLabelPredictionB,fw,fwSt,iters,graph_path,tstCountNO_4,tstCountNO_7,tstCountNO_0,tstCountYES_1,tstCountYES_2)
        
    ensb.RESULT_ENSB(fw)

    log_close(fw, fwSt, start)

if __name__ == '__main__':
    main()
