from HS_common import *
from HS_modeling import *
from HS_util import *
from HS_figure_Scr import *

arrayInputX, arrayInputY, Y_vector, fw, fwSt = ens_expr_sett()
skf = StratifiedKFold(n_splits=nb_KFold, random_state=FOLD_SEED, shuffle=True)
FOLD_SEED = FOLD_SEED +1

MBpred = mul_bin_pred()
err_mul = err_per_class()
err_bin = err_per_class()

# Separate given data into training data and validation data
foldNum = 0
for train, validation in skf.split(arrayInputX,Y_vector):

    eLabels = []; ePredictions = []
    eLabels_B = []; ePredictions_B = []; ensembleModel = []
    
    for imgCnt in range(ENSEMBLE_IMG):
        for kerCnt in range(ENSEMBLE_NUM):
            modelHS = model_load(foldNum,FOLD_SEED,kerCnt+1)
            ensembleModel.append(modelHS)
            del modelHS
    print('ensembleModel len', len(ensembleModel))

    #save log train & validation set
    logDataSetHS(modelNum,1,foldNum,train,validation,fw)
    foldNum += 1
    dataIdx = 0; indexModel = 0    
    for cntModel in ensembleModel:
        print('indexModel % ENSEMBLE_NUM', indexModel % ENSEMBLE_NUM)
        if(indexModel % ENSEMBLE_NUM == 0):
            dataIdx = dataIdx + 1
            print('dataIdx', dataIdx)
        indexModel = indexModel +1

        if(CONVOLUTION_TYPE != '2D_CNN'):
            arrayInputX=arrayInputX.reshape(imgCount,imgRow,imgCol,imgDepth,1)

        # Verification using test data
        score = cntModel.evaluate(arrayInputX[validation], arrayInputY[validation],batch_size=nb_batchSize)[1]            
        # Predict labels with models
        labelPredicts = np.argmax((cntModel.predict(arrayInputX[validation], batch_size=nb_batchSize, verbose=0, steps=None)), axis=1)
        eLabels.append(labelPredicts)
        ePredictions, ePredictions_B, eLabels_B = ens_mul_bin(ePredictions, ePredictions_B, eLabels_B,MBpred,cntModel,arrayInputX,arrayInputY,validation,fw)

    # Check the Index of the data for sorting        
    for Num in range(len(validation)):
        MBpred.yIdxPrediction.append(validation[Num])
    # Sum across ensemble members
    avr = np.average(ePredictions, axis=0)
    avr_B = np.average(ePredictions_B, axis=0)
    # Argmax across classes
    result = np.argmax(avr, axis=1)
    result_B = np.argmax(avr_B, axis=1)

    if(ENSEMBLE_MODE == 'VOT'):
        VOT_ensemble(eLabels,eLabels_B,MBpred,validation,len(ensembleModel))
    elif(ENSEMBLE_MODE == 'AVR'):
        for Num in range(len(validation)):
            MBpred.yLabelPrediction.append(result[Num])
        for Num in range(len(validation)):
            MBpred.yPrediction_B.append(avr_B[Num][1])
            MBpred.yLabelPrediction_B.append(result_B[Num])    

    ##delete model & parametor ##
    gc.collect()
    del avr, avr_B
    del ensembleModel

del skf, arrayInputX
### End of training & validation sequence!! ###

performance(MBpred, fwSt,fw)
error_class(MBpred.yLabelPrediction,Y_vector,err_mul)
error_class(MBpred.yLabelPrediction_B,Y_vector,err_bin)
error_report(err_mul,err_bin,fwSt)

fw.close()
fwSt.close()