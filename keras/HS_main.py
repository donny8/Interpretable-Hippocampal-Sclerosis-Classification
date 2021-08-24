from HS_util import *
from HS_common import *
from HS_modeling import *
from HS_figure_Scr import *

inputX, inputY, Y_vector, fw, RR = expr_sett()
start = time.time()

Totalaccuracymodel = []
for trainNum in range(TRAIN_MAX):
    foldNum = 0
    FOLD_SEED = FOLD_SEED +1
    historyTrain = historyInfoHS()        
    skf = StratifiedKFold(n_splits=nb_KFold, random_state=FOLD_SEED, shuffle=True)
    for train, validation in skf.split(inputX,Y_vector):

        valid_comp(Y_vector,validation,RR)
        print('train Index:',trainNum+1,foldNum+1,timeCheck())
        modelHS = makeHS_3DCNN_5124(foldNum,DATATYPE,fc1,fc2)

        if(foldNum == 0):
            print (modelHS.summary())
            model_json = modelHS.to_json()
            modelName = './saveModel/[%s%d%s]modelHS%s_%d{F%dK%d}.json' %(SETT,TRIAL,AUG,CONTROLTYPE,modelNum,FOLD_SEED,KERNEL_SEED)
            with open(modelName, "w") as json_file : 
                json_file.write(model_json)

        #save log train & validation set
        logDataSetHS(modelNum,trainNum,foldNum,train,validation,fw)

        ### training model ###
        for loopNum in range(loop_epochs):
            if(CONVOLUTION_TYPE != '2D_CNN'):
                inputX=inputX.reshape(imgCount,imgRow,imgCol,imgDepth,1)

            step_epoch= int(len(inputX[train])//nb_batchSize)
            hist = modelHS.fit_generator(generatorHS(inputX[train], inputY[train], nb_batchSize,AUG,CONTROLTYPE), epochs=nb_epochs, verbose=verboseType, callbacks=myCallbacks, validation_data =(inputX[validation],inputY[validation]), shuffle=False,steps_per_epoch=step_epoch)

            tempModel = './saveModel/[%s%d%s]HS%s_%d{F%dK%d}[%d](%3d).h5' %(SETT,TRIAL,AUG,CONTROLTYPE,modelNum,FOLD_SEED,KERNEL_SEED,foldNum,(loopNum+1)*nb_epochs)
            if os.path.isfile(tempModel):
                os.remove(tempModel)
            modelHS.save(tempModel)
            print("Saved model to disk")            
            
            fold_accloss(hist,historyTrain,foldNum,loopNum)
            
        fold_record(historyTrain,foldNum)
        foldNum += 1

        score = modelHS.evaluate(inputX[validation], inputY[validation],batch_size=nb_batchSize)[1]
        resultT2oblcor = (modelHS.predict(inputX[validation], batch_size=nb_batchSize, verbose=0, steps=None))
        historyTrain.accuracyFloat.append(score)
        k_accuracy = '%.3f' %(score)
        historyTrain.accuracy.append(k_accuracy)

        ##delete model & parametor ##
        print('\n\nmodel.fit -> gc.collect() : ', gc.collect())
        del hist, modelHS
        del score, resultT2oblcor

    ## kFold history mean ##
    if STUDYMODE:
        fold_record(historyTrain,foldNum)     
        Totalaccuracymodel = historyMeanHST(historyTrain,Totalaccuracymodel,fw,RR,modelNum,trainNum,foldNum)                                 
    # maxIdx designation
    avr_val_acc = np.mean(historyTrain.val_acc, axis = 0)
    avr_val_acc = np.array(avr_val_acc)
    maxAccIdx = np.argmax(avr_val_acc, axis=0)    
    del avr_val_acc
    del historyTrain
    
### End of training & validation sequence!! ###

foldCnt = 0            
best_model_save(Totalaccuracymodel,maxAccIdx,FOLD_SEED,fw,RR)
MBpred = mul_bin_pred()
for train, validation in skf.split(inputX,Y_vector):

    modelHS = model_load(foldCnt,FOLD_SEED,KERNEL_SEED)
    inputX=inputX.reshape(imgCount,imgRow,imgCol,imgDepth,1)        
    mul_bin_accuracy(MBpred,modelHS,inputX,inputY,validation,fw)
    foldCnt += 1
    del modelHS

iters=100
performance(MBpred, RR,fw)
balance(CONTROLTYPE,SETT,TRIAL,AUG,KERNEL_SEED,iters, imgCountNO_4, imgCountNO_7,imgCountNO_0,imgCountYES_1,imgCountYES_2, RR)
fin_time(start,RR,fw)
RR.close()
fw.close()