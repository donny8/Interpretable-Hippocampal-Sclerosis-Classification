import keras
from keras.callbacks import Callback
from keras.callbacks import LearningRateScheduler

from HS_common import *
from HS_figure_Scr import *
import inspect


def writer(temp_log,fw,RR):
    print(temp_log)
    fw.write(temp_log)
    RR.write(temp_log)

def expr_sett():
    
    Results_Report = './log/[%s%d%s]Result_Reports.txt' %(SETT,TRIAL, AUG)
    fileName = './log/[%s%d%s]aLog_%s[%d]{F%dK%d}.txt' %(SETT,TRIAL,AUG,CONTROLTYPE,modelNum,FOLD_SEED,KERNEL_SEED)

    RR = open(Results_Report,'a')
    RR.write('AUG = %s KERNEL_SEED = %d  CONVOLUTION_TYPE = %s  CONTROLTYPE = %s :\n' %(AUG, KERNEL_SEED, CONVOLUTION_TYPE, CONTROLTYPE)) 
    RR.write('\n batch_size : %d \n' %(nb_batchSize))

    inputX=[]; inputY=[]; Y_vector = []
    inputX, inputY = data_single()
    inputX = np.array(inputX); inputY = np.array(inputY)            


    fw = open(fileName,'a')

    #Convolution Type notification
    temp_log = 'Convolution Type : ' + CONVOLUTION_TYPE + '\n'
    writer(temp_log,fw,RR)

    #Control Type notification
    temp_log = 'Control Type : ' + CONTROLTYPE + '\n'
    writer(temp_log,fw,RR)

    temp_log = 'Kernel Seed :' + str(KERNEL_SEED) + '\n'
    writer(temp_log,fw,RR)

    if (MULTI_CHECK):
        for i in range(imgCountNO_4):
            Y_vector.append(4)
        for i in range(imgCountNO_7):
            Y_vector.append(7)
        for i in range(imgCountNO_0):
            Y_vector.append(0)
        for i in range(imgCountYES_1):
            Y_vector.append(1)
        for i in range(imgCountYES_2):
            Y_vector.append(2)
        for i in range(imgCountYES_3):
            Y_vector.append(3)
    else :
        for i in range(imgCountNO):
            Y_vector.append(0)
        for i in range(imgCountYES):
            Y_vector.append(1)
    Y_vector = np.array(Y_vector)

    if DEBUG:
        print(Y_vector)
        print(Y_vector.shape)
        debugMessage()
        
    return inputX, inputY, Y_vector, fw, RR


def valid_comp(Y_vector,validation,RR):
    checker = []
    print('validation len check:',len(validation))
    checker.append(len(validation))

    check = Y_vector[validation]

    print('four:',len(check[check==4]))
    checker.append(len(check[check==4]))

    print('seven:',len(check[check==7]))
    checker.append(len(check[check==7]))

    print('zero:',len(check[check==0]))
    checker.append(len(check[check==0]))


    print('one:',len(check[check==1]))
    checker.append(len(check[check==1]))

    print('two:',len(check[check==2]))
    checker.append(len(check[check==2]))

    print('three:',len(check[check==3]))
    checker.append(len(check[check==3]))

    RR.write('\n\nvalidation')
    RR.write(str(checker))


def lr_scheduler(epoch, lr):
    if(LEARNING_MODE == 'FAST'):
        # learning_rate setting is "1e-2"
        decay_rate = 0.5623413
        decay_step = nb_epochs - 1 #default 100
    elif(LEARNING_MODE == 'FINETUNE'):
        # learning_rate setting is "1e-3"
        decay_rate = 0.1
        decay_step = 100
    else: #MID
        decay_rate = 0.316 # root 0.1 is 0.316
        decay_step = nb_epochs - 1 #default 100

    if epoch % decay_step == 0 and epoch:
        print('!! lr decay %0.7f -> %0.7f!!:' %(lr,lr * decay_rate))
        return lr * decay_rate
    return lr


class Record(Callback):
    """Get the best model at the end of training.
    https://github.com/keras-team/keras/issues/2768
	# Arguments
        monitor: quantity to monitor.
        verbose: verbosity mode, 0 or 1.
        mode: one of {auto, min, max}.
            The decision
            to overwrite the current stored weights is made
            based on either the maximization or the
            minimization of the monitored quantity. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.
        period: Interval (number of epochs) between checkpoints.
	# Example
		callbacks = [GetBest(monitor='val_acc', verbose=1, mode='max')]
		mode.fit(X, y, validation_data=(X_eval, Y_eval),
                 callbacks=callbacks)
    """

    def __init__(self, vloss='val_loss', vacc='val_acc',tloss='train_loss',tacc='train_acc', verbose=0,
                 mode='auto', period=1, address='graph'):
        super(Record, self).__init__()
        self.vloss = vloss
        self.vacc = vacc
        self.tloss = tloss
        self.tacc = tacc
        self.verbose = verbose
        self.period = period
        self.address = address

                
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        vl = logs.get(self.vloss)
        va = logs.get(self.vacc)
        tl = logs.get(self.tloss)
        ta = logs.get(self.tacc)
        if vl is None:
            warnings.warn('Can pick best model only with %s available, '
                          'skipping.' % (self.monitor), RuntimeWarning)
        else:
            GR = open(self.address,'a+')
            array = '%f, %f, %f, %f\n' %(ta, tl, va, vl)
            GR.write(array) 
            GR.close()

class GetBest(Callback):
    """Get the best model at the end of training.
    https://github.com/keras-team/keras/issues/2768
	# Arguments
        monitor: quantity to monitor.
        verbose: verbosity mode, 0 or 1.
        mode: one of {auto, min, max}.
            The decision
            to overwrite the current stored weights is made
            based on either the maximization or the
            minimization of the monitored quantity. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.
        period: Interval (number of epochs) between checkpoints.
	# Example
		callbacks = [GetBest(monitor='val_acc', verbose=1, mode='max')]
		mode.fit(X, y, validation_data=(X_eval, Y_eval),
                 callbacks=callbacks)
    """

    def __init__(self, monitor='val_loss', verbose=0,
                 mode='auto', period=1):
        super(GetBest, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.period = period
        self.best_epochs = 0
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('GetBest mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf
                
    def on_train_begin(self, logs=None):
        self.best_weights = self.model.get_weights()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            #filepath = self.filepath.format(epoch=epoch + 1, **logs)
            current = logs.get(self.monitor)
            if current is None:
                warnings.warn('Can pick best model only with %s available, '
                              'skipping.' % (self.monitor), RuntimeWarning)
            else:
                if self.monitor_op(current, self.best):
                    if self.verbose > 0:
                        print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                              ' storing weights.'
                              % (epoch + 1, self.monitor, self.best,
                                 current))
                    self.best = current
                    self.best_epochs = epoch + 1
                    self.best_weights = self.model.get_weights()
                # else:
                #     if self.verbose > 0:
                #         print('\nEpoch %05d: %s did not improve' %
                #               (epoch + 1, self.monitor))            
        # print('LR value is %d' %(epoch + 1))            
    def on_train_end(self, logs=None):
        if self.verbose > 0:
            print('Using epoch %05d with %s: %0.5f' % (self.best_epochs, self.monitor,
                                                       self.best))
        self.model.set_weights(self.best_weights)

Graph_Report = os.getcwd()+'/graph/[%s%d%s%d]Graph_Reports.txt' %(SETT,TRIAL, AUG,KERNEL_SEED)
myCallbacks = [ GetBest(monitor='val_categorical_accuracy', verbose=1, mode='max'), LearningRateScheduler(lr_scheduler, verbose=1)
                            ,Record(vloss='val_loss',tloss='loss',vacc='val_categorical_accuracy',tacc='categorical_accuracy', address=Graph_Report) ]

        
def data_single():
    inputX = []; inputY = []
    for idx, f in enumerate(categories):
        
        label = [ 0 for i in range(nb_classes) ]
        label[idx] = 1
    
        image_dir = dirDataSet + "/" + f
        print(image_dir)
        for (imagePath, dir, files) in sorted(os.walk(image_dir)):
            ## read 3D nii file ###
            if(imagePath == image_dir):                
                if(imagePath == image_dir):
                    if DEBUG:
                        print("%s" % (imagePath))

                    # image reading #
                    files = sorted(glob.glob(imagePath + "/*.nii.gz"))

                for i, fname in enumerate(files):
                    if DEBUG:
                        print(fname) #print(i)
                    img = nib.load(fname).get_fdata()
                    inputX.append(img)
                    inputY.append(label)
                
    return inputX, inputY

def flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x

def generatorHS(x_data, y_label, batch_size, augmentation, CONTROLTYPE):
    size = len(x_data)
    while True:
        # permutation : shuffle datsets without touching the original
        idx = np.random.permutation(size)
        x_data=x_data[idx]
        y_label=y_label[idx]

        for i in range(size // batch_size):
            temp_data = []
            temp_label = []
            x_batch = x_data[i*batch_size:(i+1)*batch_size]
            y_batch = y_label[i*batch_size:(i+1)*batch_size]
            
            if (augmentation=='hflip'): 
                temp_data = flip_axis(x_batch,1) # 0:batch  1:left & right  2:forward & backward  3:up & down 
                
                if(MULTI_CHECK):
                    for label_idx in range(batch_size):
                        if(y_batch[label_idx][1] == 1) :
                            if(CONTROLTYPE=='CLRM')or(CONTROLTYPE=='ZLRM'):
                                temp_label.append([0, 0, 1])
                            else :
                                temp_label.append([0, 0, 1, 0])
                        elif(y_batch[label_idx][2] == 1) :
                            if(CONTROLTYPE=='CLRM')or(CONTROLTYPE=='ZLRM'):
                                temp_label.append([0, 1, 0])
                            else :
                                temp_label.append([0, 1, 0, 0])
                        else :
                            temp_label.append(y_batch[label_idx])
                elif((CONTROLTYPE == 'CZ')or(CONTROLTYPE == 'CC')):
                    for label_idx in range(batch_size):
                        temp_label.append(y_batch[label_idx])
                        
                x_batch = np.concatenate((x_batch,temp_data))
                y_batch = np.concatenate((y_batch,temp_label))
 
                del temp_data
                del temp_label

                
            yield x_batch, y_batch
            del x_batch
            del y_batch


            
def change(array) :
    length = len(array)
    for i in range(length):
        if( (array[i] == 2) or array[i] == 3 ):
            array[i] = 1
    return array




def LRP_data_sig():
    inputX = []; inputY = []; listFileName=[]
    for idx, f in enumerate(categories):
        
        label = [ 0 for i in range(nb_classes) ]
        label[idx] = 1
    
        image_dir = dirDataSet + "/" + f
        print(image_dir)
        for (imagePath, dir, files) in sorted(os.walk(image_dir)):
            ## read 3D nii file ###
            if(imagePath == image_dir):
                if DEBUG:
                    print("%s" % (imagePath))
                    
                # image reading #
                files = sorted(glob.glob(imagePath + "/*.nii.gz"))
                listFileName += files
    
            for i, fname in enumerate(files):
                if DEBUG:
                    print(fname) #print(i)
                if(i==0):
                    niiInfo = nib.load(fname)
                img = nib.load(fname).get_fdata()
                inputX.append(img)
                inputY.append(label)
    return inputX, inputY, listFileName, niiInfo

def nancheck(LRP):
    if( (math.isnan(LRP.max())) | (math.isnan(LRP.min() ))) :
        return True
    else :
        return False

def AUROC_SET(yPrediction, yLabelPrediction,yIdxPrediction,Y_true, fw, RR, msg):
    yPrediction = np.array(yPrediction)
    yLabelPrediction = np.array(yLabelPrediction)
    yIdxPrediction = np.array(yIdxPrediction)

    #sorting yPrediction
    tempMatrix1 = [0]*imgCount
    tempMatrix2 = [0]*imgCount

    for idx in range(imgCount):
        tempMatrix1[yIdxPrediction[idx]] = yPrediction[idx]
        tempMatrix2[yIdxPrediction[idx]] = yLabelPrediction[idx]    

    tempMatrix1 = np.array(tempMatrix1)

    temp_log = '{}'.format(100*np.round_(tempMatrix1,3))
    fw.write(temp_log)

    yPrediction = tempMatrix1
    yLabelPrediction = tempMatrix2
    

    if(msg!=None):
        temp_log = msg
        writer(temp_log,fw,RR)

    # Precision, Recall, F1_Score
    print('f1_score:        %.3f' %(f1_score(Y_true, yLabelPrediction))) # 0.93995
    print('precision_score: %.3f' %precision_score(Y_true, yLabelPrediction))   # 0.96697
    print('recall_score:    %.3f' %recall_score(Y_true, yLabelPrediction))      # 0.91440
    print('roc_auc_score:   %.3f' %roc_auc_score(Y_true, yPrediction))           # 0.98906


    tn, fp, fn, tp = confusion_matrix(Y_true, yLabelPrediction).ravel()
    temp_log = 'Softmax Sum Accuracy!! %.3f' %((tn+tp)/(tn+tp+fn+fp))
    RR.write(temp_log)
    print('tn %d, fp %d, fn %d, tp %d'%(tn, fp, fn, tp))
    RR.write('\ntn %d, fp %d, fn %d, tp %d'%(tn, fp, fn, tp)) 
    fw.write('\ntn %d, fp %d, fn %d, tp %d'%(tn, fp, fn, tp))
    temp_log = '\nSensitivity : %0.3f  Specificity : %0.3f' %(tp/(fn+tp),tn/(tn+fp))
    writer(temp_log,fw,RR)
    temp_log = '\nprecision_score %0.3f, recall_score %0.3f' %(precision_score(Y_true, yLabelPrediction),recall_score(Y_true, yLabelPrediction))
    writer(temp_log,fw,RR)
    temp_log = '\nf1_score %0.3f, roc_auc_score %0.3f' %(f1_score(Y_true, yLabelPrediction),roc_auc_score(Y_true, yPrediction))
    writer(temp_log,fw,RR)

    

    
def ROC_family(Y_true,yLabelPrediction,yPrediction,imgCountNO,imgCountYES,SETT,TRIAL,AUG,KERNEL_SEED,msg):
    tn, fp, fn, tp = confusion_matrix(Y_true, yLabelPrediction).ravel()

    precisions, recalls, thresholds = precision_recall_curve(Y_true, yPrediction)

    plt.figure(figsize=(20, 10))
    temp_log = 'Precison : %0.2f(TP[%d],FP[%d]' %(tp/(tp+fp),tp,fp)
    plt.subplot(211)
    plt.title(temp_log)
    index = np.arange(imgCountNO)

    plt.bar(index,100-(100*np.round_(yPrediction[0:imgCountNO],3)),width=0.5,color='r')
    plt.axis([1, imgCountNO, 0, 100])
    plt.xticks(np.arange(0,imgCountYES,1))
    plt.yticks(np.arange(0,101,10))
    plt.grid(True)

    temp_log = 'Specificity : %0.2f(TN[%d],FP[%d]' %(tn/(tn+fp),tn,fp)
    plt.xlabel(temp_log)
    plt.ylabel('No')

    plt.subplot(212)
    index = np.arange(imgCountYES)
    plt.bar(index,100*np.round_(yPrediction[imgCountNO:imgCount],3),width=0.5,color='b')
    plt.axis([1, imgCountYES, 0, 100])
    plt.xticks(np.arange(0,imgCountYES,1))
    plt.yticks(np.arange(0,101,10))

    plt.grid(True)

    temp_log = 'Sensitivity : %0.2f(TP[%d],FN[%d]' %(tp/(tp+fn),tp,fn)
    plt.xlabel(temp_log)
    plt.ylabel('Yes')

    saveFileName = './graph/[%s%d%s%d]PredictionPercent%s.png' %(SETT,TRIAL,AUG,KERNEL_SEED,msg)
    plt.savefig(saveFileName)


    plt.figure(figsize=(12, 5))
    plt.subplot(121)

    plt.plot(thresholds, precisions[:-1], label='Precision')
    plt.plot(thresholds, recalls[:-1], label='Recall')
    plt.xlabel('Thresholds')
    plt.legend()

    plt.subplot(122)
    plt.plot(recalls,precisions)
    plt.xlabel('Recalls')
    plt.ylabel('Precisions')
    plt.axis([0, 1.00, 0, 1.00])
    # ticks every 0.1
    plt.xticks(np.arange(0,1.1,0.1))
    plt.yticks(np.arange(0,1.1,0.1))
    plt.grid(True)

    saveFileName = './graph/[%s%d%s%d]PR Curve%s.png' %(SETT,TRIAL,AUG,KERNEL_SEED,msg)
    plt.savefig(saveFileName)


    plt.figure(figsize=(5, 5))

    fpr, tpr, thresholds = roc_curve(Y_true, yPrediction)   
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.axis([0, 1.00, 0, 1.00])
    # ticks every 0.1
    plt.xticks(np.arange(0,1.1,0.1))
    plt.yticks(np.arange(0,1.1,0.1))
    plt.grid(True)

    saveFileName = './graph/[%s%d%s%d]ROC Curve%s.png' %(SETT,TRIAL,AUG,KERNEL_SEED,msg)
    plt.savefig(saveFileName)
    
def LRPrint(normalizeLRP,name,niiInfo):
    ni_img = nib.Nifti1Image(normalizeLRP, niiInfo.affine, niiInfo.header)
    SaveFileName = './3D_output/%s.nii' %(name)
    GzSaveFileName = './3D_output/%s.nii.gz' %(name)
    nib.save(ni_img, SaveFileName)
    # Open output file.
    with open(SaveFileName, "rb") as file_in:
        # Write output.
        with gzip.open(GzSaveFileName, "wb") as file_out:
            file_out.writelines(file_in)
    if os.path.isfile(SaveFileName):
        os.remove(SaveFileName)
        
def ARG_print(KERNEL_SEED = KERNEL_SEED, SETT = SETT, DATATYPE = DATATYPE, CONTROLTYPE = CONTROLTYPE, AUG = AUG, BATCH = BATCH) :
    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    del values["frame"]
    return (values)

def fold_accloss(hist,historyTrain,foldNum,loopNum):
                
    if(loopNum == 0):
        historyTrain.val_loss[foldNum]    = hist.history['val_loss']
        historyTrain.val_acc[foldNum]     = hist.history['val_categorical_accuracy']
        historyTrain.train_loss[foldNum]  = hist.history['loss']
        historyTrain.train_acc[foldNum]   = hist.history['categorical_accuracy']
    else:
        historyTrain.val_loss[foldNum]    += hist.history['val_loss']
        historyTrain.val_acc[foldNum]     += hist.history['val_categorical_accuracy']
        historyTrain.train_loss[foldNum]  += hist.history['loss']
        historyTrain.train_acc[foldNum]   += hist.history['categorical_accuracy']

def fold_record(historyTrain,foldNum):
        
    if(foldNum < 5):        
        temp_array_1 = np.array(historyTrain.val_acc[foldNum])
        temp_array_2 = np.array(historyTrain.val_loss[foldNum])
        temp_array_3 = np.array(historyTrain.train_acc[foldNum])
        temp_array_4 = np.array(historyTrain.train_loss[foldNum])

        scipy.io.savemat('./graph/[%s%d%s%d]historyTrain.val_acc[%d].mat' %(SETT,TRIAL,AUG,KERNEL_SEED,foldNum),{"mydata": temp_array_1})
        scipy.io.savemat('./graph/[%s%d%s%d]historyTrain.val_loss[%d].mat' %(SETT,TRIAL,AUG,KERNEL_SEED,foldNum),{"mydata": temp_array_2})
        scipy.io.savemat('./graph/[%s%d%s%d]historyTrain.train_acc[%d].mat' %(SETT,TRIAL,AUG,KERNEL_SEED,foldNum),{"mydata": temp_array_3})
        scipy.io.savemat('./graph/[%s%d%s%d]historyTrain.train_loss[%d].mat' %(SETT,TRIAL,AUG,KERNEL_SEED,foldNum),{"mydata": temp_array_4})

    else:
        temp_array_1 = np.array(historyTrain.val_acc)
        temp_array_2 = np.array(historyTrain.val_loss)
        temp_array_3 = np.array(historyTrain.train_acc)
        temp_array_4 = np.array(historyTrain.train_loss)

        scipy.io.savemat('./graph/[%s%d%s%d]historyTrain.val_acc.mat' %(SETT,TRIAL,AUG,KERNEL_SEED),{"mydata": temp_array_1})
        scipy.io.savemat('./graph/[%s%d%s%d]historyTrain.val_loss.mat' %(SETT,TRIAL,AUG,KERNEL_SEED),{"mydata": temp_array_2})
        scipy.io.savemat('./graph/[%s%d%s%d]historyTrain.train_acc.mat' %(SETT,TRIAL,AUG,KERNEL_SEED),{"mydata": temp_array_3})
        scipy.io.savemat('./graph/[%s%d%s%d]historyTrain.train_loss.mat' %(SETT,TRIAL,AUG,KERNEL_SEED),{"mydata": temp_array_4})

    del temp_array_1
    del temp_array_2
    del temp_array_3
    del temp_array_4

    
def best_model_save(Totalaccuracymodel,maxAccIdx,FOLD_SEED,fw,RR):
    temp_log = '\n\n Total Average Accuracy model: %.3f\n' %(average(Totalaccuracymodel))
    writer(temp_log,fw,RR)
    ## best epoch weight search ##
    bestModelIdx = maxAccIdx//nb_epochs

    temp_log = 'maxAccIdx %d, bestModelIdx %d' %(maxAccIdx,(bestModelIdx+1)*nb_epochs) 
    writer(temp_log,fw,RR)

    currentIdx = 0
    for cnt in range(nb_KFold):
        for loopNum in range(loop_epochs):
            saveModelName = os.getcwd()+'/saveModel/[%s%d%s]HS%s_%d{F%dK%d}[%d](%3d).h5' %(SETT,TRIAL,AUG,CONTROLTYPE,
                                modelNum,FOLD_SEED,KERNEL_SEED,cnt,(loopNum+1)*nb_epochs)
            changeModelName = os.getcwd()+'/saveModel/[%s%d%s]HS%s_%d{F%dK%d}[%d](best).h5'%(SETT,TRIAL,AUG,CONTROLTYPE,
                                modelNum,FOLD_SEED,KERNEL_SEED,cnt)
            
            if os.path.isfile(saveModelName):
                if(currentIdx % loop_epochs == bestModelIdx):
                    os.rename(saveModelName, changeModelName)
                else:
                    os.remove(saveModelName)
            currentIdx = currentIdx + 1

def mul_bin_accuracy(MBpred,modelHS,normalizingInputX,inputY,validation,fw):
    # To get the binary Y one-hot vectors
    lenY = len(inputY) # number of inputY   shape : (n,2,2)
    binary_Y = np.zeros(lenY*2)
    binary_Y = binary_Y.reshape(lenY,2)
    for idx in range(lenY):
        if(inputY[idx][0]==1):
            binary_Y[idx][0]=1
        else :
            binary_Y[idx][1]=1

    resultT2oblcor = (modelHS.predict(normalizingInputX[validation], batch_size=nb_batchSize, verbose=0, steps=None))

    lenprob = len(resultT2oblcor)
    result1 = calcPredictionHS(resultT2oblcor,inputY,validation,fw)
    MBpred.p.append(result1)

    temp_prob = np.zeros(lenprob*2)
    temp_prob = temp_prob.reshape(lenprob,2)
    for idx in range(lenprob):
        temp_prob[idx][0] = resultT2oblcor[idx][0]
        temp_prob[idx][1] = sum(resultT2oblcor[idx][1::])
    result3 = calcPredictionHS(temp_prob,binary_Y,validation,fw)
    MBpred.pp.append(result3)

    if(MODE=='None'):
        ## recall-precisioin AUROC CURVE Preprocess ##
        for Num in range(len(validation)): 
            MBpred.yLabelPrediction.append(np.argmax(resultT2oblcor[Num], axis=0))
            MBpred.yPrediction.append(resultT2oblcor[Num][1]/100)
            MBpred.yIdxPrediction.append(validation[Num])
            MBpred.yLabelPrediction_B.append(np.argmax(temp_prob[Num], axis=0))
            MBpred.yPrediction_B.append(temp_prob[Num][1]/100)
    else :
        return temp_prob

        
    ##delete model & parametor ##
    print('\n\nmodel.fit -> gc.collect() : ', gc.collect())

    del temp_prob            
    del resultT2oblcor
     

def model_load(foldCnt,FOLD_SEED,KERNEL_SEED):
    ## model load ##
    modelName = os.getcwd()+'/saveModel/[%s%d%s]modelHS%s_%d{F%dK%d}.json' %(SETT,TRIAL,AUG,CONTROLTYPE,modelNum,FOLD_SEED,KERNEL_SEED)

    if os.path.isfile(modelName):
        json_file = open(modelName, "r") 
        loaded_model_json = json_file.read() 
        json_file.close() 
        model = model_from_json(loaded_model_json)        
    else:
          print('\n!!!warning!!! \n load model file not exist!!')

    weightFileName = os.getcwd()+'/saveModel/[%s%d%s]HS%s_%d{F%dK%d}[%d](best).h5' %(SETT,TRIAL,AUG,CONTROLTYPE,modelNum,FOLD_SEED,KERNEL_SEED,foldCnt)
    print('weightFileName %s' %(weightFileName))
    if os.path.isfile(weightFileName):
        model.load_weights(weightFileName)
    else:
        print('\n!!!warning!!! \n load Weight file not exist!!')
    model.compile(optimizer = adam, loss = 'categorical_crossentropy', metrics = ['categorical_accuracy'])
    
    return model
        
        
    
def performance(MBpred, RR,fw):
    yPrediction = MBpred.yPrediction
    yLabelPrediction = MBpred.yLabelPrediction
    yIdxPrediction = MBpred.yIdxPrediction

    yPrediction_B = MBpred.yPrediction_B
    yLabelPrediction_B = MBpred.yLabelPrediction_B

    if(MODE=='None'):
        p = MBpred.p; pp = MBpred.pp
    
    
    ### AUROUC CURVE & CONFUSION MATRIX ###
    Y_true = []
    Y_true_B = []

    if(MODE=='None'):
        p = np.round(p,2)
        pp = np.round(pp,2)

        temp_log = '\n\n Multi Class: {}'.format(p)
        writer(temp_log,fw,RR)

        temp_log = '\n Multi Class Average: %f' %(np.round(average(p),3)) 
        writer(temp_log,fw,RR)

        temp_log = '\n\n Softmax Probaility Sum: {}'.format(pp)
        writer(temp_log,fw,RR)

        temp_log = '\n Soft Sum Average: %f' %(np.round(average(pp),3)) 
        writer(temp_log,fw,RR)

    for i in range(imgCountNO):
        Y_true.append(0)
        Y_true_B.append(0)
    for i in range(imgCountYES_1):
        Y_true.append(1)
        Y_true_B.append(1)
    for i in range(imgCountYES_2):
        Y_true.append(2)
        Y_true_B.append(1)
    for i in range(imgCountYES_3):
        Y_true.append(3)
        Y_true_B.append(1)

    Y_true = np.array(Y_true)
    Y_true_B = np.array(Y_true_B)

    yPrediction = np.array(yPrediction)
    yLabelPrediction = np.array(yLabelPrediction)
    yIdxPrediction = np.array(yIdxPrediction)


    #sorting yPrediction
    tempMatrix1 = [0]*imgCount
    tempMatrix2 = [0]*imgCount
    for idx in range(imgCount):
        if(MODE=='None'):
            tempMatrix1[yIdxPrediction[idx]] = yPrediction[idx]
        tempMatrix2[yIdxPrediction[idx]] = yLabelPrediction[idx]  

    tempMatrix1 = np.array(tempMatrix1)
    if(MODE=='None'):
        yPrediction = tempMatrix1
    yLabelPrediction = tempMatrix2

    multi_class_confusion = confusion_matrix(Y_true, yLabelPrediction)
    tntp = np.diag(multi_class_confusion)
    correct = np.sum(tntp)
    print(tntp)
    temp_log = '\n\n [Multi Class Confusion]\n'
    writer(temp_log,fw,RR)
    temp_log = str(tntp.ravel())
    writer(temp_log,fw,RR)
    temp_log = '\nMulti Class Accuracy!! %.3f\n' %(correct/len(Y_true))    
    writer(temp_log,fw,RR)
    AUROC_SET(yPrediction_B, yLabelPrediction_B, yIdxPrediction,Y_true_B,fw,RR,'\n\n [Softmax Probability Sum] \n')


    yPrediction_B = np.array(yPrediction_B)
    yLabelPrediction_B = np.array(yLabelPrediction_B)

    #sorting yPrediction
    tempMatrixB1 = [0]*imgCount
    tempMatrixB2 = [0]*imgCount

    for idx in range(imgCount):
        tempMatrixB1[yIdxPrediction[idx]] = yPrediction_B[idx]
        tempMatrixB2[yIdxPrediction[idx]] = yLabelPrediction_B[idx]
    tempMatrixB1 = np.array(tempMatrixB1)
    yPrediction_B = tempMatrixB1
    yLabelPrediction_B = tempMatrixB2

    MBpred.yLabelPrediction = yLabelPrediction
    MBpred.yLabelPrediction_B = yLabelPrediction_B
    

    scipy.io.savemat('./graph/[%s%d%s%d]Y_true_M.mat' %(SETT,TRIAL,AUG,KERNEL_SEED),{"mydata": Y_true})
    scipy.io.savemat('./graph/[%s%d%s%d]yLabelPrediction_M.mat' %(SETT,TRIAL,AUG,KERNEL_SEED),{"mydata": yLabelPrediction})
    scipy.io.savemat('./graph/[%s%d%s%d]Y_true_B.mat' %(SETT,TRIAL,AUG,KERNEL_SEED),{"mydata": Y_true_B})
    scipy.io.savemat('./graph/[%s%d%s%d]yLabelPrediction_B.mat' %(SETT,TRIAL,AUG,KERNEL_SEED),{"mydata": yLabelPrediction_B})
    if(MODE=='None'):
        scipy.io.savemat('./graph/[%s%d%s%d]yPrediction_M.mat' %(SETT,TRIAL,AUG,KERNEL_SEED),{"mydata": yPrediction})
        scipy.io.savemat('./graph/[%s%d%s%d]yPrediction_B.mat' %(SETT,TRIAL,AUG,KERNEL_SEED),{"mydata": yPrediction_B})
    else:
        ROC_family(Y_true_B,yLabelPrediction_B,yPrediction_B,imgCountNO,imgCountYES,SETT,TRIAL,AUG,KERNEL_SEED,'_ProbSum_%s'%(MODE))



def loader(SETT,TRIAL,AUG,KERNEL_SEED,LRMB):
    if(LRMB!='M'):
        yT = io.loadmat('./graph/[%s%d%s%d]Y_true_%s.mat' %(SETT,TRIAL,AUG,KERNEL_SEED,LRMB))
        yP = io.loadmat('./graph/[%s%d%s%d]yPrediction_%s.mat' %(SETT,TRIAL,AUG,KERNEL_SEED,LRMB))
        yL = io.loadmat('./graph/[%s%d%s%d]yLabelPrediction_%s.mat' %(SETT,TRIAL,AUG,KERNEL_SEED,LRMB))

        Y_true = np.transpose(yT['mydata'])
        yPrediction = np.transpose(yP['mydata'])
        yLabelPrediction = np.transpose(yL['mydata'])
    
        return Y_true, yPrediction, yLabelPrediction
    else: 
        yT = io.loadmat('./graph/[%s%d%s%d]Y_true_%s.mat' %(SETT,TRIAL,AUG,KERNEL_SEED,LRMB))
        yL = io.loadmat('./graph/[%s%d%s%d]yLabelPrediction_%s.mat' %(SETT,TRIAL,AUG,KERNEL_SEED,LRMB))

        Y_true = np.transpose(yT['mydata'])
        yLabelPrediction = np.transpose(yL['mydata'])
    
        return Y_true, yLabelPrediction
    

    
               # No  No_Yes, Yes
def random_sample(idx0,idx1,idx2,num,Y_true,yPrediction,yLabelPrediction,expr):
    if((expr=='L')or(expr=='R')):
        idxNN = np.random.choice(idx0, num, replace=False)
        idxNY = np.random.choice(idx1, num, replace=False)
        idxN = np.hstack([idxNN, idxNY]) 
            
        true_NO = Y_true[idxN]
        true_YES = Y_true[idx2]
        true = np.vstack([true_NO, true_YES])

        label_NO = yLabelPrediction[idxN]
        label_YES = yLabelPrediction[idx2]
        label = np.vstack([label_NO,label_YES])

        pred_NO = yPrediction[idxN]
        pred_YES = yPrediction[idx2]
        pred = np.vstack([pred_NO,pred_YES])
        return true, label, pred
    
    elif(expr=='M'):
        idxM0 = np.random.choice(idx0, num, replace=False)
        idxM1 = np.random.choice(idx1, num, replace=False)
        idxM2 = idx2
        
        true_NO = Y_true[idxM0]
        true_YES1 = Y_true[idxM1]
        true_YES2 = Y_true[idxM2]
        true = np.vstack([true_NO, true_YES1, true_YES2])

        label_NO = yLabelPrediction[idxM0]
        label_YES1 = yLabelPrediction[idxM1]
        label_YES2 = yLabelPrediction[idxM2]
        label = np.vstack([label_NO, label_YES1, label_YES2])

        return true, label
    
    elif(expr=='B'):
        idxB0 = np.random.choice(idx0,num*2,replace=False)
        idxB1 = np.random.choice(idx1,num,replace=False)
        idxBY = np.hstack([idxB1,idx2])
        
        true_NO = Y_true[idxB0]
        true_YES = Y_true[idxBY]
        true = np.vstack([true_NO,true_YES])
        
        label_NO = yLabelPrediction[idxB0]
        label_YES1 = yLabelPrediction[idxBY]
        label = np.vstack([label_NO,label_YES1])

        pred_NO = yPrediction[idxB0]
        pred_YES = yPrediction[idxBY]
        pred = np.vstack([pred_NO,pred_YES])
        return true, label, pred


def balance(CONTROLTYPE,SETT,TRIAL,AUG,KERNEL_SEED,iters, C4,C7,C0,C1,C2, RR):
    np.random.seed(KERNEL_SEED)
    Y_true_M,                yLabelPrediction_M = loader(SETT,TRIAL,AUG, KERNEL_SEED,"M")
    Y_true_B, yPrediction_B, yLabelPrediction_B = loader(SETT,TRIAL,AUG, KERNEL_SEED,"B")
    yPrediction_M = []
    
    idx0 = np.arange(C0+C4+C7)
    idx1 = np.arange(C1) + C4 + C7 + C0
    idx2 = np.arange(C2) + C4 + C7 + C0 + C1
    right_len = len(idx2)
    
    multi_set = []  ;  multi_perf = []
    binary_set = [] ;  binary_perf = []
    tnn_set = [] ; fpn_set = []
    tpl_set = [] ; fnl_set = []
    tpr_set = [] ; fnr_set = []
    
    for i in range(iters):
        M_true, M_label         = random_sample(idx0,idx1,idx2,len(idx2), Y_true_M, yPrediction_M, yLabelPrediction_M,'M' )
        B_true, B_label, B_pred = random_sample(idx0,idx1,idx2,len(idx2), Y_true_B, yPrediction_B, yLabelPrediction_B,'B' )

        
        multi_class_confusion = confusion_matrix(M_true, M_label)
        tntp = np.diag(multi_class_confusion)
        correct = np.sum(tntp)
        acc = correct/len(M_true)
        multi_set.append([tntp[0],tntp[1],tntp[2]])
        multi_perf.append(acc)
        
        tn, fp, fn, tp = confusion_matrix(B_true, B_label).ravel()
        acc = (tn+tp)/(tn+fp+fn+tp)
        auc = roc_auc_score(B_true, B_pred)
        binary_set.append([tn,fp,fn,tp])
        binary_perf.append([acc,auc])

        
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
        
    multi_set = np.round(np.average(multi_set,axis=0),1)
    binary_set = np.round(np.average(binary_set,axis=0),1)
    multi_perf = np.round(np.average(multi_perf,axis=0),3)
    binary_perf = np.round(np.average(binary_perf,axis=0),3)
    
    tnn = np.round(np.average(tnn_set,axis=0),1)
    fpn = np.round(np.average(fpn_set,axis=0),1)
    tpl = np.round(np.average(tpl_set,axis=0),1)
    fnl = np.round(np.average(fnl_set,axis=0),1)
    tpr = np.round(np.average(tpr_set,axis=0),1)
    fnr = np.round(np.average(fnr_set,axis=0),1)
    
    RR.write('\n\n ======== [BALANCE] ========')    
    RR.write('\n\n[MULTI]')
    RR.write('\nNo Left Right\n')
    RR.write(str(multi_set))
    RR.write('\nacc : ')
    RR.write(str(multi_perf))
    
    RR.write('\n\n[BINARY]')
    RR.write('\ntn fp fn tp\n')
    RR.write(str(binary_set))
    RR.write('\nacc : ')
    test = (binary_set[0] + binary_set[3]) / np.sum(binary_set)
    test = np.round(test,3)
    RR.write(str(test))
    RR.write('\nauc :')
    RR.write(str(binary_perf[1]))
    RR.write('\nSensitivity : %0.3f    Specificity : %0.3f' %(binary_set[3]/(binary_set[2]+binary_set[3]),binary_set[0]/(binary_set[0]+binary_set[1])))

    RR.write('\n\nTrue Negative NO : %s' %(str(tnn)))
    RR.write('\nFalse Positive NO : %s' %(str(fpn)))
    RR.write('\nNO : %s' %(str(np.round(tnn/(tnn+fpn),3))))

    RR.write('\n\nTrue Positive Left : %s' %(str(tpl)))
    RR.write('\nFalse Negative Left : %s' %(str(fnl)))
    RR.write('\nLeft : %s' %(str(np.round(tpl/(tpl+fnl),3))))

    RR.write('\n\nTrue Positive Right : %s' %(str(tpr)))
    RR.write('\nFalse Negative Right : %s' %(str(fnr)))
    RR.write('\nRight : %s' %(str(np.round(tpr/(tpr+fnr),3))))
    
def fin_time(start,RR,fw):
    
    #Time Calculation
    finish = time.time() - start
    hour = int(finish // 3600)
    minute = int((finish - hour * 3600) // 60)
    second = int(finish - hour*3600 - minute*60)
    timetime =str(hour) +'h '+ str(minute)+'m ' + str(second)+'s'
    temp_log = "\n\n Time Elapse: %s \n\n\n" %(timetime)
    writer(temp_log,fw,RR)


#====================================================================================================#
#============================================= Ensemble =============================================#
#====================================================================================================#

def ens_expr_sett():
    inputX, inputY = data_single()
    arrayInputX = np.array(inputX); arrayInputY = np.array(inputY)
    print('ensemble array shape',arrayInputX.shape, arrayInputY.shape)

    fileName = 'log/[%s%d%s]aLog_%s[%s%d](%d)en%dX%d.txt' %(SETT,TRIAL,AUG,CONTROLTYPE,ENSEMBLE_MODE,modelNum,FOLD_SEED,ENSEMBLE_IMG,ENSEMBLE_NUM)
    fw = open(fileName,'a')
    storage = 'log/ensembleStorage.txt'
    fwSt=open(storage,'a')
    fwSt.write('\n\nTRIAL %d  SETT %s  DATATYPE %d  CONTROLTYPE %s  EnsMODE %s\n' %(TRIAL, SETT, DATATYPE, CONTROLTYPE, MODE)) 

    Y_vector = []

    print('Multi Class No:',imgCountNO)    
    print('\nMulti Class Left:',imgCountYES_1)    
    print('\nMulti Class Right:',imgCountYES_2)    
    print('\nMulti Class Bi:',imgCountYES_3)    
    for i in range(imgCountNO_4):
        Y_vector.append(4)
    for i in range(imgCountNO_7):
        Y_vector.append(7)
    for i in range(imgCountNO_0):
        Y_vector.append(0)
    for i in range(imgCountYES_1):
        Y_vector.append(1)
    for i in range(imgCountYES_2):
        Y_vector.append(2)
    for i in range(imgCountYES_3):
        Y_vector.append(3)
    Y_vector = np.array(Y_vector)

    if DEBUG:
        print(Y_vector)
        print(Y_vector.shape)
        debugMessage()

    return arrayInputX, arrayInputY, Y_vector, fw, fwSt

def ens_mul_bin(ePredictions, ePredictions_B, eLabels_B,MBpred,cntModel,arrayInputX,arrayInputY,validation,fw):
    prob = cntModel.predict(arrayInputX[validation], batch_size=nb_batchSize, verbose=0, steps=None)
    _ = calcPredictionHS(prob,arrayInputY,validation,fw)
    ePredictions.append(prob)

    temp_prob = mul_bin_accuracy(MBpred,cntModel,arrayInputX,arrayInputY,validation,fw)
    labelPredicts_B = np.argmax(temp_prob, axis=1)
    eLabels_B.append(labelPredicts_B)
    ePredictions_B.append(temp_prob)                

    return ePredictions, ePredictions_B, eLabels_B 


def VOT_ensemble(eLabels,eLabels_P,MBpred,validation,len_ens_models):

    eLabels = np.array(eLabels)                # (ensemble_models,val_len)
    eLabels = np.transpose(eLabels, (1, 0))    # (val_len,ensemble_models)
    vottingArray = eLabels
    eLabels = scipy.stats.mode(eLabels, axis=-1)[0] # (val_len,1)
    eLabels = np.squeeze(eLabels) # (val_len,)
    for Num in range(len(validation)):
        MBpred.yLabelPrediction.append(eLabels[Num])
        vottingPrediction = 0
    
    eLabels_P = np.array(eLabels_P)
    eLabels_P = np.transpose(eLabels_P, (1, 0))
    vottingArray_P = eLabels_P
    eLabels_P = scipy.stats.mode(eLabels_P, axis=-1)[0]
    eLabels_P = np.squeeze(eLabels_P)
    for Num in range(len(validation)):
        MBpred.yLabelPrediction_B.append(eLabels_P[Num])
        vottingPrediction = 0
        for cnt in range(len_ens_models):
            vottingPrediction = vottingPrediction + vottingArray_P[Num][cnt]
        MBpred.yPrediction_B.append(vottingPrediction/len_ens_models)

def error_class(yLabelPrediction,Y_vector,err_cls):
    for idx in range(len(yLabelPrediction)):
        if(Y_vector[idx]==0):
            if(yLabelPrediction[idx]==0):
                err_cls.NO_TRUE+=1
            elif(yLabelPrediction[idx]==1):
                err_cls.NO_LEFT+=1
            elif(yLabelPrediction[idx]==2):
                err_cls.NO_RIGHT+=1
        elif(Y_vector[idx]==4):
            if(yLabelPrediction[idx]==0):
                err_cls.FOUR_TRUE+=1
            elif(yLabelPrediction[idx]==1):
                err_cls.FOUR_LEFT+=1
            elif(yLabelPrediction[idx]==2):
                err_cls.FOUR_RIGHT+=1
        elif(Y_vector[idx]==7):
            if(yLabelPrediction[idx]==0):
                err_cls.SEVEN_TRUE+=1
            elif(yLabelPrediction[idx]==1):
                err_cls.SEVEN_LEFT+=1
            elif(yLabelPrediction[idx]==2):
                err_cls.SEVEN_RIGHT+=1
        elif(Y_vector[idx]==1):
            if(yLabelPrediction[idx]==0):
                err_cls.LEFT_NO+=1
            elif(yLabelPrediction[idx]==1):
                err_cls.LEFT_TRUE+=1
            elif(yLabelPrediction[idx]==2):
                err_cls.LEFT_RIGHT+=1
        elif(Y_vector[idx]==2):
            if(yLabelPrediction[idx]==0):
               err_cls.RIGHT_NO+=1
            elif(yLabelPrediction[idx]==1):
                err_cls.RIGHT_LEFT+=1
            elif(yLabelPrediction[idx]==2):
                err_cls.RIGHT_TRUE+=1

def error_report(err_mul,err_bin,fwSt):
    temp_log = '\n\n [Multi Class Errors]'
    fwSt.write(temp_log)

    temp_log = '\n NO : %0.3f (%d/%d/%d)' %(np.round(err_mul.NO_TRUE/imgCountNO_0,3),err_mul.NO_TRUE,err_mul.NO_LEFT, err_mul.NO_RIGHT)
    fwSt.write(temp_log)

    if(imgCountNO_4!=0):
        temp_log = '\n FOUR : %0.3f (%d/%d/%d)' %(np.round(err_mul.FOUR_TRUE/imgCountNO_4,3),err_mul.FOUR_TRUE,err_mul.FOUR_LEFT, err_mul.FOUR_RIGHT)
        fwSt.write(temp_log)

    temp_log = '\n SEVEN : %0.3f (%d/%d/%d)' %(np.round(err_mul.SEVEN_TRUE/imgCountNO_7,3),err_mul.SEVEN_TRUE,err_mul.SEVEN_LEFT, err_mul.SEVEN_RIGHT)
    fwSt.write(temp_log)

    temp_log = '\n LEFT : %0.3f (%d/%d/%d)' %(np.round(err_mul.LEFT_TRUE/imgCountYES_1,3),err_mul.LEFT_NO, err_mul.LEFT_TRUE,err_mul.LEFT_RIGHT)
    fwSt.write(temp_log)

    temp_log = '\n RIGHT : %0.3f (%d/%d/%d)' %(np.round(err_mul.RIGHT_TRUE/imgCountYES_2,3),err_mul.RIGHT_NO, err_mul.RIGHT_LEFT, err_mul.RIGHT_TRUE)
    fwSt.write(temp_log)


    temp_log = '\n\n [Binary Class Errors]'
    fwSt.write(temp_log)
    temp_log = '\n NO : %0.3f (%d/%d)' %(np.round(err_bin.NO_TRUE/imgCountNO_0,3),err_bin.NO_TRUE,err_bin.NO_LEFT + err_bin.NO_RIGHT)
    fwSt.write(temp_log)

    if(imgCountNO_4!=0):
        temp_log = '\n FOUR : %0.3f (%d/%d)' %(np.round(err_bin.FOUR_TRUE/imgCountNO_4,3),err_bin.FOUR_TRUE,err_bin.FOUR_LEFT + err_bin.FOUR_RIGHT)
        fwSt.write(temp_log)

    temp_log = '\n SEVEN : %0.3f (%d/%d)' %(np.round(err_bin.SEVEN_TRUE/imgCountNO_7,3),err_bin.SEVEN_TRUE,err_bin.SEVEN_LEFT + err_bin.SEVEN_RIGHT)
    fwSt.write(temp_log)

    temp_log = '\n LEFT : %0.3f (%d/%d)' %(np.round(err_bin.LEFT_TRUE/imgCountYES_1,3),err_bin.LEFT_NO,err_bin.LEFT_TRUE+err_bin.LEFT_RIGHT)
    fwSt.write(temp_log)

    temp_log = '\n RIGHT : %0.3f (%d/%d)' %(np.round(err_bin.RIGHT_TRUE/imgCountYES_2,3),err_bin.RIGHT_NO,err_bin.RIGHT_TRUE + err_bin.RIGHT_LEFT)
    fwSt.write(temp_log)

#====================================================================================================#
#=============================================== LRP ================================================#
#====================================================================================================#

def LRP_expr_sett():
    inputX, inputY, listFileName, niiInfo = LRP_data_sig()  
    inputX = np.array(inputX); inputY = np.array(inputY); listFileName = np.array(listFileName)
    inputX=inputX.reshape(imgCount,imgRow,imgCol,imgDepth,1)

    Y_vector = []
    for i in range(imgCountNO_4):
        Y_vector.append(4)
    for i in range(imgCountNO_7):
        Y_vector.append(7)
    for i in range(imgCountNO_0):
        Y_vector.append(0)
    for i in range(imgCountYES_1):
        Y_vector.append(1)
    for i in range(imgCountYES_2):
        Y_vector.append(2)
    for i in range(imgCountYES_3):
        Y_vector.append(3)
    Y_vector = np.array(Y_vector)
            

    dir_path = './3D_output/%s_%s_%d%s' %(CONTROLTYPE,AUG,DATATYPE,LRPMSG)
    if(not(os.path.isdir(dir_path))):
            os.mkdir(dir_path)
    LReport = dir_path+'/[%s%d%s]LRP_Reports.txt' %(SETT,PERCENT, AUG)
    LR = open(LReport,'a')

    return inputX, inputY, listFileName, niiInfo, Y_vector, LR

def LRPsave(ni_img,SaveFileName,GzSaveFileName):
    nib.save(ni_img, SaveFileName)
    # Open output file.
    with open(SaveFileName, "rb") as file_in:
        # Write output.
        with gzip.open(GzSaveFileName, "wb") as file_out:
            file_out.writelines(file_in)
        if os.path.isfile(SaveFileName):
            os.remove(SaveFileName)

def averageLRP(nomalizeLRP,cnt,niiInfo,name,gzname):
    if(cnt!=0):
        nomalizeLRP = nomalizeLRP / cnt
        ni_img = nib.Nifti1Image(nomalizeLRP, niiInfo.affine, niiInfo.header)
        LRPsave(ni_img, name, gzname)

def norm_n_save(target_analyze, target_label, target_name, target_score, neuronCnt, foldNum, prob, nan_lrp, TotalLRP, LRP_pred, niiInfo, LR):
    nomalizeLRP = (target_analyze - np.min(target_analyze))
    nomalizeLRP = nomalizeLRP - nomalizeLRP[backgraoundPixeloffset][backgraoundPixeloffset][backgraoundPixeloffset] #backGround Sub.
    nomalizeLRP = np.where((nomalizeLRP < 0), 0 ,nomalizeLRP) #backGround to "zero"

    # distribution 0~1 #
    nomalizeLRP = nomalizeLRP - np.min(nomalizeLRP) 
    nomalizeLRPori = 100*nomalizeLRP / np.max(nomalizeLRP)
    # cutoff by percent #
    nomalizeLRP = np.where((nomalizeLRPori < PERCENT), 0 ,nomalizeLRPori)
                    
    TotalLRP = TotalLRP + nomalizeLRP
    label = np.argmax(target_label)

    ##  File name patient number ##
    y = target_name[-12:-7]
    if(neuronCnt==0):
        print('\nfilename',target_name)
        print('data : ',y)
        temp = '\n\n%s :' %(y)
        LR.write(temp)

    if(neuronCnt == label):
        if (nancheck(nomalizeLRPori)):
            print('NaN occurred !')
        best = np.argmax(prob)
        print('Prob : ', target_score)
        print('Model Prediction : ',best)
        
        if(neuronCnt==0):
            LRP_pred.NO += 1
            if not(nancheck(nomalizeLRPori)):
                if(best==neuronCnt):
                    LRP_pred.NO_true += 1
                    LRP_pred.NO_avg += nomalizeLRP
        elif(neuronCnt==1):
            LRP_pred.LEFT += 1
            if not(nancheck(nomalizeLRPori)):
                if(best==neuronCnt):
                    LRP_pred.LEFT_true += 1
                    LRP_pred.LRP_pred.LEFT_avg += nomalizeLRP
        elif(neuronCnt==2):
            LRP_pred.RIGHT += 1
            if not(nancheck(nomalizeLRPori)):
                if(best==neuronCnt):
                    LRP_pred.RIGHT_true += 1
                    LRP_pred.RIGHT_avg += nomalizeLRP
                    
    ni_img = nib.Nifti1Image(nomalizeLRP, niiInfo.affine, niiInfo.header)
        
    if(nancheck(nomalizeLRPori)):
        nan_lrp.append(y)

    SaveFileName = './3D_output/%s_%s_%d%s/%s_FOLD_%d/%s_%s_nc%d[%d].nii' %(CONTROLTYPE,AUG,DATATYPE,LRPMSG,ruleLRP,foldNum,y,ruleLRP,neuronCnt,target_score*100)
    GzSaveFileName = './3D_output/%s_%s_%d%s/%s_FOLD_%d/%s_%s_nc%d[%d].nii.gz' %(CONTROLTYPE,AUG,DATATYPE,LRPMSG,ruleLRP,foldNum,y,ruleLRP,neuronCnt,target_score*100)
    LRPsave(ni_img, SaveFileName, GzSaveFileName)

    if(neuronCnt==0):
        LR.write('\nNO : ')
    elif(neuronCnt==1):
        LR.write('\nLEFT : ')           
    elif(neuronCnt==2):
        LR.write('\nRIGHT : ')
    temp_prob = np.round(target_score,2)
    LR.write(str(temp_prob))
    
    return TotalLRP, nan_lrp

def LRPavg_n_result(TotalLRP,LRP_pred,nan_lrp,niiInfo,LR):
    # TotalLRP
    TotalLRP = TotalLRP / imgCount
    TotalLRP = TotalLRP.reshape(imgRow,imgCol,imgDepth,1)
    ni_img = nib.Nifti1Image(TotalLRP, niiInfo.affine, niiInfo.header)

    SaveFileName = './3D_output/%s_%s_%d%s/%s_LRP_average.nii' %(CONTROLTYPE,AUG,DATATYPE,LRPMSG,ruleLRP)
    GzSaveFileName = './3D_output/%s_%s_%d%s/%s_LRP_average.nii.gz' %(CONTROLTYPE,AUG,DATATYPE,LRPMSG,ruleLRP)
    LRPsave(ni_img, SaveFileName, GzSaveFileName)
        
    NO_avg_Name = './3D_output/%s_%s_%d%s/%s_LRP_NO_avg.nii' %(CONTROLTYPE,AUG,DATATYPE,LRPMSG,ruleLRP)
    GZ_NO_avg_Name = './3D_output/%s_%s_%d%s/%s_LRP_NO_avg.nii.gz' %(CONTROLTYPE,AUG,DATATYPE,LRPMSG,ruleLRP)
    averageLRP(LRP_pred.NO_avg,LRP_pred.NO_true,niiInfo,NO_avg_Name,GZ_NO_avg_Name)

    LEFT_avg_Name = './3D_output/%s_%s_%d%s/%s_LRP_LEFT_avg.nii' %(CONTROLTYPE,AUG,DATATYPE,LRPMSG,ruleLRP)
    GZ_LEFT_avg_Name = './3D_output/%s_%s_%d%s/%s_LRP_LEFT_avg.nii.gz' %(CONTROLTYPE,AUG,DATATYPE,LRPMSG,ruleLRP)
    averageLRP(LRP_pred.LEFT_avg,LRP_pred.LEFT_true,niiInfo,LEFT_avg_Name,GZ_LEFT_avg_Name)

    RIGHT_avg_Name = './3D_output/%s_%s_%d%s/%s_LRP_RIGHT_avg.nii' %(CONTROLTYPE,AUG,DATATYPE,LRPMSG,ruleLRP)
    GZ_RIGHT_avg_Name = './3D_output/%s_%s_%d%s/%s_LRP_RIGHT_avg.nii.gz' %(CONTROLTYPE,AUG,DATATYPE,LRPMSG,ruleLRP)
    averageLRP(LRP_pred.RIGHT_avg,LRP_pred.RIGHT_true,niiInfo,RIGHT_avg_Name,GZ_RIGHT_avg_Name)

    temp = '\n\nNO : ' + str(LRP_pred.NO)
    LR.write(temp)
    temp = '\nNO_true : ' + str(LRP_pred.NO_true)
    LR.write(temp)
    temp = '\nNO_acc :' + str(np.around(LRP_pred.NO_true/LRP_pred.NO,3))
    LR.write(temp)

    temp = '\n\nLEFT : ' + str(LRP_pred.LEFT)
    LR.write(temp)
    temp = '\nLEFT_true : ' + str(LRP_pred.LEFT_true)
    LR.write(temp)
    temp = '\nLEFT_acc :' + str(np.around(LRP_pred.LEFT_true/LRP_pred.LEFT,3))
    LR.write(temp)

    temp = '\n\nRIGHT : ' + str(LRP_pred.RIGHT)
    LR.write(temp)
    temp = '\nRIGHT_true : ' + str(LRP_pred.RIGHT_true)
    LR.write(temp)
    temp = '\nRIGHT_acc :' + str(np.around(LRP_pred.RIGHT_true/LRP_pred.RIGHT,3))
    LR.write(temp)

    temp = '\n\nNaN data :%s' %(str(nan_lrp))
    LR.write(temp)