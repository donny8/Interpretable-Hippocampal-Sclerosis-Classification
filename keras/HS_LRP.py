from HS_common import *
from HS_modeling import *
from HS_util import *
from HS_figure_Scr import *

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import innvestigate
import innvestigate.utils as iutils

foldNum = 0
nan_lrp = []
inputX, inputY, listFileName, niiInfo, Y_vector, LR= LRP_expr_sett()
LRP_pred = LRP_result()
skf = StratifiedKFold(n_splits=nb_KFold, random_state=FOLD_SEED, shuffle=True)
FOLD_SEED += 1

for train, validation in skf.split(inputX,Y_vector):
    dir_path = './3D_output/%s_%s_%d%s/%s_FOLD_%d' %(CONTROLTYPE,AUG,DATATYPE,LRPMSG,ruleLRP,foldNum)
    if(not(os.path.isdir(dir_path))):
        os.mkdir(dir_path)    

    inputFoldX = inputX[validation]
    inputFoldY = inputY[validation]
    listFileNameFold = listFileName[validation]
    Y_true = Y_vector[validation]

    model = model_load(foldNum,FOLD_SEED,KERNEL_SEED)
    print('model evaluation :',model.evaluate(inputFoldX, inputFoldY,batch_size=nb_batchSize)[1])
    ## Strip softmax layer
    model_wo_softmax = iutils.model_wo_softmax(model)
    model_wo_softmax.compile(optimizer = adam, loss = 'categorical_crossentropy', metrics = ['categorical_accuracy'])
    
    if(ruleLRP=='lrp.alpha_beta'):
        alpha=5; beta=4
        analyzer = innvestigate.create_analyzer(ruleLRP, model_wo_softmax, neuron_selection_mode = "index",alpha=alpha,beta=beta)
    else:
        analyzer = innvestigate.create_analyzer(ruleLRP, model_wo_softmax, neuron_selection_mode = "index")
    print('fold inputX shape', inputFoldX.shape)
    
    ###############################################
    inputStartIdx = 0 ; inputEndIdx = 1
    TotalLRP = np.zeros((imgRow,imgCol,imgDepth,1))

    # The number of the validation data
    for loopCnt in range(len(inputFoldX)):
        prob = model.predict_on_batch(inputFoldX[loopCnt:loopCnt+1])
        prob = np.array(prob)
        prob = np.reshape(prob,nb_classes)

        # The number of the classes
        for neuronCnt in range(nb_classes):
            result = analyzer.analyze(inputFoldX[inputStartIdx:inputEndIdx],neuron_selection=neuronCnt) ## here input data.
            result = np.array(result)    
            if(ruleLRP!='lrp.alpha_1_beta_0'):
                result[0] = np.where((result[0]<0),0,result[0])

            target_analyze = result[0]
            target_label = inputFoldY[loopCnt:loopCnt+1]
            target_name = listFileNameFold[inputStartIdx]
            target_score = prob[neuronCnt]
            TotalLRP, nan_lrp = norm_n_save(target_analyze, target_label, target_name, target_score, neuronCnt, foldNum, prob, nan_lrp, TotalLRP, LRP_pred, niiInfo, LR)
           
        inputStartIdx += 1
        inputEndIdx +=  1
        print('LRP START,END', inputStartIdx,inputEndIdx)
        
    foldNum += 1

LRPavg_n_result(TotalLRP,LRP_pred,nan_lrp,niiInfo,LR)
LR.close()