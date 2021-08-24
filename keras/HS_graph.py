from HS_common import *
from HS_util import ROC_family

def figureHistory(foldNum,val_acc,val_loss,train_acc,train_loss,KERNEL_SEED, TRIAL):
    fig = plt.figure()
    ax_acc = fig.add_subplot(111)

    ax_acc.plot(range(nb_epochs*loop_epochs), val_acc, label='acc(%)', color='darkred')
    plt.xlabel('epochs')
    plt.ylabel('acc(%)')
    ax_acc.grid(linestyle='--', color='lavender')
    plt.yticks(np.arange(0.4,1.1,0.1))

    # val_loss maximum value limit
    for index, value in enumerate(val_loss):
        if value > 2.6:
            val_loss[index] = 2.6
    
    ax_loss = ax_acc.twinx()
    ax_loss.plot(range(nb_epochs*loop_epochs), val_loss, label='loss', color='darkblue')
    plt.ylabel('loss')
    ax_loss.yaxis.tick_right()
    plt.yticks(np.arange(0.2,2.8,0.2))

    plt.legend()
    if(foldNum == nb_KFold):
        saveFileName = './graph/[%s%d%s%d]val_mean.png' %(SETT,TRIAL,AUG,KERNEL_SEED)  
    else:
        saveFileName = './graph/[%s%d%s%d]val_%d.png' %(SETT,TRIAL,AUG,KERNEL_SEED,foldNum+1)

    plt.savefig(saveFileName)
    ax_acc.legend()

    fig1 = plt.figure()
    ax_acc1 = fig1.add_subplot(111)

    ax_acc1.plot(range(nb_epochs*loop_epochs), train_acc, label='acc(%)', color='darkred')
    plt.xlabel('epochs')
    plt.ylabel('acc(%)')
    ax_acc1.grid(linestyle='--', color='lavender')
    plt.yticks(np.arange(0.4,1.1,0.1))

    ax_loss = ax_acc1.twinx()
    ax_loss.plot(range(nb_epochs*loop_epochs), train_loss, label='loss', color='darkblue')
    plt.ylabel('loss')
    ax_loss.yaxis.tick_right()
    ax_loss.grid(linestyle='--', color='lavender')

    plt.legend()
    ax_acc.legend()
    if(foldNum == nb_KFold):
        saveFileName = './graph/[%s%d%s%d]train_mean.png' %(SETT,TRIAL,AUG,KERNEL_SEED)  
    else:
        saveFileName = './graph/[%s%d%s%d]train_%d.png' %(SETT,TRIAL,AUG,KERNEL_SEED,foldNum+1)
        
    plt.savefig(saveFileName)

    return 

def historyMeanHS(foldNum,val_loss,val_acc,train_loss,train_acc,KERNEL_SEED, TRIAL):
    avr_val_loss = np.mean(val_loss, axis = 1)
    avr_val_acc = np.mean(val_acc, axis = 1)
    avr_train_loss = np.mean(train_loss, axis = 1)
    avr_train_acc = np.mean(train_acc, axis = 1)

    #Draw acc, val score graph Start.
    figureHistory(foldNum, avr_val_acc,avr_val_loss,avr_train_acc,avr_train_loss, KERNEL_SEED, TRIAL)
    return 

if(foldNum<5):
    train_acc = io.loadmat('./graph/[%s%d%s%d]historyTrain.train_acc[%d].mat' %(SETT,TRIAL,AUG,KERNEL_SEED,foldNum))
    val_acc = io.loadmat('./graph/[%s%d%s%d]historyTrain.val_acc[%d].mat' %(SETT,TRIAL,AUG,KERNEL_SEED,foldNum))
    train_loss = io.loadmat('./graph/[%s%d%s%d]historyTrain.train_loss[%d].mat' %(SETT,TRIAL,AUG,KERNEL_SEED,foldNum))
    val_loss = io.loadmat('./graph/[%s%d%s%d]historyTrain.val_loss[%d].mat' %(SETT,TRIAL,AUG,KERNEL_SEED,foldNum))

    tacc = np.transpose(train_acc['mydata'])
    vacc = np.transpose(val_acc['mydata'])
    tloss = np.transpose(train_loss['mydata'])
    vloss = np.transpose(val_loss['mydata'])



if(foldNum < 5) :
    figureHistory(foldNum,vacc,vloss,tacc,tloss,KERNEL_SEED, TRIAL)
elif(foldNum == 5):
    if(not(MULTI_CHECK)):
        Yt = io.loadmat('./graph/[%s%d%s%d]Y_true.mat' %(SETT,TRIAL,AUG,KERNEL_SEED))
        yP = io.loadmat('./graph/[%s%d%s%d]yPrediction.mat' %(SETT,TRIAL,AUG,KERNEL_SEED))
        yLP = io.loadmat('./graph/[%s%d%s%d]yLabelPrediction.mat' %(SETT,TRIAL,AUG,KERNEL_SEED))

        Y_true = np.transpose(Yt['mydata'])
        yPrediction = np.transpose(yP['mydata'])
        yPrediction = np.reshape(yPrediction,(-1,))
        yLabelPrediction = np.transpose(yLP['mydata'])


        ROC_family(Y_true,yLabelPrediction,yPrediction,imgCountNO,imgCountYES,SETT,TRIAL,AUG,KERNEL_SEED,'_Multi')

    else :        
        # ========================================= Prob Sum ========================================= #
        Yt = io.loadmat('./graph/[%s%d%s%d]Y_true_P.mat' %(SETT,TRIAL,AUG,KERNEL_SEED))
        yP = io.loadmat('./graph/[%s%d%s%d]yPrediction_P.mat' %(SETT,TRIAL,AUG,KERNEL_SEED))
        yLP = io.loadmat('./graph/[%s%d%s%d]yLabelPrediction_P.mat' %(SETT,TRIAL,AUG,KERNEL_SEED))

        Y_true = np.transpose(Yt['mydata'])
        yPrediction = np.transpose(yP['mydata'])
        yPrediction = np.reshape(yPrediction,(-1,))
        yLabelPrediction = np.transpose(yLP['mydata'])
        ROC_family(Y_true,yLabelPrediction,yPrediction,imgCountNO,imgCountYES,SETT,TRIAL,AUG,KERNEL_SEED,'_ProbSum')

        

if(foldNum<5 or foldNum == 0):
    train_acc_file = './graph/[%s%d%s%d]historyTrain.train_acc[%d].mat' %(SETT,TRIAL,AUG,KERNEL_SEED,foldNum)
    val_acc_file = './graph/[%s%d%s%d]historyTrain.val_acc[%d].mat' %(SETT,TRIAL,AUG,KERNEL_SEED,foldNum)
    train_loss_file = './graph/[%s%d%s%d]historyTrain.train_loss[%d].mat' %(SETT,TRIAL,AUG,KERNEL_SEED,foldNum)
    val_loss_file = './graph/[%s%d%s%d]historyTrain.val_loss[%d].mat' %(SETT,TRIAL,AUG,KERNEL_SEED,foldNum)
    if os.path.isfile(train_acc_file):
        os.remove(train_acc_file)
    if os.path.isfile(val_acc_file):
        os.remove(val_acc_file)
    if os.path.isfile(train_loss_file):
        os.remove(train_loss_file)
    if os.path.isfile(val_loss_file):
        os.remove(val_loss_file)