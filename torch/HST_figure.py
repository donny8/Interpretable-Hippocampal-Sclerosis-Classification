from HST_common import *
import matplotlib.pyplot as plt
from Args.argument import get_args
args=get_args()


def acclossGraph(train_loss,train_acc,test_loss,test_acc,Report,fold,dir_path):
    
    # For saving .mat files to draw a graph
    if(type(train_loss)==torch.Tensor):
        temp_array1 = train_loss.numpy() ; temp_array2 = train_acc.numpy()
        temp_array3 = test_loss.numpy() ; temp_array4 = test_acc.numpy()
    else:
        temp_array1 = train_loss ; temp_array2 = train_acc
        temp_array3 = test_loss ; temp_array4 = test_acc
    
    if(fold<5):
        scipy.io.savemat(dir_path+'/[%s%d%s]train_loss_%s_D%d{F%dK%d}[%d].mat' %(SETT,TRIAL,AUG,CONTROLTYPE,DATATYPE,FOLD_SEED,KERNEL_SEED,fold),{"mydata": temp_array1})
        scipy.io.savemat(dir_path+'/[%s%d%s]train_acc_%s_D%d{F%dK%d}[%d].mat' %(SETT,TRIAL,AUG,CONTROLTYPE,DATATYPE,FOLD_SEED,KERNEL_SEED,fold),{"mydata": temp_array2})
        scipy.io.savemat(dir_path+'/[%s%d%s]val_loss_%s_D%d{F%dK%d}[%d].mat' %(SETT,TRIAL,AUG,CONTROLTYPE,DATATYPE,FOLD_SEED,KERNEL_SEED,fold),{"mydata": temp_array3})
        scipy.io.savemat(dir_path+'/[%s%d%s]val_acc_%s_D%d{F%dK%d}[%d].mat' %(SETT,TRIAL,AUG,CONTROLTYPE,DATATYPE,FOLD_SEED,KERNEL_SEED,fold),{"mydata": temp_array4})
    elif(fold==5):
        scipy.io.savemat(dir_path+'/[%s%d%s]train_loss_%s_D%d{F%dK%d}.mat' %(SETT,TRIAL,AUG,CONTROLTYPE,DATATYPE,FOLD_SEED,KERNEL_SEED),{"mydata": temp_array1})
        scipy.io.savemat(dir_path+'/[%s%d%s]train_acc_%s_D%d{F%dK%d}.mat' %(SETT,TRIAL,AUG,CONTROLTYPE,DATATYPE,FOLD_SEED,KERNEL_SEED),{"mydata": temp_array2})
        scipy.io.savemat(dir_path+'/[%s%d%s]val_loss_%s_D%d{F%dK%d}.mat' %(SETT,TRIAL,AUG,CONTROLTYPE,DATATYPE,FOLD_SEED,KERNEL_SEED),{"mydata": temp_array3})
        scipy.io.savemat(dir_path+'/[%s%d%s]val_acc_%s_D%d{F%dK%d}.mat' %(SETT,TRIAL,AUG,CONTROLTYPE,DATATYPE,FOLD_SEED,KERNEL_SEED),{"mydata": temp_array4})

    temp_array1 = str(temp_array1)
    temp_array2 = str(temp_array2)
    temp_array3 = str(temp_array3)
    temp_array4 = str(temp_array4)

    Report.write('\n\n train_loss :')
    Report.write(temp_array1)
    Report.write('\n\n train_acc :')
    Report.write(temp_array2)
    Report.write('\n\n test_loss :')
    Report.write(temp_array3)
    Report.write('\n\n test_acc :')
    Report.write(temp_array4)

    del temp_array1
    del temp_array2
    del temp_array3
    del temp_array4


def figureHistory(val_acc,val_loss,train_acc,train_loss):
    fig = plt.figure()
    ax_acc = fig.add_subplot(111)

    # ax_acc.plot(range(nb_epochs), val_acc, label='acc(%)', color='darkred')
    ax_acc.plot(range(nb_epochs), val_acc, label='acc(%)', color='darkred')
    plt.xlabel('epochs')
    plt.ylabel('acc(%)')
    ax_acc.grid(linestyle='--', color='lavender')
    plt.yticks(np.arange(0.4,1.1,0.1))
    plt.rc('xtick', labelsize=10)    # fontsize of the tick labels

    # val_loss maximum value limit
    for index, value in enumerate(val_loss):
        if value > 2.6:
            val_loss[index] = 2.6
    
    ax_loss = ax_acc.twinx()
    # ax_loss.plot(range(nb_epochs), val_loss, label='loss', color='darkblue')
    ax_loss.plot(range(nb_epochs), val_loss, label='loss', color='darkblue')
    plt.ylabel('loss')
    ax_loss.yaxis.tick_right()
    plt.yticks(np.arange(0.2,2.8,0.2))
    # ax_loss.grid(linestyle='--', color='lavender')

    plt.legend()
    saveFileName = './graph/[%s]%s_test(lr_%s_epoch_%d).png' %(dataset,model,lrs,nb_epochs)  
    plt.savefig(saveFileName)
    # plt.show()
    ax_acc.legend()

    fig1 = plt.figure()
    ax_acc1 = fig1.add_subplot(111)

    # ax_acc1.plot(range(nb_epochs), train_acc, label='acc(%)', color='darkred')
    ax_acc1.plot(range(nb_epochs), train_acc, label='acc(%)', color='darkred')
    plt.xlabel('epochs')
    plt.ylabel('acc(%)')
    ax_acc1.grid(linestyle='--', color='lavender')
    plt.yticks(np.arange(0.4,1.1,0.1))
    plt.rc('xtick', labelsize=10)    # fontsize of the tick labels

    ax_loss = ax_acc1.twinx()
    # ax_loss.plot(range(nb_epochs), train_loss, label='loss', color='darkblue')
    ax_loss.plot(range(nb_epochs), train_loss, label='loss', color='darkblue')
    plt.ylabel('loss')
    ax_loss.yaxis.tick_right()
    ax_loss.grid(linestyle='--', color='lavender')
    # ticks every 0.1

    plt.legend()
    ax_acc.legend()
    saveFileName = './graph/[%s]%s_train(lr_%s_epoch_%d).png' %(dataset,model,lrs,nb_epochs)          
    plt.savefig(saveFileName)
    # plt.savefig('./graph/hs_train1.png')
    # plt.show()
