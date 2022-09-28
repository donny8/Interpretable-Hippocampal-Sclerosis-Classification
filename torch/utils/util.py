from .common import *
from .model import *
from .figure import acclossGraph
import inspect
import shutil
import requests

def mkdir(path):
    if not(os.path.isdir(path)):
        os.mkdir(path)

def log_intro():
    # log file
    mkdir('./log')
    Report = './log/[%s%d%s]aLog_%s[%d]{F%dK%d}.txt' %(SETT,TRIAL,AUG,CONTROLTYPE,DATATYPE,FOLD_SEED,KERNEL_SEED)
    Summary = './log/[%s%d%s]Result_Reports.txt' %(SETT,TRIAL, AUG)
    Rep = open(Report,'a') ; Sum = open(Summary,'a')
    temp_log = '\n\n'+str(args)[9:]
    if(SETT=='FUL'):
        temp_log = temp_log + '  FULEPOCH %d\n' %(FULEPOCH)
    else:
        temp_log = temp_log + '\n'
    printer(temp_log,Rep,Sum)
    return Rep, Sum

def log_close(Rep, Sum, start):
    printer(timer(start),Rep,Sum)
    Rep.close()
    Sum.close()

def printer(temp_log,report,summary):
    print(temp_log)
    report.write(temp_log)
    summary.write(temp_log)    

def seed_set(rdmsd):
    random.seed(rdmsd)
    np.random.seed(rdmsd)
    torch.manual_seed(rdmsd)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed(rdmsd)
    else:
        print('[CUDA unavailable]'); 
        sys.exit()

def source_load(categories,dirDataSet):
    inputX, inputY, listFileName, niiInfo = data_single(categories,dirDataSet)    
    Y_vector = []
    if(SETT=='SIG'):
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
    return inputX, inputY, Y_vector, listFileName, niiInfo


def data_single(categories, dirDataSet):
    inputX = []; inputY = []; listFileName = []
    for idx, f in enumerate(categories):        
        label = idx
        image_dir = dirDataSet + "/" + f
        print(image_dir)
        for (imagePath, dir, files) in sorted(os.walk(image_dir)):
            ## read 3D nii file ###
            if(imagePath == image_dir):                
                print("%s" % (imagePath))                
                # image reading #
                files = sorted(glob.glob(imagePath + "/*.nii.gz"))
                listFileName += (files)
    
            for i, fname in enumerate(files):
                print(fname) #print(i)
                if(i==0):
                    niiInfo = nib.load(fname)
                img = nib.load(fname).get_fdata()
                inputX.append(img)
                inputY.append(label)
    inputX = torch.tensor(inputX)    
    inputY = torch.tensor(inputY)
    listFileName = np.array(listFileName,dtype=object)
    return inputX, inputY, listFileName, niiInfo



def CV_train(inputX,inputY,Y_vector,Rep,Sum):
    seed_set(KERNEL_SEED)
    mkdir('./saveModel')
    avr_trn_acc = [] ; avr_trn_loss = [] ; avr_val_acc = [] ; avr_val_loss = []
    for fold, (train_index, val_index) in enumerate(KFOLD.split(inputX,Y_vector)):
        temp_log = '\nfold %d train_index : %s' %(fold,train_index)
        Rep.write(temp_log)
        temp_log = '\nfold %d validation_index : %s' %(fold,val_index)
        Rep.write(temp_log)
        # Fold Initialize
        trn_loss = torch.zeros(ENDEPOCH) ; val_loss= torch.zeros(ENDEPOCH) ; trn_acc = torch.zeros(ENDEPOCH) ; val_acc = torch.zeros(ENDEPOCH)
        # Model Initialize

        print('\n\n==> [ Fold %d ] Building model..'%(fold+1))
        if(CONTROLTYPE=='CLRM'): net = HSCNN(ksize)
        net.to(device)
        if (device == 'cuda') and (ksize==4): net = torch.nn.DataParallel(net)

#        if(fold==0): summary(net, input_size=(1,160,200,170))
#        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size, gamma = lreduce , last_epoch=-1)
#        optimizer=optim.Adam(params=net.parameters(),lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=decay_rate)
        if(OPT=='SGD'): optimizer = optim.SGD(net.parameters(), lr=lr,momentum=MOM, weight_decay=wdecay)
        elif(OPT == 'Adam'): optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=wdecay)

        step_size = ENDEPOCH//lrperiod
        if(SCH=='CALR'):
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=step_size)
        elif(SCH=='SLR'):
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=lreduce)
        elif(SCH=='RLRP'):
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=step_size, factor=lreduce)

        train_loader, val_loader = CV_data_load(inputX,inputY,train_index,val_index,AUG,True)

        for epoch in range(ENDEPOCH):
            for param_group in optimizer.param_groups:
                print("\nCurrent learning rate is: {}".format(param_group['lr']))
            print('Epoch {} / {}   Fold {} / {}'.format(epoch + 1, ENDEPOCH, fold + 1 , KFOLD.get_n_splits()))
            train(net,epoch,optimizer,criterion,trn_acc,trn_loss,train_loader)
            vld_loss = validation(net,epoch,fold,criterion,val_acc,val_loss,val_loader)
            realtime_graph(SETT,TRIAL,AUG,KERNEL_SEED,fold,epoch,trn_acc,trn_loss,val_acc,val_loss,os.getcwd()+graph_path)
            if(SCH=='RLRP'): scheduler.step(vld_loss)
            else: scheduler.step()
        acclossGraph(trn_loss,trn_acc,val_loss,val_acc,Rep,fold,os.getcwd()+graph_path)    
        avr_trn_acc.append(trn_acc) ; avr_val_acc.append(val_acc) ; avr_trn_loss.append(trn_loss) ; avr_val_loss.append(val_loss)

        del net, train_loader, val_loader
    argmax_acc = avr_calc(avr_trn_acc,avr_trn_loss,avr_val_acc,avr_val_loss,Rep,Sum,os.getcwd()+graph_path)

    for cnt in range(nb_KFold):
        for ep in range(ENDEPOCH):
            saveModelName = os.getcwd()+'/saveModel/[%s%d%s]HS%s_D%d{F%dK%d}[%d](%d).pt'%(SETT,TRIAL,AUG,CONTROLTYPE,DATATYPE,FOLD_SEED,KERNEL_SEED,cnt,ep)
            changeModelName = os.getcwd()+'/saveModel/[%s%d%s]HS%s_D%d{F%dK%d}[%d](best).pt'%(SETT,TRIAL,AUG,CONTROLTYPE,DATATYPE,FOLD_SEED,KERNEL_SEED,cnt)
            if os.path.isfile(saveModelName):
                if(ep == argmax_acc):
                    os.rename(saveModelName, changeModelName)
                else:
                    os.remove(saveModelName)


def CV_data_load(inputX,inputY,train_index,val_index,AUG,switch): # switch : train/eval
    ### Dividing data into folds
    x_train = inputX[train_index]
    x_val = inputX[val_index]
    y_train = inputY[train_index]
    y_val = inputY[val_index]
    if(AUG=='hflip'):    
        temp_data = torch.flip(x_train,[1,]) # 0:batch  1:horizontal  2:forward&backward   3:vertical
        temp_label =  torch.zeros(len(y_train))

        for idx in range(len(y_train)):
            if(y_train[idx] == 1) : temp_label[idx]= 2
            elif(y_train[idx] == 2) : temp_label[idx]= 1 
            else : temp_label[idx]= y_train[idx]

        temp_label = temp_label.long()
        x_train = torch.cat([x_train,temp_data],dim=0)
        y_train = torch.cat([y_train,temp_label],dim=0)

    train_data = torch.utils.data.TensorDataset(x_train, y_train)
    val_data = torch.utils.data.TensorDataset(x_val, y_val)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size =BATCH, shuffle = True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size = BATCH, shuffle = switch)
    return train_loader, val_loader


def train(net, epoch,optimizer,criterion,trn_acc,trn_loss,train_loader):
    net.train()
    train_acc = 0 ; train_loss = 0 ; correct = 0 ; total = 0
    for batch_index, (images, labels) in enumerate(train_loader):
        images = images.view(-1,1,imgRow,imgCol,imgDepth)
        images = images.float()
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        output = net(images)
        loss = criterion(output, labels)        
        
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, pred = output.max(1)
        correct += pred.eq(labels).sum().item()
        total += labels.size(0)
        progress_bar(batch_index,len(train_loader), 'Loss: %.3f | Acc: %.3f%% ( %d / %d )' % (train_loss/(batch_index+1), 100.*correct/total,correct, total))
        del images
        del labels
    trn_loss[epoch] = train_loss/(batch_index+1)
    trn_acc[epoch] = 100.*correct/total

    
    
def validation(net,epoch,fold,criterion,val_acc,val_loss,val_loader):
    global best_acc
    net.eval()
    vld_acc = 0 ; vld_loss = 0 ; correct = 0 ; total = 0
    if(epoch==0): best_acc=0
    with torch.no_grad():
        for batch_index, (images, labels) in enumerate(val_loader):

            images = images.view(-1,1,imgRow,imgCol,imgDepth)
            images = images.float()
            images = images.to(device)
            labels = labels.to(device)
            output = net(images)
            loss = criterion(output,labels)            
            
            vld_loss += loss.item()
            _, pred = output.max(1)
            correct += pred.eq(labels).sum().item()
            total += labels.size(0)
            progress_bar(batch_index,len(val_loader), 'Loss: %.3f | Acc: %.3f%% ( %d / %d )' % (vld_loss/(batch_index+1), 100.*correct/total,correct, total))
            del images, labels
            
    # Save checkpoint.
    acc = correct/total
    if(acc>best_acc):
        best_acc = acc
        print(f'New Best Accuracy : {best_acc:.3f} at epoch {epoch}')
    savePath = './saveModel/[%s%d%s]HS%s_D%d{F%dK%d}[%d](%d).pt'%(SETT,TRIAL,AUG,CONTROLTYPE,DATATYPE,FOLD_SEED,KERNEL_SEED,fold,epoch)
    torch.save(net.state_dict(), savePath)

    val_loss[epoch] = vld_loss/(batch_index+1)
    val_acc[epoch] = 100.*correct/total
    return vld_loss/(batch_index+1)
    

    
TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time    
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    _, term_width = shutil.get_terminal_size()
    term_width = int(term_width)
    if current == 0:
        begin_time = time.time()  # Reset for new bar.
    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1
    
    proc = []
    proc.append(' [')
    eq = '='*cur_len
    proc.append(eq)
    proc.append('>')
    re = '.'*rest_len
    proc.append(re)
    proc.append(']')
    
    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    proc.append('  Step: %s' % format_time(step_time))
    proc.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        proc.append(' | ' + msg)
    msg = ''.join(proc)
    sys.stdout.write(msg)

    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()    
    
    
    
def format_time(seconds):
    days = int(seconds / 3600/24) ; seconds = seconds - days*3600*24 ; hours = int(seconds / 3600)
    seconds = seconds - hours*3600 ; minutes = int(seconds / 60) ; seconds = seconds - minutes*60
    secondsf = int(seconds) ; seconds = seconds - secondsf ; millis = int(seconds*1000)
    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f
        
    
def realtime_graph(SETT,TRIAL,AUG,KERNEL_SEED,fold,epoch,trn_acc,trn_loss,val_acc,val_loss,graph_path):
    Graph_Report = graph_path+'/[%s%d%s%d]Graph_Reports[F%d].txt' %(SETT,TRIAL, AUG,KERNEL_SEED,fold)
    GR = open(Graph_Report,'a+')
    epoch_result = '%f, %f, %f, %f\n' %(trn_acc[epoch], trn_loss[epoch], val_acc[epoch], val_loss[epoch])
    GR.write(epoch_result) 
    GR.close()
    

def avr_calc(avr_trn_acc,avr_trn_loss,avr_val_acc,avr_val_loss,Rep,Sum,graph_path):
    mean_trn_loss = avr2mean(avr_trn_loss)
    mean_val_loss = avr2mean(avr_val_loss)
    mean_trn_acc = avr2mean(avr_trn_acc)
    mean_val_acc = avr2mean(avr_val_acc)

    argmax_acc = torch.argmax(mean_val_acc)
    Sum.write('\nBest epoch : %d' %(argmax_acc))
    acclossGraph(mean_trn_loss,mean_trn_acc,mean_val_loss,mean_val_acc,Rep,5,graph_path)    
    return argmax_acc

def avr2mean(tensor):
    tensor = torch.cat(tensor,dim=-1)
    tensor = tensor.view(nb_KFold,-1)
    mean = torch.mean(tensor,dim=0)
    return mean

def CV_eval(inputX,inputY,Y_vector,Rep,Sum) :
    
    yIdxPrediction = []
    multi = [] ; binary = []; yPredictionB = []; yLabelPredictionB=[]; 
    for fold, (train_index, val_index) in enumerate(KFOLD.split(inputX,Y_vector)):
        model_path = os.getcwd()+'/saveModel/[%s%d%s]HS%s_D%d{F%dK%d}[%d](best).pt'%(SETT,TRIAL,AUG,CONTROLTYPE,DATATYPE,FOLD_SEED,KERNEL_SEED,fold)
        
        if(CONTROLTYPE=='CLRM'): net = HSCNN(ksize)
        if (device == 'cuda') and (ksize==4): net = torch.nn.DataParallel(net)
        net.load_state_dict(torch.load(model_path))
        net.to(device)

        train_loader, val_loader = CV_data_load(inputX,inputY,train_index,val_index,AUG,False)
        yLP, yPB, yLPB, multi_acc, binary_acc = evaluation(net,val_loader)
        multi.append(multi_acc)
        binary.append(binary_acc)

        if(fold==0):
            yIdxPrediction = val_index
            yLabelPrediction = yLP
            yPredictionB = yPB
            yLabelPredictionB = yLPB
        else:
            yIdxPrediction=np.concatenate([yIdxPrediction, val_index])
            yLabelPrediction = torch.cat([yLabelPrediction, yLP])
            yPredictionB = torch.cat([yPredictionB, yPB])
            yLabelPredictionB = torch.cat([yLabelPredictionB ,yLPB])

        del net, train_loader, val_loader
    acc_roc(multi,binary,yIdxPrediction,yLabelPrediction,yPredictionB,yLabelPredictionB,Rep,Sum,iters,graph_path,imgCountNO_4,imgCountNO_7,imgCountNO_0,imgCountYES_1,imgCountYES_2)


def evaluation(net,eval_loader):
    net.eval()
    evl_acc = 0 ; correct = 0 ; total = 0 ; correctB = 0 ; totalB = 0
    with torch.no_grad():
        for batch_index, (images, labels) in enumerate(eval_loader):
            images = images.view(-1,1,imgRow,imgCol,imgDepth)
            images = images.float()
            images = images.to(device)
            labels = labels.to(device)
            output = F.softmax(net(images),dim=1)  

            labelsB = labels.detach().clone()
            labelsB[labelsB!=0] = 1
            labelsB = labelsB.to(device)
                                    
            # Multi-to-Binary
            out_yes = output[:,1:3].sum(dim=1)
            out_no = output[:,0]
            outputB = torch.stack([out_no,out_yes],dim=1)

            yPredictionB = outputB[:,1]
            
            _, pred = output.max(1) #yLabelPrediction
            correct += pred.eq(labels).sum().item()
            total += labels.size(0)
                        
            _, predB = outputB.max(1) #yLabelPredictionB
            correctB += predB.eq(labelsB).sum().item()
            totalB += labelsB.size(0)
            
            if(batch_index==0):
                yLP = pred.detach().clone()
                yLPB = predB.detach().clone()
                yPB = yPredictionB.detach().clone()
            else :
                yLP = torch.cat([yLP,pred])
                yLPB = torch.cat([yLPB,predB])
                yPB = torch.cat([yPB,yPredictionB])

            del images, labels, output, out_yes, out_no, outputB
            
    multi_acc = 100.*correct/total
    binary_acc = 100.*correctB/totalB
    return yLP, yPB, yLPB, multi_acc, binary_acc

def acc_roc(multi,binary,yIdxPrediction,yLabelPrediction,yPredictionB,yLabelPredictionB,Rep,Sum,iters,graph_path, NO4, NO7, NO0, YES1, YES2):    

    NO = NO4 + NO7 + NO0
    YES = YES1 + YES2
    TOTAL = NO + YES
    
    # True Labels for Multi- and Binary Classification
    Y_true = [] ; Y_trueB = []
    for i in range(NO):
        Y_true.append(0)
        Y_trueB.append(0)
    for i in range(YES1):
        Y_true.append(1)
        Y_trueB.append(1)
    for i in range(YES2):
        Y_true.append(2)
        Y_trueB.append(1)
    Y_true = torch.tensor(Y_true) ; Y_trueB = torch.tensor(Y_trueB)

    tempMatrix1 = [0]*TOTAL ; tempMatrix2 = [0]*TOTAL ; tempMatrix3 = [0]*TOTAL
    
    for idx in range(TOTAL):
        tempMatrix1[yIdxPrediction[idx]] = yLabelPrediction[idx]  
        tempMatrix2[yIdxPrediction[idx]] = yPredictionB[idx]  
        tempMatrix3[yIdxPrediction[idx]] = yLabelPredictionB[idx]
    tempMatrix1 = torch.tensor(tempMatrix1) ; tempMatrix2 = torch.tensor(tempMatrix2) ; tempMatrix3 = torch.tensor(tempMatrix3)
    yLabelPrediction = tempMatrix1 ; yPredictionB = tempMatrix2 ; yLabelPredictionB = tempMatrix3
    
    multi_class_confusion = confusion_matrix(Y_true, yLabelPrediction)
    tntp = np.diag(multi_class_confusion)
    correct = np.sum(tntp)
    printer('\n\n [Multi Class]\n',Rep,Sum)
    printer(str(tntp.ravel()),Rep,Sum)
    printer('\nAccuracy! %.3f' %(correct/len(Y_true)),Rep,Sum)
    printer('\nFolds:{}'.format(multi),Rep,Sum)


    printer('\n\n [Binary Class]',Rep,Sum)
    tn, fp, fn, tp = confusion_matrix(Y_trueB, yLabelPredictionB).ravel()
    printer('\ntn %d, fp %d, fn %d, tp %d'%(tn, fp, fn, tp),Rep,Sum)
    printer('\nAccuracy! %.3f' %((tn+tp)/(tn+tp+fn+fp)),Rep,Sum)
    printer('\nFolds:{}'.format(binary),Rep,Sum)
    printer('\n\nSensitivity : %0.3f' %(tp/(fn+tp)),Rep,Sum)
    printer('\nSpecificity : %0.3f' %(tn/(tn+fp)),Rep,Sum)
    printer('\nF1_score : %.3f' %(f1_score(Y_trueB, yLabelPredictionB)),Rep,Sum)
    printer('\nAUROC : %.3f' %roc_auc_score(Y_trueB, yPredictionB),Rep,Sum)
    


    if(type(Y_true)==torch.Tensor):
        Y_true = Y_true.numpy()
        yLabelPrediction = yLabelPrediction.numpy()
        Y_trueB = Y_trueB.numpy()
        yPredictionB = yPredictionB.numpy()
        yLabelPredictionB = yLabelPredictionB.numpy()

    scipy.io.savemat(os.getcwd()+graph_path+'/[%s%d%s%d]Y_true_M.mat' %(SETT,TRIAL,AUG,KERNEL_SEED),{"mydata": Y_true})
    scipy.io.savemat(os.getcwd()+graph_path+'/[%s%d%s%d]yLabelPrediction_M.mat' %(SETT,TRIAL,AUG,KERNEL_SEED),{"mydata": yLabelPrediction})

    scipy.io.savemat(os.getcwd()+graph_path+'/[%s%d%s%d]Y_true_B.mat' %(SETT,TRIAL,AUG,KERNEL_SEED),{"mydata": Y_trueB})
    scipy.io.savemat(os.getcwd()+graph_path+'/[%s%d%s%d]yPrediction_B.mat' %(SETT,TRIAL,AUG,KERNEL_SEED),{"mydata": yPredictionB})
    scipy.io.savemat(os.getcwd()+graph_path+'/[%s%d%s%d]yLabelPrediction_B.mat' %(SETT,TRIAL,AUG,KERNEL_SEED),{"mydata": yLabelPredictionB})

    if(SETT=='FUL'):
        num4=10 ; num7=20 ; num0=20
    elif(SETT=='SIG'):
        num4=24 ; num7=48 ; num0=48        
#    balance(CONTROLTYPE,SETT,TRIAL,AUG,KERNEL_SEED,iters, NO4, NO7,NO0,YES1,YES2,num4,num7,num0,Sum)
    
def calbalcd(true,label,num,lst):
    check = len(true[true==num])
    if(check):
        where = np.where(true==num)
        corresd = label[where]
        cal = len(corresd[corresd==0]) / len(true[true==num])
        lst.append(cal)
        return lst
    else:
        return lst

    
def balance(CONTROLTYPE,SETT,TRIAL,AUG,KERNEL_SEED,iters, C4,C7,C0,C1,C2,num4,num7,num0,RR):
    np.random.seed(KERNEL_SEED)
    Y_true_M,                yLabelPrediction_M = loader(SETT,TRIAL,AUG, KERNEL_SEED,"M")
    Y_true_B, yPrediction_B, yLabelPrediction_B = loader(SETT,TRIAL,AUG, KERNEL_SEED,"B")
    yPrediction_M = []
    
    idx4 = np.arange(C4)
    idx7 = np.arange(C7) + C4
    idx0 = np.arange(C0) + C4 + C7
    idx1 = np.arange(C1) + C4 + C7 + C0
    idx2 = np.arange(C2) + C4 + C7 + C0 + C1
    right_len = len(idx2)
    
    multi_set = []  ;  multi_perf = []  ;  binary_set = [] ;  binary_perf = []
    tnn_set = [] ; fpn_set = [] ; tpl_set = [] ; fnl_set = [] ; tpr_set = [] ; fnr_set = []
    sens = [] ; spec = [] ; f1_scores = []
    
    M0s = []  ;  M4s = [] ; M7s = []
    B0s = []  ;  B4s = [] ; B7s = []
    Y_true_M[:C4] = 4
    Y_true_M[C4:C4+C7] = 7
    Y_true_B[:C4] = 4
    Y_true_B[C4:C4+C7] = 7

    for i in range(iters):
        M_true, M_label         = random_sample(idx4,idx7,idx0,idx1,idx2,num4,num7,num0, Y_true_M, yPrediction_M, yLabelPrediction_M,'M' )
        B_true, B_label, B_pred = random_sample(idx4,idx7,idx0,idx1,idx2,num4,num7,num0, Y_true_B, yPrediction_B, yLabelPrediction_B,'B' )
        
        M0s = calbalcd(M_true,M_label,0,M0s)
        M4s = calbalcd(M_true,M_label,4,M4s)
        M7s = calbalcd(M_true,M_label,7,M7s)
        B0s = calbalcd(B_true,B_label,0,B0s)
        B4s = calbalcd(B_true,B_label,4,B4s)
        B7s = calbalcd(B_true,B_label,7,B7s)
        M_true[M_true==4] = 0 ; M_true[M_true==7] = 0
        B_true[B_true==4] = 0 ; B_true[B_true==7] = 0
        
        multi_class_confusion = confusion_matrix(M_true, M_label)
        tntp = np.diag(multi_class_confusion)
        correct = np.sum(tntp)
        acc = correct/len(M_true)
        multi_set.append([tntp[0],tntp[1],tntp[2]]) ; multi_perf.append(acc)

        tn, fp, fn, tp = confusion_matrix(B_true, B_label).ravel()
        acc = (tn+tp)/(tn+fp+fn+tp)
        auc = roc_auc_score(B_true, B_pred)
        f1 = f1_score(B_true, B_label)
        binary_set.append([tn,fp,fn,tp]) ; binary_perf.append([acc,auc])
        sens.append(tp/(tp+fn)) ; spec.append(tn/(tn+fp)) ; f1_scores.append(f1)

        num_yes = 2 * right_len
        BN = B_label[:num_yes]
        BL = B_label[num_yes:num_yes+right_len]
        BR = B_label[num_yes+right_len:]

        tnn = len(BN[BN==0]) ;  fpn = len(BN[BN==1])
        tpl = len(BL[BL==1]) ;  fnl = len(BL[BL==0])
        tpr = len(BR[BR==1]) ;  fnr = len(BR[BR==0])
    
        tnn_set.append(tnn) ; fpn_set.append(fpn)
        tpl_set.append(tpl) ; fnl_set.append(fnl)
        tpr_set.append(tpr) ; fnr_set.append(fnr)

    RR.write('\n\n ======== [BALANCE] ========')
    RR.write('\n\n[Multi Control]\nZero: %.2f'%(np.average(M0s)))
    RR.write('\nFour: %.3f'%(np.average(M4s)))
    RR.write('\nSeven: %.3f\n'%(np.average(M7s)))
    RR.write('\n[Binary Control]\nZero: %.2f'%(np.average(B0s)))
    RR.write('\nFour: %.3f'%(np.average(B4s)))
    RR.write('\nSeven: %.3f\n'%(np.average(B7s)))

    balance_record(sens,spec,f1_scores,multi_set,multi_perf,binary_set,binary_perf,tnn_set, fpn_set,tpl_set,fnl_set,tpr_set,fnr_set,RR)
    

def loader(SETT,TRIAL,AUG,KERNEL_SEED,LRMB):
    yT = io.loadmat(os.getcwd()+graph_path+'/[%s%d%s%d]Y_true_%s.mat' %(SETT,TRIAL,AUG,KERNEL_SEED,LRMB))
    yL = io.loadmat(os.getcwd()+graph_path+'/[%s%d%s%d]yLabelPrediction_%s.mat' %(SETT,TRIAL,AUG,KERNEL_SEED,LRMB))
    Y_true = np.transpose(yT['mydata'])  ;  yLabelPrediction = np.transpose(yL['mydata'])
    if(LRMB!='M'):
        yP = io.loadmat(os.getcwd()+graph_path+'/[%s%d%s%d]yPrediction_%s.mat' %(SETT,TRIAL,AUG,KERNEL_SEED,LRMB))
        yPrediction = np.transpose(yP['mydata'])
        return Y_true, yPrediction, yLabelPrediction    
    else: 
        return Y_true, yLabelPrediction
    
    
def random_sample(idx4,idx7,idx0,idx1,idx2,num4,num7,num0,Y_true,yPrediction,yLabelPrediction,expr):
    
    if(expr=='M'):

        idxM7 = np.random.choice(idx7, num7//2, replace=False)
        idxM0 = np.random.choice(idx0, num0//2, replace=False)
        idxM1 = np.random.choice(idx1, len(idx2), replace=False)
        idxM2 = idx2
        
        true_NO0 = Y_true[idxM0]  ; true_NO7 = Y_true[idxM7]
        true_YES1 = Y_true[idxM1] ; true_YES2 = Y_true[idxM2]        
        label_NO0 = yLabelPrediction[idxM0]  ; label_NO7 = yLabelPrediction[idxM7]
        label_YES1 = yLabelPrediction[idxM1] ; label_YES2 = yLabelPrediction[idxM2]

        if(num4!=0):
            idxM4 = np.random.choice(idx4, num4//2, replace=False)
            true_NO4 = Y_true[idxM4] ; label_NO4 = yLabelPrediction[idxM4]
            true = np.vstack([true_NO4,true_NO7,true_NO0,true_YES1, true_YES2])
            label = np.vstack([label_NO4,label_NO7,label_NO0, label_YES1, label_YES2])
        else:
            true = np.vstack([true_NO7,true_NO0,true_YES1, true_YES2])
            label = np.vstack([label_NO7,label_NO0, label_YES1, label_YES2])

        return true, label
    
    elif(expr=='B'):
            
        idxB7 = np.random.choice(idx7, num7, replace=False)
        idxB0 = np.random.choice(idx0, num0, replace=False)
        idxB1 = np.random.choice(idx1,len(idx2),replace=False)
        idxBY = np.hstack([idxB1,idx2])

        true_NO0 = Y_true[idxB0] ; true_NO7 = Y_true[idxB7] ; true_YES = Y_true[idxBY]
        label_NO0 = yLabelPrediction[idxB0] ; label_NO7 = yLabelPrediction[idxB7] ; label_YES1 = yLabelPrediction[idxBY]
        pred_NO0 = yPrediction[idxB0] ; pred_NO7 = yPrediction[idxB7] ; pred_YES = yPrediction[idxBY]

        if(num4!=0):
            idxB4 = np.random.choice(idx4, num4, replace=False)
            true_NO4 = Y_true[idxB4] ; label_NO4 = yLabelPrediction[idxB4] ; pred_NO4 = yPrediction[idxB4]
            
            true = np.vstack([true_NO4,true_NO7,true_NO0, true_YES])
            label = np.vstack([label_NO4,label_NO7,label_NO0, label_YES1])
            pred = np.vstack([pred_NO4,pred_NO7,pred_NO0,pred_YES])
        else:
            true = np.vstack([true_NO7,true_NO0, true_YES])
            label = np.vstack([label_NO7,label_NO0, label_YES1])
            pred = np.vstack([pred_NO7,pred_NO0,pred_YES])
        
        return true, label, pred


def balance_record(sens,spec,f1_scores,multi_set,multi_perf,binary_set,binary_perf,tnn_set, fpn_set,tpl_set,fnl_set,tpr_set,fnr_set,RR):
        
    sens = np.array(sens)
    spec = np.array(spec)
    f1_scores = np.array(f1_scores)

    multi_set = np.round(np.average(multi_set,axis=0),1)
    multi_perf = np.round(np.average(multi_perf,axis=0),3)
    binary_set = np.round(np.average(binary_set,axis=0),0)
    binary_perf = np.round(np.average(binary_perf,axis=0),3)

    tnn = np.round(np.average(tnn_set,axis=0),1) ; fpn = np.round(np.average(fpn_set,axis=0),1)
    tpl = np.round(np.average(tpl_set,axis=0),1) ; fnl = np.round(np.average(fnl_set,axis=0),1)
    tpr = np.round(np.average(tpr_set,axis=0),1) ; fnr = np.round(np.average(fnr_set,axis=0),1)
    
    test = (binary_set[0] + binary_set[3]) / np.sum(binary_set)
    test = np.round(test,3)

    RR.write('\n\n[MULTI]')
    RR.write('\nNo Left Right\n')
    RR.write(str(multi_set))
    RR.write('\nAccuracy : ')
    RR.write(str(multi_perf))

    RR.write('\n\n[BINARY]')
    RR.write('\ntn fp fn tp\n')
    RR.write(str(binary_set))
    RR.write('\nAccuracy : ')
    RR.write(str(test))
    RR.write('\nSensitivity : %0.3f'%(np.average(sens)))
    RR.write('\nSpecificity : %0.3f'%(np.average(spec)))
    RR.write('\nF1-score : %0.3f'%(np.average(f1_scores)))
    RR.write('\nAUROC :')
    RR.write(str(binary_perf[1]))

    RR.write('\n\nTrue Negative NO : %s' %(str(tnn)))
    RR.write('\nFalse Positive NO : %s' %(str(fpn)))
    RR.write('\nNO : %s' %(str(np.round(tnn/(tnn+fpn),3))))

    RR.write('\n\nTrue Positive Left : %s' %(str(tpl)))
    RR.write('\nFalse Negative Left : %s' %(str(fnl)))
    RR.write('\nLeft : %s' %(str(np.round(tpl/(tpl+fnl),3))))

    RR.write('\n\nTrue Positive Right : %s' %(str(tpr)))
    RR.write('\nFalse Negative Right : %s' %(str(fnr)))
    RR.write('\nRight : %s' %(str(np.round(tpr/(tpr+fnr),3))))

def timer(start):
    #Time Calculation
    finish = time.time() - start
    hour = int(finish // 3600)
    minute = int((finish - hour * 3600) // 60)
    second = int(finish - hour*3600 - minute*60)
    timetime =str(hour) +'h '+ str(minute)+'m ' + str(second)+'s'
    temp_log = "\nTime Elapse: %s \n\n" %(timetime)
    return temp_log


def FUL_data_load(inputX,inputY,testX,testY,AUG,switch): # switch : train/eval
    ### Dividing data into folds
    x_train = inputX
    x_val = testX
    y_train = inputY
    y_val = testY
    if(AUG=='hflip'):    
        temp_data = torch.flip(x_train,[1,]) # 0:batch  1:horizontal  2:forward&backward   3:vertical
        temp_label =  torch.zeros(len(y_train))

        for idx in range(len(y_train)):
            if(y_train[idx] == 1) :
                temp_label[idx]= 2
            elif(y_train[idx] == 2) :
                temp_label[idx]= 1 
            else :
                temp_label[idx]= y_train[idx]

        temp_label = temp_label.long()
        x_train = torch.cat([x_train,temp_data],dim=0)
        y_train = torch.cat([y_train,temp_label],dim=0)

    train_data = torch.utils.data.TensorDataset(x_train, y_train)
    val_data = torch.utils.data.TensorDataset(x_val, y_val)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size =BATCH, shuffle = True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size = BATCH, shuffle = switch)
    return train_loader, val_loader


def FUL_train(inputX,inputY,testX,testY,Rep,Sum):
    seed_set(KERNEL_SEED)
    trn_loss = torch.zeros(FULEPOCH+1) ; val_loss= torch.zeros(FULEPOCH+1) ; trn_acc = torch.zeros(FULEPOCH+1) ; val_acc = torch.zeros(FULEPOCH+1)
    fold = 0
    # Model Initialize
    print('\n\n==> Building model..')
    if(CONTROLTYPE=='CLRM'): net = HSCNN(ksize)
    net.to(device)
    if (device == 'cuda') and (ksize == 4): net = torch.nn.DataParallel(net)

#    summary(net, input_size=(1,160,200,170))
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size, gamma = lreduce , last_epoch=-1)
    #optimizer=optim.Adam(params=net.parameters(),lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=decay_rate)
    if(OPT=='SGD'):
        optimizer = optim.SGD(net.parameters(), lr=lr,momentum=MOM, weight_decay=wdecay)
    elif(OPT == 'Adam'):
        optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=wdecay)
        
    step_size = ENDEPOCH//lrperiod 
    if(SCH=='CALR'):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=step_size)
    elif(SCH=='SLR'):
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=lreduce)
    elif(SCH=='RLRP'):
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=step_size, factor=lreduce)
    train_loader, val_loader = FUL_data_load(inputX,inputY,testX,testY,AUG,True)

    for epoch in range(FULEPOCH+1):
        for param_group in optimizer.param_groups:
            print("\nCurrent learning rate is: {}".format(param_group['lr']))
        print('Epoch {} / {}  '.format(epoch + 1, FULEPOCH+1,))
        train(net,epoch,optimizer,criterion,trn_acc,trn_loss,train_loader)
        tst_loss = test(net,epoch,fold,criterion,val_acc,val_loss,val_loader)
        realtime_graph(SETT,TRIAL,AUG,KERNEL_SEED,fold,epoch,trn_acc,trn_loss,val_acc,val_loss,os.getcwd()+graph_path)
        if(SCH=='RLRP'): scheduler.step(tst_loss)
        else: scheduler.step()
    acclossGraph(trn_loss,trn_acc,val_loss,val_acc,Rep,fold,os.getcwd()+graph_path)

    del net, train_loader, val_loader


def test(net,epoch,fold,criterion,val_acc,val_loss,val_loader):
    net.eval()
    vld_acc = 0 ; vld_loss = 0
    correct = 0 ; total = 0
    with torch.no_grad():
        for batch_index, (images, labels) in enumerate(val_loader):
            images = images.view(-1,1,imgRow,imgCol,imgDepth)
            images = images.float()
            images = images.cuda()
            labels = labels.cuda()
            
            output = net(images)
            loss = criterion(output,labels)
            vld_loss += loss.item()
            _, pred = output.max(1)
            correct += pred.eq(labels).sum().item()
            total += labels.size(0)
            progress_bar(batch_index,len(val_loader), 'Loss: %.3f | Acc: %.3f%% ( %d / %d )' % (vld_loss/(batch_index+1), 100.*correct/total,correct, total))
            del images, labels
    
    if(epoch==FULEPOCH):
        final_acc = 100.*correct/total
        print('Final Accuracy : %f at %d th epoch ' %(final_acc, epoch+1))
        savePath = './saveModel/[%s%d%s]HS%s_D%d{F%dK%d}[best].pt'%(SETT,TRIAL,AUG,CONTROLTYPE,DATATYPE,FOLD_SEED,KERNEL_SEED)
        torch.save(net.state_dict(), savePath)

    val_loss[epoch] = vld_loss/(batch_index+1)
    val_acc[epoch] = 100.*correct/total
    return vld_loss/(batch_index+1)
    
    
def FUL_eval(inputX,inputY,testX,testY,Rep,Sum) :

    yIdxPrediction = []
    multi = [] ; binary = []
    fold=0
    model_path = os.getcwd()+'/saveModel/[%s%d%s]HS%s_D%d{F%dK%d}[best].pt'%(SETT,TRIAL,AUG,CONTROLTYPE,DATATYPE,FOLD_SEED,KERNEL_SEED)

    if(CONTROLTYPE=='CLRM'): net = HSCNN(ksize)
    elif('ADNI' in CONTROLTYPE): net = ADNICNN(ksize)
    net.to(device)
    if (device == 'cuda') and (ksize == 4): net = torch.nn.DataParallel(net)
    net.load_state_dict(torch.load(model_path))

    train_loader, tst_loader = FUL_data_load(inputX,inputY,testX,testY,AUG,False)
    yLP, yPB, yLPB, multi_acc, binary_acc = evaluation(net,tst_loader)
    multi.append(multi_acc)
    binary.append(binary_acc)

    yIdxPrediction = np.arange(len(inputY))
    yLabelPrediction = yLP
    yPredictionB = yPB
    yLabelPredictionB = yLPB

    del net, train_loader, tst_loader
    acc_roc(multi,binary,yIdxPrediction,yLabelPrediction,yPredictionB,yLabelPredictionB,Rep,Sum,iters,graph_path,tstCountNO_4,tstCountNO_7,tstCountNO_0,tstCountYES_1,tstCountYES_2)