from HST_util_opt import *


def objective(trial):
    seed_set(opts.seed)
    inputX, inputY, Y_vector, _, _ = source_load(categories, dirDataSet)
    avr_acc = []; avr_lss = []
    
    # ============================================ Optuna Hyperparam Target ============================================ #
    # Learning rate
    lr = trial.suggest_float("lr",5e-3,5e-1,log=True)
        
    # Weight decay
    if(opts.wdecay==None):weight_decay = trial.suggest_float("weight_decay",1e-4,1e-3)
    else: weight_decay = opts.wdecay
                    
    # Optimizer
    if(opts.opt==None): optimizer_name = trial.suggest_categorical("optimizer",["RMS","SGD"])
    else: optimizer_name = opt
        
    if(opts.mom == None) : momentum = trial.suggest_float("momentum",0,1)
    else : momentum = opts.mom
                    
    # LR Scheduler
    if(sch==None): scheduler_name = trial.suggest_categorical("scheduler",["CALR","SLR"])
    else : scheduler_name = sch

    # LR period
    start_epoch = 0
    if(opts.lrperiod == None): lrperiod = trial.suggest_int("lr_period",6,10) #2,8
    else: lrperiod = opts.lrperiod        
            
    # LR reduce
    if(opts.lreduce == 0.0): lreduce = trial.suggest_float("lr_reduce",5e-2,0.5)
    else: lreduce = opts.lreduce
        
    if(opts.debug):
        end_epoch = 4
        step_epoch = end_epoch
    else: 
        end_epoch = opts.epoch
        step_epoch = end_epoch//lrperiod
                    
    # Batch size
    if(opts.batch ==128): batch_size = trial.suggest_int("batch",16,42)
    else: batch_size = opts.batch
#    if(opts.ksize==0): ksize = trial.suggest_int("kernel_size",3,4)
#    else: ksize=opts.ksize
    ksize = 4
    # ================================================================================================================== #
    
    for fold, (train_index, val_index) in enumerate(KFOLD.split(inputX,Y_vector)):
        # Model Initialize
#        if(controltype=='CLRM'): net = HSCNN()
#        elif('ADNI' in controltype): net = ADNICNN(ksize)
        net = HSCNN(ksize)
        net.to(device)
        if device == 'cuda': net = torch.nn.DataParallel(net)

        if(optimizer_name=='SGD'): optimizer = optim.SGD(net.parameters(), lr=lr,momentum=momentum, weight_decay=weight_decay, nesterov=True)
        elif(optimizer_name == 'Adam'): optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay)
        elif(optimizer_name == 'RMS'): optimizer = torch.optim.RMSprop(net.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum, centered=False)

        if(scheduler_name=='CALR'): scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=step_epoch)
        elif(scheduler_name=='SLR'): scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_epoch, gamma=lreduce)
        elif(scheduler_name=='RLRP'): scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=step_epoch, factor=lreduce)
        
        train_loader, val_loader = CV_data_load(inputX,inputY,train_index,val_index,'hflip',batch_size)

        bstAcc = 0
        for epoch in range(end_epoch):
            train(net,optimizer,criterion,train_loader)
            val_acc, val_loss = validation(net,criterion,val_loader)
            if(mode != 'DUAL'): trial.report(val_acc,step=epoch+fold*end_epoch)
            avr_acc.append(val_acc)
            avr_lss.append(val_loss)
            if(val_acc > bstAcc): bstAcc = val_acc
            print('Fold %d | Epoch %d / %d | Acc %.3f | Best %.3f'%(fold+1,epoch+1,end_epoch,val_acc, bstAcc), end='\r',flush=True)
            
            if(mode != "DUAL"):
                if opts.prune:
                    if trial.should_prune(): 
                        raise optuna.TrialPruned()
            if(scheduler_name=='RLRP'): scheduler.step(val_loss)
            else: scheduler.step()

        del net, train_loader, val_loader
    del inputX

    if("acc" in mode) or ("DUAL" in mode): option = 1
    bestacc, lastacc = best_last(avr_acc,trial,option)
    if("loss" in mode) or ("DUAL" in mode): option = 0
    bestlss, lastlss = best_last(avr_lss,trial,option)
    
    if ("acc" in mode): 
        if("BEST" in mode) : return bestacc
        elif("LAST" in mode) : return lastacc
        
    elif ("loss" in mode):
        if("BEST" in mode) : return bestlss
        elif("LAST" in mode) : return lastlss

    else:
        return lastacc, lastlss
    
def train(net,optimizer,criterion,train_loader):
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
        del images, labels

    
    
def validation(net,criterion,val_loader):
    net.eval()
    vld_loss = 0 ; correct = 0 ; total = 0
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
            del images, labels
            
    # Save checkpoint.
    acc = correct/total
    loss = vld_loss/(batch_index+1)
    return acc, loss

if __name__ == "__main__":
    directions = []
    if not('loss' in mode): directions.append('maximize')   
    if not('acc' in mode): directions.append('minimize')

    if (mode == 'DUAL'):
        study = optuna.create_study(sampler=NSGAIISampler(seed=opts.seed), directions=directions)
    else:
        optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
        
        if(opt!=None)and(sch!=None)and(cma): sampler = CmaEsSampler(seed=opts.seed)
        else: 
            if(multiV): sampler = TPESampler(seed=opts.seed, multivariate=True)
            else: sampler = TPESampler(seed=opts.seed)
            
        if(prune):
            if(opts.pruner=='MED'): pruner = optuna.pruners.MedianPruner()
            elif(opts.pruner=='HYP'): pruner = optuna.pruners.HyperbandPruner()
            study = optuna.create_study(sampler=sampler, pruner=pruner, directions=directions)
        else:study = optuna.create_study(sampler=sampler, directions=directions)

    if(opts.debug): study.optimize(objective, n_trials=2)
    else: study.optimize(objective, n_trials=opts.ntrial, catch=(RuntimeError,), gc_after_trial=True)        
    study_record(study)