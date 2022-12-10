import ray
from utils import *
from ray import tune
from ray.tune.suggest.basic_variant import BasicVariantGenerator
from ray.tune.suggest.optuna import OptunaSearch
from ray.tune.schedulers import MedianStoppingRule

from utils.argument import get_args
args = get_args()

ALGORITHM=args.ALGORITHM
CHECKPOINT=args.CHECKPOINT
find_lr=args.find_lr
find_mn=args.find_mn
find_wd=args.find_wd
find_sch=args.find_sch
find_damp=args.find_dp
N_HPO=args.N_HPO
MARGIN=args.MARGIN
N_GPU = torch.cuda.device_count()
NESTEROV = args.nesterov

def create_config(study_name):
    
    tune_config = {'study':study_name}

    default_lr = 1e-2; lower_lr = 5e-3; upper_lr = 5e-1
    default_wd = 5e-4; lower_wd = 1e-3; upper_wd = 1e-2
    default_mn = 0.1; lower_mn = 1e-3; upper_mn = 1
    default_sch = 1; lower_sch = 1; upper_sch = 4
    lower_dp = 1e-12 ; upper_dp = 1

    if(find_lr):
        if(ALGORITHM=='grid'): tune_config['lr']=tune.grid_search(make_grid(lower_lr,upper_lr,30,log=True))
        else: tune_config['lr']=tune.loguniform(lower_lr, upper_lr)
    else: tune_config['lr']=default_lr

    if(find_wd):tune_config['wd']=tune.loguniform(lower_wd, upper_wd)
    else: tune_config['wd']=default_wd

    if(find_mn):tune_config['mn']=tune.loguniform(lower_mn, upper_mn)
    else: tune_config['mn']=default_mn # Applying the "1 - momentum"

    if(find_sch):tune_config['sch']=tune.randint(lower_sch, upper_sch)
    else: tune_config['sch']=default_sch # Applying the "1 - momentum"

    if(find_damp): tune_config['damp']=tune.loguniform(lower_dp, upper_dp)
    else: tune_config['damp']= 0 ;
        
    return tune_config



def objective(config, checkpoint_dir=None, inputX=None, inputY=None,Y_vector=None):
    seed_set(KERNEL_SEED)
    fold_acc = np.zeros(KFOLD.get_n_splits() * ENDEPOCH)
    fold_acc = np.reshape(fold_acc, [-1, ENDEPOCH])
    for fold, (train_index, val_index) in enumerate(KFOLD.split(inputX,Y_vector)):

        best_acc = 0
        # Model Initialize
        net = Torch3D(ksize)
        net.to(device)
        if (device == 'cuda'): net = torch.nn.DataParallel(net)

        optimizer = optim.SGD(net.parameters(), lr=config['lr'], weight_decay=config['wd'],momentum=1-config['mn'],
                             nesterov=NESTEROV, dampening = config['damp'])

        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=int(ENDEPOCH/config['sch']), T_mult=1)
        train_loader, val_loader = CV_data_load(inputX,inputY,train_index,val_index,AUG,True)

        for epoch in range(ENDEPOCH):
            hpo_train(net,epoch,optimizer,criterion,train_loader)
            test_acc = hpo_eval(net,criterion,val_loader)
            scheduler.step()

            fold_acc[fold,epoch] = test_acc
            mean_test_acc = fold_acc[:fold+1,epoch].mean()
            best_acc = max(mean_test_acc, best_acc)
            
            with tune.checkpoint_dir(epoch) as checkpoint_dir:
                path = os.path.join(checkpoint_dir, "checkpoint")
                checkpoint_state = {
                    "epoch": epoch + 1,
                    "best_acc": best_acc
                }
                torch.save(checkpoint_state, path)

            print(
                'Epoch {} / {}   Fold {} / {}  Test acc {}  Mean acc {} Best mean acc {}'.format(
                epoch + 1, ENDEPOCH, 
                fold + 1 , KFOLD.get_n_splits(), 
                test_acc, mean_test_acc, best_acc )
            )
            tune.report(acc = test_acc, mean_acc=mean_test_acc, best_acc=best_acc) # Used for Pruning and Sampling

        del net, train_loader, val_loader


def hpo_train(net, epoch,optimizer,criterion,train_loader):
    net.train()
    train_loss = 0 ; correct = 0 ; total = 0
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
        del images
        del labels

def hpo_eval(net,criterion,val_loader):
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
            
    acc = correct/total
    return acc

# Generate Trial String Representation
def trial_name_id(trial):
    tag = trial.experiment_tag.partition('_')[0]
    return "{}".format(tag)

# Data Loading
inputX, inputY, Y_vector, _, _ = source_load(categories, dirDataSet)

if __name__ == '__main__':

    # Initialize Ray
    ray.init() 

    # Configure Settings
    study_name = os.path.basename(os.path.normpath(CHECKPOINT)) # get last directory of the provided checkpoint path
    local_path = os.path.abspath(os.path.join(CHECKPOINT, os.pardir)) # get first directory of the provided checkpoint path
    tune_config = create_config(study_name)

    # Configure Console Output (CLI Reporter)
    reporter = tune.CLIReporter()
    reporter.add_metric_column("acc")
    reporter.add_metric_column("mean_acc")
    reporter.add_metric_column("best_acc")


    if ("tpe" in ALGORITHM):
#        pts, rew = get_preeval_results(['checkpoints/exp26_grid_lr', 'checkpoints/exp26_grid_wd', 'checkpoints/exp26_grid_mn'], ["lr", "wd", "mn"])
        sampler = OptunaSearch(
            sampler= util_tpe.TPESampler(n_startup_trials=10, n_ei_candidates=24, margin = MARGIN,
                    seed=KERNEL_SEED, multivariate=True, constant_liar=False, record = CHECKPOINT)
        )
    elif ("random" in ALGORITHM):
        sampler = BasicVariantGenerator(max_concurrent=N_GPU)
    elif ("grid" in ALGORITHM):
        sampler = BasicVariantGenerator(max_concurrent=N_GPU)

    if "prune" in ALGORITHM:
        pruner = MedianStoppingRule()
    else:
        pruner = None # default: FIFOScheduler



    # Run Experiments
    print('Selected Algorithm: {}'.format(ALGORITHM))
    tune.run(
        tune.with_parameters(objective, inputX=inputX, inputY=inputY, Y_vector=Y_vector),
        name = study_name,
        metric = "best_acc",
        mode = "max",
        config = tune_config,
        resources_per_trial = {"cpu": 4, "gpu": 1},
        num_samples = N_HPO,
        local_dir = local_path,
        search_alg = sampler,
        scheduler = pruner,
        log_to_file = True,
        keep_checkpoints_num = 1,
        checkpoint_score_attr = "best_acc",
        progress_reporter = reporter,
        trial_name_creator = trial_name_id,
        raise_on_failed_trial = False,
        resume = False
    )

    ray.shutdown()
    end_time = time.time() - start
    print('Total run time: {}s (approx. {})'.format(end_time, convert_sec(end_time)))

    # Save Results
    analysis, df_analysis, best_config, best_accuracy = load_tune_analysis(f'checkpoints/{study_name}')
    print('Best Configurations: {} | Best Accuracy: {}'.format(best_config, best_accuracy))


