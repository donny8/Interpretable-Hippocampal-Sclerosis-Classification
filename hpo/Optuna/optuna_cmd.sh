#!/bin/sh
#!/bin/bash



#CUDA_VISIBLE_DEVICES=1 python HST_main.py --CONTROLTYPE CLRM --SETT SIG --AUG hflip --KERNEL_SEED 2 --TRIAL 25 --DATATYPE 67 --BATCH 42 --MODEL 3D_5124 --step 50 --TALK new_loss_2 --drop 1
#CUDA_VISIBLE_DEVICES=2 python HST_main.py --CONTROLTYPE CLRM --SETT SIG --AUG hflip --KERNEL_SEED 14 --TRIAL 25 --DATATYPE 67 --BATCH 42 --MODEL 3D_5124 --step 50 --TALK new_loss_14 --drop 1
#CUDA_VISIBLE_DEVICES=2 python HST_main.py --CONTROLTYPE CLRM --SETT SIG --AUG hflip --KERNEL_SEED 19 --TRIAL 25 --DATATYPE 67 --BATCH 42 --MODEL 3D_5124 --step 50 --TALK new_loss_19 --drop 1



#CUDA_VISIBLE_DEVICES=1 python HST_ensemble.py --CONTROLTYPE CLRM --SETT SIG --AUG hflip --KERNEL_SEED 25 --TRIAL 25 --DATATYPE 67 --BATCH 42 --MODEL 3D_5124 --step 50 --TALK AVR_ensemble --K 5  --K 18 --K 22 --K 28 --K 38 --EnsMODE AVR
#CUDA_VISIBLE_DEVICES=1 python HST_ensemble.py --CONTROLTYPE CLRM --SETT SIG --AUG hflip --KERNEL_SEED 25 --TRIAL 25 --DATATYPE 67 --BATCH 42 --MODEL 3D_5124 --step 50 --TALK VOT_ensemble --K 5  --K 18 --K 22 --K 28 --K 38 --EnsMODE VOT



#CUDA_VISIBLE_DEVICES=2 python HST_opt.py --mode DUAL --epoch 100 --ntrial 40 --prune 1 --datatype 60 --seed 1 --opt SGD --sch RLRP --cma 1 --msg BEST_acc_CMA
#python slack.py --experiment "NIPA_HS_BEST_CMA"

# Optuna
CUDA_VISIBLE_DEVICES=0 python HST_opt.py --mode BEST_acc --epoch 150 --ntrial 128 --prune 1 --datatype 60 --seed 2 --msg GRAD-INDEP-TPE --controltype CLRM --opt SGD --sch RLRP  --batch 42 --pruner MED
python slack.py --expr "GRAD-INDEP-TPE"



CUDA_VISIBLE_DEVICES=0 python HST_opt.py --mode BEST_acc --epoch 150 --ntrial 128 --prune 1 --datatype 60 --seed 2 --msg GRAD-MV-TPE --controltype CLRM --opt SGD --sch RLRP  --batch 42 --pruner MED --multiV 1
python slack.py --expr "GRAD-Multi-Variate-TPE"


# Debug
#CUDA_VISIBLE_DEVICES=0 python HST_opt.py --mode BEST_acc --epoch 10 --ntrial 2 --prune 1 --datatype 64 --seed 2 --msg GRAD_BEST_acc_SGD_RLRP --controltype CLRM --opt SGD --sch RLRP  --batch 4 --ksize 3
#CUDA_VISIBLE_DEVICES=0 python HST_opt.py --mode BEST_acc --epoch 10 --ntrial 2 --prune 1 --datatype 64 --seed 2 --msg GRAD_BEST_acc_SGD_RLRP --controltype CLRM --opt SGD --sch RLRP  --batch 4 --ksize 4

#CUDA_VISIBLE_DEVICES=2 python HST_opt.py --mode BEST_acc --epoch 75 --ntrial 200 --prune 1 --datatype 35 --seed 1 --msg BEST_acc_SGD_RLRP_ADNI12_B40_LRperd15 --controltype ADNI_BN --opt SGD --sch RLRP  --batch 40 --lrperiod 8 
