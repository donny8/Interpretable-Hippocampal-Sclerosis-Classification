#!/bin/sh
#!/bin/bash

# Average ensemble
CUDA_VISIBLE_DEVICES=1 python HS_ensemble.py --AUG hflip --TRIAL 9 --SETT SIG --DATATYPE 64 --CONTROLTYPE CLRM --EnsMODE AVR --KERNEL_SEED 1

# Voting ensemble
CUDA_VISIBLE_DEVICES=1 python HS_ensemble.py --AUG hflip --TRIAL 9 --SETT SIG --DATATYPE 64 --CONTROLTYPE CLRM --EnsMODE VOT --KERNEL_SEED 1