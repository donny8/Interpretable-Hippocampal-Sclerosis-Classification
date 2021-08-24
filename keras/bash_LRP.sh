#!/bin/sh
#!/bin/bash

# Cutoff : 5%
CUDA_VISIBLE_DEVICES=0 python HS_LRP.py --SETT SIG --TRIAL 9 --RULE lrp.z --KERNEL_SEED 1 --CONTROLTYPE CLRM --AUG hflip --DATATYPE 64 --PERCENT 5 --LRPMSG Directory_name