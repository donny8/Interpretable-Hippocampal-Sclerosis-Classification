#!/bin/sh
#!/bin/bash

# Single Model Training
CUDA_VISIBLE_DEVICES=1 python HS_main.py --CONTROLTYPE CLRM --SETT SIG --AUG hflip --KERNEL_SEED 1 --TRIAL 1 --DATATYPE 64 --BATCH 16 --MODEL 3D_BASIC5124 --FC1 64 --FC2 64 