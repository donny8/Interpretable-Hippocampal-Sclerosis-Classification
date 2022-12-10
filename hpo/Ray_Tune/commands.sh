#!/bin/sh
#!/bin/bash

CUDA_VISIBLE_DEVICES=1,2 python HST_hpo.py --ALGORITHM 'tpe_dp' --N_HPO 256 --CHECKPOINT 'checkpoints/tpe_dp' --find-lr --find-wd --find-mn --find-dp &>> checkpoints/tpe_dp.log

CUDA_VISIBLE_DEVICES=1,2 python HST_hpo.py --ALGORITHM 'tpe_nestrov' --N_HPO 256 --CHECKPOINT 'checkpoints/tpe_nesterov' --find-lr --find-wd --find-mn --nesterov &>> checkpoints/tpe_nesterov.log
