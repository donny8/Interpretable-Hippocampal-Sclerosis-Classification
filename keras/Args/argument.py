import argparse
import torch

def str2bool(v):
    if v.lower() in ('yes', 'true','True', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'False','f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--MODEL', type=str, default='3D_BASIC', help='net type')
    parser.add_argument('--TEST_EPOCH', type=int, default=1, help='which epoch to use')
    parser.add_argument('--KERNEL_SEED', type=int, default=1, required=True, help='dataset to use')
    parser.add_argument('--FOLD_SEED', type=int, default=0, help='5-fold-validation seed')
    parser.add_argument('--CONTROLTYPE', type=str, default='CC', required=True, help=' Multi / Binary ')
    parser.add_argument('--SETT',type=str, default='CAT', help='Concatenate or Single ')
    parser.add_argument('--TRIAL',type=int, default=1)
    parser.add_argument('--AUG',type=str, default='None')
    parser.add_argument('--DATATYPE',type=int)
    parser.add_argument('--BATCH',type=int, default=16)
    parser.add_argument('--FC1',type=int, default=32)
    parser.add_argument('--FC2',type=int, default=8)

    ##======================== ENS ========================##
    parser.add_argument('--EnsMODE',type=str, default='None')
    ##======================== LRP ========================##
    parser.add_argument('--PERCENT',type=int, default=40)
    parser.add_argument('--RULE',type=str, default='None')
    parser.add_argument('--LRPMSG',type=str, default='Obl')
    ##======================== PRED =======================##
    parser.add_argument('--LOADATA',type=int)
    parser.add_argument('--LOADCONTROL',type=str)
    parser.add_argument('--NUMODEL',type=int)
    ##===================== BALANCE =======================##
    parser.add_argument('--LOSS1',type=float, default=1.0)
    parser.add_argument('--LOSS2',type=float, default=1.0)
    parser.add_argument('--BALANCE',type=int, default=0)
    
    
    args = parser.parse_args()
    return args