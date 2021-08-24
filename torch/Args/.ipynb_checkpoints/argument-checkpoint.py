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
    parser.add_argument('--KERNEL_SEED', type=int, default=1, required=True, help='dataset to use')
    parser.add_argument('--CONTROLTYPE', type=str, default='CC', required=True, help=' Multi / Binary ')
    parser.add_argument('--SETT',type=str, default='SIG', help='Concatenate or Single ')
    parser.add_argument('--TRIAL',type=int, default=1)
    parser.add_argument('--AUG',type=str, default='hflip')
    parser.add_argument('--DATATYPE',type=int)
    parser.add_argument('--BATCH',type=int, default=16)
    parser.add_argument('--TALK',type=str)
    parser.add_argument('--ENDEPOCH',type=int, default=200)

    parser.add_argument('--debug',default=0,type=int, help='debug mode')
    parser.add_argument('--drop',default=0,type=int, help='Whether to drop last in batch')
    parser.add_argument('--step',default=75,type=int)
    parser.add_argument('--lr',default=1e-2,type=float)
    
    ##======================== Ensemble ========================##
    parser.add_argument('--EnsMODE',type=str, default='None')
    ##======================== LRP ========================##
    parser.add_argument('--FOLDNUM',type=int, default=0)
    parser.add_argument('--RULE',type=str, default='None')

    args = parser.parse_args()
    return args