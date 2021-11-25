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
    parser.add_argument('--MODEL', type=str, default='3D_BASIC', help='model architecture type')
    parser.add_argument('--KERNEL_SEED', type=int, default=1, help='random seed')
    parser.add_argument('--CONTROLTYPE', type=str, default='CLRM', help=' C047 / C1 / C2 ')
    parser.add_argument('--SETT',type=str, default='SIG', help='Evaluated on the CV : SIG or the test dataset : FUL  ')
    parser.add_argument('--TRIAL',type=int, default=1)
    parser.add_argument('--AUG',type=str, default='hflip')
    parser.add_argument('--DATATYPE',type=int)
    parser.add_argument('--BATCH',type=int, default=42)
    parser.add_argument('--TALK',type=str)
    parser.add_argument('--ENDEPOCH',type=int, default=300)
    parser.add_argument('--FULEPOCH',type=int, default=299)
    parser.add_argument('--OPT',default='Adam',type=str)
    parser.add_argument('--SCH',default='SLR',type=str)
    parser.add_argument('--MOM',default=0.9,type=float)

    parser.add_argument('--mgpu',default=0,type=int, help='debug mode')
    parser.add_argument('--debug',default=0,type=int, help='debug mode')
    parser.add_argument('--lr',default=1e-2,type=float)
    parser.add_argument('--lreduce',default=0.5623413,type=float)
    parser.add_argument('--lrperiod',default=4,type=int)
    parser.add_argument('--wdecay',default=5e-4,type=float)
    parser.add_argument('--ksize',default=3,type=int)

    ##======================== Ensemble ========================##

    parser.add_argument('--EnsMODE',type=str, default='None')
    parser.add_argument('--K', type=int, action='append')

    ##======================== LRP ========================##

    parser.add_argument('--RULE',type=str, default='None')
    parser.add_argument('--PERCENT',type=int, default=5)
    parser.add_argument('--LRPMSG',type=str, default='Obl')

    args = parser.parse_args()
    return args