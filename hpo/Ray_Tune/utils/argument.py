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
    parser.add_argument('--MODEL', type=str, default='3D_5124', help='model architecture type')
    parser.add_argument('--KERNEL_SEED', type=int, default=1, help='random seed')
    parser.add_argument('--CONTROLTYPE', type=str, default='CLRM', help=' C047 / C1 / C2 ')
    parser.add_argument('--SETT',type=str, default='SIG', help='Evaluated on the CV : SIG or the test dataset : FUL  ')
    parser.add_argument('--TRIAL',type=int, default=1)
    parser.add_argument('--AUG',type=str, default='hflip')
    parser.add_argument('--DATATYPE',type=int, default=60)
    parser.add_argument('--BATCH',type=int, default=42)
    parser.add_argument('--TALK',type=str)
    parser.add_argument('--ENDEPOCH',type=int, default=300)
    parser.add_argument('--FULEPOCH',type=int, default=299)
    parser.add_argument('--OPT',default='SGD',type=str)
    parser.add_argument('--SCH',default='CAWR',type=str)
    parser.add_argument('--MOM',default=0.9,type=float)

    ##======================== Ensemble ========================##

    parser.add_argument('--EnsMODE',type=str, default='None')
    parser.add_argument('--K', type=int, action='append')

    ##======================== LRP ========================##

    parser.add_argument('--RULE',type=str, default='None')
    parser.add_argument('--PERCENT',type=int, default=5)
    parser.add_argument('--LRPMSG',type=str, default='Obl')

    ##======================== HPO ========================##

    parser.add_argument('--ALGORITHM', type=str, help='name of sampler and pruner algorithms')
    parser.add_argument('--N_HPO',default=128,type=int, help='The number of HPO search; Trial')
    parser.add_argument('--CHECKPOINT', type=str, help='The location of the experimental results')
    parser.add_argument('--MARGIN',default=0.0,type=float)
    parser.add_argument('--find-lr', action='store_true', default=False, help='Search LR')
    parser.add_argument('--find-wd', action='store_true', default=False, help='Search WD')
    parser.add_argument('--find-mn', action='store_true', default=False, help='Search MN')
    parser.add_argument('--find-sch', action='store_true', default=False, help='Search SGDR T0')
    parser.add_argument('--find-dp', action='store_true', default=False, help='Search Dampening')
    parser.add_argument('--nesterov', action='store_true', default=False, help='Turn on the nesterov')



    parser.add_argument('--lreduce',default=0.5623413,type=float)
    parser.add_argument('--lrperiod',default=4,type=int)


    args = parser.parse_args()
    return args
