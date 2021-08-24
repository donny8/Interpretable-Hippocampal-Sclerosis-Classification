import argparse
#import torch

def str2bool(v):
    if v.lower() in ('yes', 'true','True', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'False','f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--SEED', type=int, default=1, required=True, help='Seed of the model you want to load')
    parser.add_argument('--INTEREST', type=int, default=1, required=True, help='Layer of interest')
    parser.add_argument('--PERP', type=int, default=30, required=True, help='Perplexity')
    parser.add_argument('--EPSI', type=int, default=200, required=True, help='Episode')
    parser.add_argument('--STEP', type=int, default=1000, required=True, help='The number of steps')
    parser.add_argument('--SELECT', type=str, default='UMAP', help='Dimensionality reduction methods : UMAP or tSNE')
    
    parser.add_argument('--AUG',type=str, default='hflip', help='Data augmentation')
    parser.add_argument('--TRIAL',type=int, default=1, help='Just to differentiate')
    parser.add_argument('--SETT',type=str, default='SIG', help='Concatenate or Single')
    parser.add_argument('--CONTROLTYPE', type=str, default='CC', required=True, help=' Multi-class / Binary class ')

    args = parser.parse_args()
    return args