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
    parser.add_argument('--seed', type=int, default=1, required=True, help='dataset to use')
    parser.add_argument('--interest', type=int, default=1, required=True, help='dataset to use')
    parser.add_argument('--perp', type=int, default=30, required=True, help='dataset to use')
    parser.add_argument('--epsi', type=int, default=200, required=True, help='dataset to use')
    parser.add_argument('--step', type=int, default=1000, required=True, help='dataset to use')
    parser.add_argument('--select', type=str, default='tSNE', help='dataset to use')
    
    args = parser.parse_args()
    return args