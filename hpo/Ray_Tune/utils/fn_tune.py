"""
HPO Framework: Ray-Tune
"""
import os
import numpy as np
import pandas as pd
import torch
#import wandb

from ray import tune
#from ray.tune.integration.wandb import wandb_mixin

__all__ = ['load_tune_analysis', 'get_best_trial_curve', 'get_trial_order', 'get_trial_curves', 'make_grid', 'get_preeval_results', 'dummy']

# Load Analysis of the Result
def load_tune_analysis(experiment_dir):
    analysis = tune.ExperimentAnalysis(experiment_dir)
    df_analysis = analysis.dataframe()
    best_config = analysis.get_best_config(metric="best_acc", mode="max")
    # Note that tune.ExperimentAnalysis() currently has a bug (https://github.com/ray-project/ray/issues/9419).
    idx = df_analysis["best_acc"].idxmax()
    best_accuracy = df_analysis["best_acc"][idx]
    return analysis, df_analysis, best_config, best_accuracy

# Get Training Curve of the Best Trial
def get_best_trial_curve(df, val='accuracy'):
    idx = df["best_acc"].idxmax()
    log_path = df["logdir"][idx]
    data = pd.read_csv(log_path+'/progress.csv', sep=',')
    if val == 'accuracy':
        best_curve = np.array(data["acc"]).tolist()
    if val == 'loss':
        best_curve = np.array(data["loss"]).tolist()
    return best_curve

# Get Training Curves of All the Trials
def _find_str_idx(s, ch):
    return [i for i, letter in enumerate(s) if letter == ch]

def get_trial_order(df):
    log_path = [os.path.basename(os.path.normpath(p)) for p in df["logdir"].tolist()]
    trial_order = []
    for i, name in enumerate(log_path):
        idx = _find_str_idx(name, '_')
        trial_order.append((int(name[:idx[0]]), i)) # no `-1` for random + ASHA
    flag_tag = (0 in [x for x, _ in trial_order])
    # Note: depending on the search algorithm in Ray Tune, experiment starts with 0 (grid, random) or 1 (BOHB)
    # To account for this, if tag starts with 1, we subtract -1 to match Pythonic indexing
    if not flag_tag:
        trial_order = [((x-1), y) for x, y in trial_order]
    trial_order = sorted(trial_order, key=lambda x: x[0])
    trial_order = [t[1] for t in trial_order]
    return trial_order

def get_trial_curves(df, val='accuracy'):
    curves = []
    trial_order = get_trial_order(df)
    for t in range(len(trial_order)):
        idx = trial_order[t]
        log_path = df["logdir"][idx]
        data = pd.read_csv(log_path+'/progress.csv',sep=',')
        if val == 'accuracy':
            curve = np.array(data["acc"])
        if val == 'loss':
            curve = np.array(data["loss"])
        curves.append(curve)
    return curves

# Compute List of Values for Grid Search
def make_grid(lb, ub, n_grid, log=True):
    if log:
        lb = np.log(lb)
        ub = np.log(ub)
    interval = (ub - lb) / n_grid
    grid = np.arange(lb, ub, interval)
    if log: grid = np.e ** grid
    return grid.tolist()

# Extract Pre-evaluated Points and Rewards
def get_preeval_results(experiment_dirs, config_names):
    assert len(experiment_dirs) == len(config_names), 'Size of inputs must match. \n'
    pts = []
    rew = []
    for n in range(len(experiment_dirs)):
        experiment_dir = experiment_dirs[n]
        config_name = config_names[n]
        _, df, _, _ = load_tune_analysis(experiment_dir)
        # Get Evaluated Rewards
        rew = rew + df["best_acc"].tolist()
        # Get Evaluated Points
        base_vals = {"lr": 0.1, "wd": 5e-4, "mn": 0.9}
        base_name = list(base_vals.keys())
        samp_vals = df["config/"+config_name].tolist()
        samp_keys = [config_name] * len(samp_vals)
        hp = []
        for s in base_name:
            if s == config_name:
                dics = [{d[0]: d[1]} for d in list(zip(*[samp_keys, samp_vals]))]
                hp.append(dics)
            else:
                keys = [s] * len(samp_vals) 
                vals = [base_vals[s]] * len(samp_vals)
                dics = [{d[0]: d[1]} for d in list(zip(*[keys, vals]))]
                hp.append(dics)
        hp = list(map(list, zip(*hp)))
        evp = []
        for h in hp:
            evp.append({k: v for d in h for k, v in d.items()})
        pts = pts + evp
    return pts, rew

# Dummy Trainable Function
#@wandb_mixin
def dummy(config, checkpoint_dir=None, args=None):
    start_epoch = 0
    # Load checkpoint
    if checkpoint_dir:
        checkpoint = torch.load(os.path.join(checkpoint_dir, "checkpoint"))
        test_score = checkpoint['test_score']
        start_epoch = checkpoint['epoch']
    # Compute objective value
    for epoch in range(start_epoch, args.epochs):
        test_score = np.abs(np.random.randn() * config["a"] + config["b"])
        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            checkpoint_state = {
                "epoch": epoch + 1,
                "test_score": test_score
            }
            torch.save(checkpoint_state, path)
        tune.report(test_score=test_score)
        wandb.log(dict(test_score=test_score))
    return None