import os
import glob
import torch
import random
import shutil
import numpy as np


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(state: dict, checkpoint_dir: str, step: int, keep_last: int = 3):
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f'ckpt_step_{step}.pth')
    torch.save(state, path)
    # keep only last N checkpoints
    files = sorted(glob.glob(os.path.join(checkpoint_dir, 'ckpt_step_*.pth')), key=os.path.getmtime)
    while len(files) > keep_last:
        try:
            os.remove(files[0])
        except Exception:
            pass
        files = sorted(glob.glob(os.path.join(checkpoint_dir, 'ckpt_step_*.pth')), key=os.path.getmtime)


def load_checkpoint(path: str, model: torch.nn.Module = None, optimizer: torch.optim.Optimizer = None, device='cpu'):
    ckpt = torch.load(path, map_location=device)
    if model is not None and 'model_state' in ckpt:
        model.load_state_dict(ckpt['model_state'])
    if optimizer is not None and 'opt_state' in ckpt:
        optimizer.load_state_dict(ckpt['opt_state'])
    return ckpt


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = None
        self.counter = 0
        self.should_stop = False

    def step(self, metric):
        if self.best is None or metric > self.best + self.min_delta:
            self.best = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
