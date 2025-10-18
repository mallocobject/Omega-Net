import torch
import numpy as np
from accelerate import Accelerator


class EarlyStopping:
    def __init__(self, accelerator: Accelerator, patience=7, delta=0, save_mode=True):
        pass

    def __call__(self, val_loss, model, path):
        pass

    def save_checkpoint(self, val_loss, model, path):
        pass
