import warnings
warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")
from typing import Callable

import os
import sys

import torch


import pandas as pd
from tqdm import tqdm

# add src to python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils.BasicUtils import get_device_from_gpu_id, get_metric_from_str, PartyPath



# evaluate the model on the test set
def evaluate(model, test_loader, metric_fn: Callable, gpu_id=0, n_classes=1):
    device = get_device_from_gpu_id(gpu_id)
    model.to(device)
    model.eval()
    with torch.no_grad():
        y_all = torch.zeros([0, 1]).to(device)
        y_pred_all = torch.zeros([0, 1]).to(device)
        for Xs, y in test_loader:
            # to device
            Xs = [Xi.to(device) for Xi in Xs]
            y = y.to(device).reshape(-1, 1)
            y_pred = model(Xs)
            if n_classes == 1:
                y_pred = y_pred.reshape(-1, 1)
            else:
                y_pred = torch.argmax(y_pred, dim=1).reshape(-1, 1)
            y_pred_all = torch.cat((y_pred_all, y_pred), dim=0)
            y_all = torch.cat((y_all, y), dim=0)
        y_pred_all = y_pred_all.cpu().numpy()
        y_all = y_all.cpu().numpy()
    return metric_fn(y_pred_all, y_all)
