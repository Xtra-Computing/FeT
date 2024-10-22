import os
import pathlib
from typing import Callable


import torch
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

from src.metric.RMSE import RMSE


class PartyPath:
    def __init__(self, dataset_path, n_parties, party_id=0, splitter='imp', weight=1, beta=1, seed=None,
                 fmt='pkl', comm_root=None):
        self.dataset_path = dataset_path
        path = pathlib.Path(self.dataset_path)
        self.dataset_name = path.stem
        self.n_parties = n_parties
        self.party_id = party_id
        self.splitter = splitter
        self.weight = weight
        self.beta = beta
        self.seed = seed
        self.fmt = fmt
        self.comm_root = comm_root  # the root of communication log

    def data(self, type='train'):
        path = pathlib.Path(self.dataset_path)
        if type is None:
            type_str = ''
        else:
            type_str = f"_{type}"
        if self.splitter == 'imp':
            # insert meta information before the file extension (extension may not be .csv)
            path = path.with_name(f"{path.stem}_party{self.n_parties}-{self.party_id}_{self.splitter}"
                                  f"_weight{self.weight:.1f}"
                                  f"{'_seed' + str(self.seed) if self.seed is not None else ''}{type_str}.{self.fmt}")
        elif self.splitter == 'corr':
            path = path.with_name(f"{path.stem}_party{self.n_parties}-{self.party_id}_{self.splitter}"
                                  f"_beta{self.beta:.1f}"
                                  f"{'_seed' + str(self.seed) if self.seed is not None else ''}{type_str}.{self.fmt}")
        else:
            raise NotImplementedError(f"Splitter {self.splitter} is not implemented. "
                                      f"Splitter should be in ['imp', 'corr']")
        return str(path)

    @property
    def train_data(self):
        return self.data('train')

    @property
    def test_data(self):
        return self.data('test')

    @property
    def comm_log(self):
        if self.comm_root is None:
            raise FileNotFoundError("comm_root is None")
        comm_dir = os.path.join(self.comm_root, self.dataset_name)
        os.makedirs(comm_dir, exist_ok=True)
        path = pathlib.Path(comm_dir)
        if self.splitter == 'imp':
            path = path / (f"{self.dataset_name}_party{self.n_parties}_{self.splitter}_weight{self.weight:.1f}"
                           f"{'_seed' + str(self.seed) if self.seed is not None else ''}.log")
        elif self.splitter == 'corr':
            path = path / (f"{self.dataset_name}_party{self.n_parties}_{self.splitter}_beta{self.beta:.1f}"
                           f"{'_seed' + str(self.seed) if self.seed is not None else ''}.log")
        else:
            raise NotImplementedError(f"Splitter {self.splitter} is not implemented."
                                      f" splitter should be in ['imp', 'corr']")
        return str(path)

# def party_path(dataset_path, n_parties, party_id, splitter='imp', weight=1, beta=1, seed=None, type='train',
#                fmt='pkl') -> str:
#     assert type in ['train', 'test']
#     path = pathlib.Path(dataset_path)
#     if splitter == 'imp':
#         # insert meta information before the file extension (extension may not be .csv)
#         path = path.with_name(f"{path.stem}_party{n_parties}-{party_id}_{splitter}"
#                               f"_weight{weight:.1f}{'_seed' + str(seed) if seed is not None else ''}_{type}.{fmt}")
#     elif splitter == 'corr':
#         path = path.with_name(f"{path.stem}_party{n_parties}-{party_id}_{splitter}"
#                               f"_beta{beta:.1f}{'_seed' + str(seed) if seed is not None else ''}_{type}.{fmt}")
#     else:
#         raise NotImplementedError(f"Splitter {splitter} is not implemented. splitter should be in ['imp', 'corr']")
#     return str(path)


def get_device_from_gpu_id(gpu_id):
    if gpu_id is None:
        return torch.device('cpu')
    else:
        return torch.device(f'cuda:{gpu_id}')


def get_metric_from_str(metric) -> Callable:
    supported_list = ['acc', 'rmse', 'r2']
    assert metric in supported_list
    if metric == 'acc':
        return accuracy_score
    elif metric == 'rmse':
        return lambda y_true, y_pred: RMSE()(y_true, y_pred)
    elif metric == 'r2':
        return r2_score
    else:
        raise NotImplementedError(f"Metric {metric} is not implemented. metric should be in {supported_list}")


def get_metric_positive_from_str(metric) -> bool:
    supported_list = ['acc', 'rmse', 'r2']
    assert metric in supported_list
    if metric in ['acc', 'r2']:
        return True
    elif metric in ['rmse']:
        return False
    else:
        raise NotImplementedError(f"Metric {metric} is not implemented. metric should be in {supported_list}")


def get_split_points(array, size):
    assert size > 1

    prev = array[0]
    split_points = [0]
    for i in range(1, size):
        if prev != array[i]:
            prev = array[i]
            split_points.append(i)

    split_points.append(size)
    return split_points


def move_item_to_end_(arr, items):
    for item in items:
        arr.insert(len(arr), arr.pop(arr.index(item)))


def move_item_to_start_(arr, items):
    for item in items[::-1]:
        arr.insert(0, arr.pop(arr.index(item)))


def equal_split(n, k):
    if n % k == 0:
        return [n // k for _ in range(k)]
    else:
        return [n // k for _ in range(k - 1)] + [n % k]
