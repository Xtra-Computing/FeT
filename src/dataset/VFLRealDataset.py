import abc
import os.path
import sys
from typing import Sequence, List, Tuple, Optional
from copy import deepcopy
import pickle
import multiprocessing as mp
import ctypes

import numpy as np
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.dataset.LocalDataset import LocalDataset
from src.dataset.VFLDataset import VFLSynAlignedDataset
from src.utils.BasicUtils import PartyPath
from src.dataset.DataSampler import Top1Sampler, TopkUniformSampler, TopkSimAsProbSampler, TopkSampler, RandomSampler


class VFLRealDataset(Dataset):
    def __init__(self, local_datasets=None, primary_party_id=0, key_cols=None, multi_primary=False,
                 primary_train_local_dataset=None, ks=100, sample_rate_before_topk=None, cache_key=None,
                 use_cache=False, primary_train_indices=None, **kwargs):
        self.key_cols = key_cols
        self.sample_rate_before_topk = sample_rate_before_topk
        if local_datasets is None or isinstance(local_datasets[0], LocalDataset):
            self.local_datasets = local_datasets
        elif isinstance(local_datasets[0], Sequence):
            # (List[numpy.ndarray], ndarray)  =>  ([X1, X2, ...], y)
            Xs, y = local_datasets
            self.local_datasets = []
            for i, X in enumerate(Xs):
                if key_cols is not None:
                    key = X[:, :key_cols]
                    X = X[:, key_cols:]
                else:
                    key = None
                if i == primary_party_id:
                    self.local_datasets.append(LocalDataset(X, y=y, key=key))
                else:
                    self.local_datasets.append(LocalDataset(X, key=key))

        else:
            raise TypeError(f"local_datasets should be either LocalDataset or Sequence, "
                            f"but got {type(local_datasets)}")

        if primary_train_local_dataset is None:
            # this object the training dataset
            assert primary_train_indices is None
            self.primary_train_local_dataset = self.local_datasets[primary_party_id]
        else:
            # this object the test/val dataset
            assert primary_train_indices is not None
            self.primary_train_local_dataset = primary_train_local_dataset

        self.primary_party_id = primary_party_id
        self.multi_primary = multi_primary
        self.cache_key = cache_key

        if key_cols is not None:
            print("Using TopkSampler")
            self.data_sampler = TopkSampler([local_dataset.key for local_dataset in self.local_datasets], ks=ks,
                                            primary_party_id=primary_party_id, seed=0, multi_primary=multi_primary,
                                            sample_rate_before_topk=sample_rate_before_topk, cache_key=cache_key,
                                            indices=primary_train_indices)
        else:
            print("Using RandomSampler")
            self.data_sampler = RandomSampler([local_dataset.X.shape[0] for local_dataset in self.local_datasets],
                                              n_samples=ks,
                                              primary_party_id=primary_party_id, seed=0)

        self.n_parties = len(local_datasets)
        self.ks = ks
        self.to_tensor_()

        self.use_cache = use_cache
        self.cache_key = cache_key
        self.cache = None

    def __len__(self):
        return len(self.local_datasets[self.primary_party_id])

    def __getitem__(self, idx):
        if self.use_cache:
            if self.cache is None:
                raise ValueError("Cache is not loaded")
            if len(self.cache) <= idx:
                raise IndexError(f"Index {idx} is out of range {len(self.cache)}")
            return self.cache[idx]

        indices_per_party = self.data_sampler.sample(idx)
        if not self.data_sampler.return_multi:
            # single index for secondary parties
            Xs = []
            y_idx = idx
            for pid in range(self.n_parties):
                if pid == self.primary_party_id:
                    key_X, _ = self.primary_train_local_dataset[indices_per_party[pid]]
                else:
                    key_X, _ = self.local_datasets[pid][indices_per_party[pid]]
                Xs.append(key_X)
            y = self.local_datasets[self.primary_party_id].y[y_idx]
            return Xs, y
        else:
            # list of indices for secondary parties
            Xs = []
            for pid in range(self.n_parties):
                index = torch.tensor(indices_per_party[pid])
                local_dataset = self.primary_train_local_dataset \
                    if pid == self.primary_party_id else self.local_datasets[pid]

                if self.local_datasets[pid].key is None:
                    key = np.zeros_like(index).reshape(-1, 1) * np.nan  # for successful collate_fn
                else:
                    key = torch.index_select(local_dataset.key, 0, index)

                X = torch.index_select(local_dataset.X, 0, index)

                if pid == self.primary_party_id:
                    # For test set, the nearest neighbors may not be itself. Remove
                    # the farthest neighbor and add itself. For training set this is
                    # not necessary, but we do it anyway for simplicity.
                    x_self = self.local_datasets[pid].X[idx].unsqueeze(0)
                    X = torch.cat([x_self, X[:-1]], dim=0)
                Xs.append((key, X))
            y = self.local_datasets[self.primary_party_id].y[idx]
            return Xs, y

    # def create_cache(self):
    #     """
    #     Create cache for the dataset by iterating through all data
    #     :return:
    #     """
    #     if not self.use_cache:
    #         return
    #     os.makedirs(os.path.dirname(self.cache_key), exist_ok=True)
    #
    #     self.use_cache = False  # temporarily disable cache, force __getitem__ to calculate data
    #     self.cache = [None] * len(self)
    #     for idx in range(len(self)):
    #         self.cache[idx] = self.__getitem__(idx)
    #     self.use_cache = True
    #
    #     with open(self.cache_path, 'wb') as f:
    #         print(f"Creating cache of {len(self.cache)} records to {self.cache_path}")
    #         pickle.dump(self.cache, f)
    #         print(f"Saved cache to {self.cache_path}")
    #
    # def load_cache(self):
    #     """
    #     Load cache for the dataset
    #     :return:
    #     """
    #     if not self.use_cache:
    #         return
    #
    #     with open(self.cache_path, 'rb') as f:
    #         print(f"Loading cache from {self.cache_path}")
    #         self.cache = pickle.load(f)
    #         print(f"Loaded cache of {len(self.cache)} records from {self.cache_path}")

    @property
    def local_key_channels(self):
        key_channels = []
        for local_dataset in self.local_datasets:
            if local_dataset.key is None:
                key_channels.append(0)
            elif len(local_dataset.key.shape) == 1:
                key_channels.append(1)
            else:
                key_channels.append(local_dataset.key.shape[1])
        return key_channels

    @property
    def local_input_channels(self):
        return [local_dataset.X.shape[1] if len(local_dataset.X.shape) == 2 else 1
                for local_dataset in self.local_datasets]

    @property
    def local_key_X_channels(self):
        X_channels = self.local_input_channels
        key_channels = self.local_key_channels
        return [X + key for X, key in zip(X_channels, key_channels)]

    @classmethod
    def from_csv(cls, paths: Sequence, multi_primary=False, ks=100, key_cols=None, **kwargs):
        """
        Create a VFLRealDataset from csv files
        :param paths: paths to csv files
        :param multi_primary: whether to have multiple primary parties
        :return: a VFLRealDataset
        """
        local_datasets = [LocalDataset.from_csv(path, key_cols=key_cols, **kwargs) for path in paths]
        return cls(local_datasets, multi_primary=multi_primary, ks=ks, key_cols=key_cols, **kwargs)

    @classmethod
    def from_syn_aligned(cls, dataset: VFLSynAlignedDataset, ks=100, key_cols=None, **kwargs):
        """
        Create a VFLRealDataset from a VFLSynAlignedDataset
        :param dataset: a VFLSynAlignedDataset
        :return: a VFLRealDataset
        """
        return cls(dataset.local_datasets, dataset.primary_party_id, ks=ks, key_cols=key_cols, **kwargs)

    @classmethod
    def _from_split_datasets(cls, X_train, X_val, X_test, y_train, y_val, y_test, secondary_datasets,
                             key_train=None, key_val=None, key_test=None, **vfl_args):
        train_local_datasets = [LocalDataset(X_train, y_train, key_train)] + secondary_datasets
        if X_val is not None:
            val_local_datasets = [LocalDataset(X_val, y_val, key_val)] + secondary_datasets
        else:
            val_local_datasets = None
        if X_test is not None:
            test_local_datasets = [LocalDataset(X_test, y_test, key_test)] + secondary_datasets
        else:
            test_local_datasets = None

        # keep only hyperparameters
        vfl_args = {k: v for k, v in vfl_args.items() if v is None or isinstance(v, (int, float, str, bool))}

        # remove training-only arguments
        val_args = {k: v for k, v in vfl_args.items() if (v is None or isinstance(v, (int, float, str, bool))) and
                    k not in ['sample_rate_before_topk']}
        test_args = deepcopy(val_args)

        if 'cache_key' in vfl_args and vfl_args['cache_key'] is not None:
            vfl_args['cache_key'] = vfl_args['cache_key'] + '-train'
        if 'cache_key' in val_args and val_args['cache_key'] is not None:
            val_args['cache_key'] = val_args['cache_key'] + '-val'
        if 'cache_key' in test_args and test_args['cache_key'] is not None:
            test_args['cache_key'] = test_args['cache_key'] + '-test'

        multi_primary = vfl_args.get('multi_primary', False)
        match val_local_datasets, test_local_datasets:
            case (None, None):
                return cls(train_local_datasets, **vfl_args), None, None
            case (None, _):
                train_dataset = cls(train_local_datasets, **vfl_args)
                if multi_primary:
                    test_dataset = cls(test_local_datasets,
                                       primary_train_local_dataset=train_dataset.primary_train_local_dataset,
                                       primary_train_indices=train_dataset.data_sampler.indices,
                                       **test_args)
                else:
                    test_dataset = cls(test_local_datasets, **test_args)
                return train_dataset, None, test_dataset
            case (_, None):
                train_dataset = cls(train_local_datasets, **vfl_args)
                if multi_primary:
                    val_dataset = cls(val_local_datasets,
                                      primary_train_local_dataset=train_dataset.primary_train_local_dataset,
                                      primary_train_indices=train_dataset.data_sampler.indices,
                                      **val_args)
                else:
                    val_dataset = cls(val_local_datasets, **val_args)
                return train_dataset, val_dataset, None
            case (_, _):
                if multi_primary:
                    train_dataset = cls(train_local_datasets, **vfl_args)
                    val_dataset = cls(val_local_datasets,
                                      primary_train_local_dataset=train_dataset.primary_train_local_dataset,
                                      primary_train_indices=train_dataset.data_sampler.indices,
                                      **val_args)
                    test_dataset = cls(test_local_datasets,
                                       primary_train_local_dataset=train_dataset.primary_train_local_dataset,
                                       primary_train_indices=train_dataset.data_sampler.indices,
                                       **test_args)
                else:
                    train_dataset = cls(train_local_datasets, **vfl_args)
                    val_dataset = cls(val_local_datasets, **val_args)
                    test_dataset = cls(test_local_datasets, **test_args)
                return train_dataset, val_dataset, test_dataset

    def to_tensor_(self):
        for local_dataset in self.local_datasets:
            local_dataset.to_tensor_()

    def scale_y_(self, lower=0, upper=1, scaler=None):
        return self.local_datasets[self.primary_party_id].scale_y_(lower=lower, upper=upper, scaler=scaler)

    def normalize_(self, scalers=None, include_key=False):
        """
        Normalize the features
        :param scalers: If scaler is None, normalize *all* parties and return the scalers. Otherwise, only normalize
                        the primary party and use the given scalers.
        :param include_key: whether to normalize the key
        :return:
        """
        if scalers is None:
            scalers = [None] * self.n_parties
            for pid in range(self.n_parties):
                scalers[pid] = self.local_datasets[pid].normalize_(include_key=include_key)
            return scalers
        else:
            if len(scalers) != self.n_parties:
                raise ValueError(f"Length of scalers {len(scalers)} does not match n_parties {self.n_parties}")
            for pid in range(self.n_parties):
                if pid == self.primary_party_id:
                    self.local_datasets[pid].normalize_(scaler=scalers[pid], include_key=include_key)
            return None

    def split_train_test_primary(self, val_ratio=0.1, test_ratio=0.2, random_state=None, shuffle=False):
        """
        Split the dataset into train and test set.
        :param val_ratio: ratio of validation set, if None, no validation set will be generated
        :param test_ratio: ratio of test set, if None, no test set will be generated
        :param random_state: random state, by default None
        :param hard_train_test_split: the split point for hard train-test split, 
            e.g., cifar10 has 50K train data and 10K test data, we set it as 50K, and the input dataset should be the concatenation of [train_data, test_data]. 
            Default 0 means no hard train-test split. 
        :return: three VFLRealDataset, train, val, test
        """
        primary_dataset = self.local_datasets[self.primary_party_id]
        secondary_datasets = (list(self.local_datasets[:self.primary_party_id]) +
                              list(self.local_datasets[self.primary_party_id + 1:]))
        key, X, y = primary_dataset.data

        if y is None:
            raise ValueError(f"y on the primary party {self.primary_party_id} should not be None")

        if key is not None:
            match val_ratio, test_ratio:
                case (None, None):
                    raise ValueError("val_ratio and test_ratio cannot be both None")
                case (None, _):
                    X_train, X_test, y_train, y_test, key_train, key_test = train_test_split(X, y, key,
                                                                                                 test_size=test_ratio,
                                                                                                 random_state=random_state,
                                                                                             shuffle=shuffle)

                    return VFLRealDataset._from_split_datasets(X_train, None, X_test, y_train, None, y_test,
                                                               secondary_datasets,
                                                               key_train, None, key_test, **self.__dict__)
                case (_, None):
                    X_train, X_val, y_train, y_val, key_train, key_val = train_test_split(X, y, key,
                                                                                          test_size=val_ratio,
                                                                                          random_state=random_state,
                                                                                          shuffle=shuffle)
                    return VFLRealDataset._from_split_datasets(X_train, X_val, None, y_train, y_val, None,
                                                               secondary_datasets,
                                                               key_train, key_val, None, **self.__dict__)
                case (_, _):
                    X_train_val, X_test, y_train_val, y_test, key_train_val, key_test = (
                        train_test_split(X, y, key, test_size=test_ratio, random_state=random_state,
                                         shuffle=shuffle))

                    X_train, X_val, y_train, y_val, key_train, key_val = (
                        train_test_split(X_train_val, y_train_val, key_train_val,
                                         test_size=val_ratio / (1 - test_ratio),
                                         random_state=random_state,
                                         shuffle=shuffle))
                    return VFLRealDataset._from_split_datasets(X_train, X_val, X_test, y_train, y_val, y_test,
                                                               secondary_datasets,
                                                               key_train, key_val, key_test, **self.__dict__)
        else:
            # key is None
            match val_ratio, test_ratio:
                case (None, None):
                    raise ValueError("val_ratio and test_ratio cannot be both None")
                case (None, _):
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio,
                                                                        random_state=random_state, shuffle=shuffle)
                    return VFLRealDataset._from_split_datasets(X_train, None, X_test, y_train, None, y_test,
                                                               secondary_datasets,
                                                               **self.__dict__)
                case (_, None):
                    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_ratio,
                                                                      random_state=random_state, shuffle=shuffle)
                    return VFLRealDataset._from_split_datasets(X_train, X_val, None, y_train, y_val, None,
                                                               secondary_datasets,
                                                               **self.__dict__)
                case (_, _):
                    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_ratio,
                                                                                random_state=random_state, shuffle=shuffle)
                    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val,
                                                                      test_size=val_ratio / (1 - test_ratio),
                                                                      random_state=random_state,shuffle=shuffle)
                    return VFLRealDataset._from_split_datasets(X_train, X_val, X_test, y_train, y_val, y_test,
                                                               secondary_datasets,
                                                               **self.__dict__)
