import abc
from typing import Protocol
import pickle

import numpy as np
import pandas
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset


class LocalDataset(Dataset):
    """
    Base class for local datasets
    """

    def __init__(self, X, y=None, key=None, **kwargs):
        """
        Required parameters:
        :param X: features (array)

        Optional parameters:
        :param key: key of the ID (array)
        :param y: labels (1d array)
        """
        self.key = key
        if isinstance(X, np.ndarray):
            self.X = X.astype(np.float32)
        elif isinstance(X, torch.Tensor):
            self.X = X.float()
        else:
            raise TypeError(f"X should be either np.ndarray or torch.Tensor, but got {type(X)}")

        if y is None:
            self.y = None
        elif isinstance(y, np.ndarray):
            self.y = y.astype(np.float32)
        elif isinstance(y, torch.Tensor):
            self.y = y.float() if y is not None else None
        else:
            raise TypeError("y should be either np.ndarray or torch.Tensor")

        self.check_shape()

        # if key is not provided, set key as nan to avoid collate_fn error
        if self.key is None:
            self.key = np.arange(self.X.shape[0]).reshape(-1, 1)
            # self.key = np.full((self.X.shape[0], 1), np.nan)

    def __add__(self, other):
        """
        Concatenate two LocalDataset
        """
        if not isinstance(other, LocalDataset):
            raise TypeError(f"other should be LocalDataset, but got {type(other)}")

        if self.X.shape[1:] != other.X.shape[1:]:
            raise ValueError(f"self.X.shape[1:] != other.X.shape[1:]")

        X = np.concatenate([self.X, other.X], axis=0)
        if self.y is None and other.y is None:
            y = None
        else:
            if self.y is None and other.y is not None:
                raise ValueError(f"self.y is None, but other.y is not None")
            if self.y is not None and other.y is None:
                raise ValueError(f"self.y is not None, but other.y is None")
            if self.y is not None and other.y is not None:
                if self.y.shape[1:] != other.y.shape[1:]:
                    raise ValueError(f"self.y.shape[1:] != other.y.shape[1:]")
            y = np.concatenate([self.y, other.y], axis=0) if self.y is not None else None
        if self.key is None and other.key is None:
            key = None
        else:
            if self.key.shape[1:] != other.key.shape[1:]:
                raise ValueError(f"self.key.shape[1:] != other.key.shape[1:]")
            key = np.concatenate([self.key, other.key], axis=0)
        return LocalDataset(X, y, key)

    @torch.no_grad()
    def check_shape(self):
        if self.y is not None:
            assert self.X.shape[0] == self.y.shape[0], "The number of samples in X and y should be the same"
        if self.key is not None:
            assert self.X.shape[0] == self.key.shape[0], "The number of samples in X and key should be the same"

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        """
        :param idx: the index of the item
        :return: key[idx], X[idx], y[idx]
        """
        X = self.X[idx]
        key = self.key[idx] if self.key is not None else None
        y = self.y[idx] if self.y is not None else None
        return (key, X), y

    @property
    def data(self):
        return self.key, self.X, self.y

    @property
    def key_X_dim(self):
        if self.key is None:
            return self.X.shape[1]
        else:
            return self.X.shape[1] + self.key.shape[1]

    @classmethod
    def from_csv(cls, csv_path, header=None, key_cols=1, **kwargs):
        """
        Load dataset from csv file. The key_cols columns are keys, the last column is the label, and the rest
        columns are features.
        :param csv_path: path to csv file
        :param header: row number(s) to use as the column names, and the start of the data.
                       Same as the header in pandas.read_csv()
        :param key_cols: Int. Number of key columns. | key1 | key2 | key.. | keyN | X1 | X2 | X3.. | Xn | y |
        """
        df = pd.read_csv(csv_path, header=header)
        if key_cols is None:
            key = None
            X = df.iloc[:, :-1].values
        else:
            assert df.shape[1] > key_cols + 1, "The number of columns should be larger than key_cols + 1"
            key = df.iloc[:, :key_cols].values
            X = df.iloc[:, key_cols:-1].values
        y = df.iloc[:, -1].values
        return cls(X, y, key, **kwargs)

    @classmethod
    def from_pickle(cls, pickle_path):
        with open(pickle_path, 'rb') as f:
            return pickle.load(f)
        
    def to_pickle(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def to_csv(self, path, type='raw'):
        # flatten >=2 dimensional X (e.g. image) to 1 dimensional
        if len(self.X.shape) > 2:
            X = self.X.reshape(self.X.shape[0], -1)
        else:
            X = self.X

        assert type in ['raw', 'fedtree', 'fedtrans'], "type should be in ['raw', 'fedtree', 'fedtrans']"
        if type == 'raw':
            df = pd.DataFrame(np.concatenate([X, self.y.reshape(-1, 1)], axis=1))
            df.to_csv(path, header=False, index=False)
        if type == 'fedtrans':
            y = self.y
            if y is None:
                # create dummy y
                y = np.array([None for i in range(X.shape[0])])
            df = pd.DataFrame(np.concatenate([self.key, X, y.reshape(-1, 1)], axis=1))
            for i in range(self.key.shape[1]):
                df.rename(columns={i: f'key{i}'}, inplace=True)
            for i in range(X.shape[1]):
                df.rename(columns={i + self.key.shape[1]: f'x{i}'}, inplace=True)
            df.rename(columns={ df.shape[1] - 1: 'y'}, inplace=True)
            df.to_csv(path, header=True, index=False) # You have to have a header to be able to read it back in. otherwise we don't know if there is a y column or not
        elif type == 'fedtree':
            if self.key is None:
                raise ValueError("key is None. FedTree requires key column.")
            if len(self.key.shape) != 1 and self.key.shape[1] != 1:
                raise ValueError("FedTree does not support multi-dimensional key.")
            if self.y is None:
                columns = ['id'] + [f'x{i}' for i in range(X.shape[1])]
                df = pd.DataFrame(np.concatenate([self.key.reshape(-1, 1), X], axis=1), columns=columns)
            else:
                columns = ['id', 'y'] + [f'x{i}' for i in range(X.shape[1])]
                df = pd.DataFrame(np.concatenate([self.key.reshape(-1, 1), self.y.reshape(-1, 1), X], axis=1),
                                  columns=columns)
            df.to_csv(path, index=False)
        else:
            raise NotImplementedError(f"CSV type {type} is not implemented.")

    def to_tensor_(self):
        """
        Convert X, y, key to torch.Tensor
        """
        if isinstance(self.X, np.ndarray):
            self.X = torch.from_numpy(self.X).float()
        if isinstance(self.y, np.ndarray):
            self.y = torch.from_numpy(self.y).float()
        if isinstance(self.key, np.ndarray):
            self.key = torch.from_numpy(self.key).float()

    def scale_y_(self, lower=0, upper=1, scaler=None):
        """
        Scale the label to [lower, upper]
        """
        if self.y is None:
            return None

        if scaler is None:
            scaler = MinMaxScaler(feature_range=(lower, upper))
            self.y = scaler.fit_transform(self.y.reshape(-1, 1)).reshape(-1)
            return scaler
        else:
            self.y = scaler.transform(self.y.reshape(-1, 1)).reshape(-1)
            return None

    def normalize_(self, scaler=None, include_key=False):
        """
        Normalize the features
        """
        if scaler is None:
            scaler = StandardScaler()
            if include_key:
                key_X = np.concatenate([self.key, self.X], axis=1)
                key_X = scaler.fit_transform(key_X)
                self.key = key_X[:, :self.key.shape[1]]
                self.X = key_X[:, self.key.shape[1]:]
            else:
                self.X = scaler.fit_transform(self.X)
            return scaler
        else:
            if include_key:
                key_X = np.concatenate([self.key, self.X], axis=1)
                key_X = scaler.transform(key_X)
                self.key = key_X[:, :self.key.shape[1]]
                self.X = key_X[:, self.key.shape[1]:]
            else:
                self.X = scaler.transform(self.X)
            return None

    def split_train_test(self, val_ratio=0.1, test_ratio=0.2, random_state=None, shuffle=False):
        """
        Split the dataset into train and test set.
        :param val_ratio: ratio of validation set, if None, no validation set will be generated
        :param test_ratio: ratio of test set, if None, no test set will be generated
        :param random_state: random state, by default None
        :param hard_train_test_split: the split point for hard train-test split, 
            e.g., cifar10 has 50K train data and 10K test data, we set it as 50K, and the input dataset should be the concatenation of [train_data, test_data]. 
            Default 0 means no hard train-test split. 
        :return: three LocalDataset, train, val, test
        """
        key, X, y = self.data

        if y is None:
            raise ValueError(f"y should not be None")

        def train_test_split_ignore_none(X, y, key, test_size, random_state, shuffle):
            if key is None:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                                    random_state=random_state, shuffle=shuffle)
                return X_train, X_test, y_train, y_test, None, None
            else:
                X_train, X_test, y_train, y_test, key_train, key_test = train_test_split(X, y, key, test_size=test_size,
                                                                                         random_state=random_state, shuffle=shuffle)
                return X_train, X_test, y_train, y_test, key_train, key_test

        match val_ratio, test_ratio:
            case (None, None):
                raise ValueError("val_ratio and test_ratio cannot be both None")
            case (None, _):
                X_train, X_test, y_train, y_test, key_train, key_test = train_test_split_ignore_none(X, y, key,
                                                                                            test_size=test_ratio,
                                                                                            random_state=random_state,
                                                                                            shuffle=shuffle)
                return [LocalDataset(X_train, y_train, key_train),
                        None,
                        LocalDataset(X_test, y_test, key_test)]
            case (_, None):
                X_train, X_val, y_train, y_val, key_train, key_val = train_test_split_ignore_none(X, y, key,
                                                                                      test_size=val_ratio,
                                                                                      random_state=random_state,
                                                                                      shuffle=shuffle)
                return [LocalDataset(X_train, y_train, key_train),
                        LocalDataset(X_val, y_val, key_val),
                        None]
            case (_, _):
                X_train_val, X_test, y_train_val, y_test, key_train_val, key_test = (
                        train_test_split_ignore_none(X, y, key, test_size=test_ratio, random_state=random_state,
                                                     shuffle=shuffle))

                X_train, X_val, y_train, y_val, key_train, key_val = (
                    train_test_split_ignore_none(X_train_val, y_train_val, key_train_val, test_size=val_ratio / (1 - test_ratio),
                                     random_state=random_state, shuffle=shuffle))

                return [LocalDataset(X_train, y_train, key_train),
                        LocalDataset(X_val, y_val, key_val),
                        LocalDataset(X_test, y_test, key_test)]


