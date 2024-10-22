import os
import sys
import numpy as np
import random
from sklearn.datasets import load_svmlight_file
import pickle
import inspect
from scipy.sparse import csr_matrix


class TwoPartyLoader:
    def __init__(self, num_features, num_common_features: int,
                 common_feature_noise_scale=0.0, data_fmt='libsvm', dataset_name=None, cache_path=None,
                 n_classes=2, seed=0):
        """
        :param cache_path: path for cache of the object
        :param dataset_name: name of the dataset
        :param num_features_per_party: number of features on both party, including common features
        :param num_common_features: number of common features
        """
        self.cache_path = cache_path
        self.dataset_name = dataset_name
        self.data_fmt = data_fmt
        self.n_classes = n_classes
        self.common_feature_noise_scale = common_feature_noise_scale
        self.num_common_features = num_common_features
        self.num_features = num_features
        self.seeds = list(range(seed, seed + 3))

        self.X = None
        self.y = None
        self.Xs = None

    def load_dataset(self, path=None, use_cache=True, scale_label=False):
        """
        :param use_cache: whether to use cache
        :param path: path of the ml dataset
        :param scale_label: whether to scale back the label from [0,1] to int. True in covtype.scale01.
        :return: features, labels
        """
        if use_cache and self.X is not None and self.y is not None:
            assert self.num_features == self.X.shape[1], "Total number of features mismatch."
            return self.X, self.y

        assert path is not None
        print("Loading {} dataset".format(self.dataset_name))
        if inspect.isfunction(self.data_fmt):
            X, y = self.data_fmt(path)
        elif self.data_fmt == 'libsvm':
            X, y = load_svmlight_file(path)
            X = X.toarray()

            # hard code for a strange dataset whose labels are 1 & 2
            if self.dataset_name == 'covtype.binary':
                y -= 1
        elif self.data_fmt == 'csv':
            dataset = np.loadtxt(path, delimiter=',', skiprows=1)
            X = dataset[:, :-1]
            y = dataset[:, -1].reshape(-1)
        else:
            assert False, "Unsupported ML dataset format"

        if scale_label:
            y = np.rint(y * (self.n_classes - 1)).astype('int32')

        assert self.num_features == X.shape[1], "Total number of features mismatch."
        print("Done")
        if use_cache:
            self.X, self.y = X, y

        return X, y

    def load_parties(self, path=None, use_cache=True, scale_label=False):
        X, y = self.load_dataset(path, use_cache, scale_label)
        if use_cache and self.Xs is not None:
            print("Loading parties from cache")
            return self.Xs, self.y

        # assuming all the features are useful
        print("Splitting features to two parties")

        # randomly divide trained features to two parties
        shuffle_state = np.random.RandomState(self.seeds[0])
        shuffle_state.shuffle(X.T)  # shuffle columns
        trained_features = X[:, self.num_common_features:]
        trained_features1 = trained_features[:, :trained_features.shape[1] // 2]
        trained_features2 = trained_features[:, trained_features.shape[1] // 2:]

        # append common features
        common_features = X[:, :self.num_common_features]
        noise_state = np.random.RandomState(self.seeds[2])
        noised_common_features = common_features.copy() + noise_state.normal(
            scale=self.common_feature_noise_scale, size=common_features.shape)
        X1 = np.concatenate([trained_features1, common_features], axis=1)
        X2 = np.concatenate([noised_common_features, trained_features2], axis=1)

        assert X1.shape[1] + X2.shape[1] - self.num_common_features == self.X.shape[1]

        if use_cache:
            # refresh cached Xs
            self.Xs = [X1, X2]
        print("Done")
        return [X1, X2], y

    def to_pickle(self, save_path: str):
        with open(save_path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def from_pickle(load_path: str):
        with open(load_path, 'rb') as f:
            return pickle.load(f)


class ThreePartyLoader:
    def __init__(self, num_features, num_common_features: int,
                 common_feature_noise_scale=0.0, data_fmt='libsvm', dataset_name=None, cache_path=None,
                 n_classes=2, seed=0):
        """
                :param cache_path: path for cache of the object
                :param dataset_name: name of the dataset
                :param num_features_per_party: number of features on both party, including common features
                :param num_common_features: number of common features
                """
        self.cache_path = cache_path
        self.dataset_name = dataset_name
        self.data_fmt = data_fmt
        self.n_classes = n_classes
        self.common_feature_noise_scale = common_feature_noise_scale
        self.num_common_features = num_common_features
        self.num_features = num_features
        self.seeds = list(range(seed, seed + 3))

        self.X = None
        self.y = None
        self.Xs = None

    def load_dataset(self, path=None, use_cache=True, scale_label=False):
        """
        :param use_cache: whether to use cache
        :param path: path of the ml dataset
        :param scale_label: whether to scale back the label from [0,1] to int. True in covtype.scale01.
        :return: features, labels
        """
        if use_cache and self.X is not None and self.y is not None:
            assert self.num_features == self.X.shape[1], "Total number of features mismatch."
            return self.X, self.y

        assert path is not None
        print("Loading {} dataset".format(self.dataset_name))
        if inspect.isfunction(self.data_fmt):
            X, y = self.data_fmt(path)
        elif self.data_fmt == 'libsvm':
            X, y = load_svmlight_file(path)
            X = X.toarray()

            # hard code for a strange dataset whose labels are 1 & 2
            if self.dataset_name == 'covtype.binary':
                y -= 1
        elif self.data_fmt == 'csv':
            dataset = np.loadtxt(path, delimiter=',', skiprows=1)
            X = dataset[:, :-1]
            y = dataset[:, -1].reshape(-1)
        else:
            assert False, "Unsupported ML dataset format"

        if scale_label:
            y = np.rint(y * (self.n_classes - 1)).astype('int32')

        assert self.num_features == X.shape[1], "Total number of features mismatch."
        print("Done")
        if use_cache:
            self.X, self.y = X, y

        return X, y

    def load_parties(self, path=None, use_cache=True, scale_label=False):
        X, y = self.load_dataset(path, use_cache, scale_label)
        if use_cache and self.Xs is not None:
            print("Loading parties from cache")
            return self.Xs, self.y

        # assuming all the features are useful
        print("Splitting features to two parties")

        # randomly divide trained features to three parties
        shuffle_state = np.random.RandomState(self.seeds[0])
        shuffle_state.shuffle(X.T)  # shuffle columns
        trained_features = X[:, self.num_common_features:]
        trained_features1, trained_features2, trained_features3 = np.split(trained_features, 3, axis=1)

        # append common features
        common_features = X[:, :self.num_common_features]
        noise_state1 = np.random.RandomState(self.seeds[1])
        noise_state2 = np.random.RandomState(self.seeds[2])
        noised_common_features1 = common_features.copy() + noise_state1.normal(
            scale=self.common_feature_noise_scale, size=common_features.shape)
        noised_common_features2 = common_features.copy() + noise_state2.normal(
            scale=self.common_feature_noise_scale, size=common_features.shape)
        X1 = np.concatenate([trained_features1, common_features], axis=1)
        X2 = np.concatenate([noised_common_features1, trained_features2, noised_common_features2], axis=1)
        X3 = np.concatenate([noised_common_features2, trained_features3], axis=1)

        if use_cache:
            # refresh cached Xs
            self.Xs = [X1, X2, X3]
        print("Done")
        return [X1, X2, X3], y

    def to_pickle(self, save_path: str):
        with open(save_path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def from_pickle(load_path: str):
        with open(load_path, 'rb') as f:
            return pickle.load(f)
