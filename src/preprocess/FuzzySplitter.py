import random
from typing import List, Tuple
import argparse
import os
import os.path
import sys

from sklearn.decomposition import PCA
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

# add src to python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.dataset.LocalDataset import LocalDataset
from src.utils.BasicUtils import PartyPath
from src.preprocess.FeatureSplitter import ImportanceSplitter


def get_fuzzy_key(X: np.ndarray, key_dim: int = 5, train_ratio = 0.7) -> np.ndarray:
    """
    Split the data into multiple parties. The key of each sample is calculated by PCA. Gaussian noise is added to the
    key to make it fuzzy.
    :param X: [np.ndarray] (N x D) data matrix
    :param key_dim: [int] key dimension
    :return: [np.ndarray] (N x key_dim) key matrix
    """
    # PCA

    pca = PCA(n_components=key_dim)
    train_size = int(X.shape[0] * train_ratio)
    pca.fit(X[:train_size])     # fit PCA on the training data to avoid information leakage
    key = pca.transform(X)      # transform the entire data

    # scale to [-1, 1]
    scaler = StandardScaler()
    scaler.fit(key[:train_size])
    key = scaler.transform(key)

    return key


def get_noise_key(n_instances, key_dim, noise_scale):
    """
    Generate a key matrix with Gaussian noise.
    :param n_instances: [int] number of instances
    :param key_dim: [int] key dimension
    :param noise_scale: [float] scale of the Gaussian noise
    :return: [np.ndarray] (N x key_dim) key matrix
    """
    return np.random.normal(0, noise_scale, (n_instances, key_dim))


def fuzzy_split(X: np.ndarray, n_parties: int, key_dim: int = 5, noise_scale: float = 0.0,
                key_base_noise: float = 1.0) -> List[np.ndarray]:
    """
    Split the data into multiple parties. The key of each sample is calculated by PCA. Gaussian noise is added to the
    key to make it fuzzy.
    :param X: [np.ndarray] (N x D) data matrix
    :param key_dim: [int] key dimension
    :param n_parties: [int] number of parties
    :param key_base_noise: [float] scale of the base noise
    :param noise_scale: [float] scale of the Gaussian noise
    :return: [List[np.ndarray]] list of key matrices for each party
    """

    raw_Xs = np.array_split(X, n_parties, axis=1)  # split the data into multiple parties

    key = get_fuzzy_key(raw_Xs[0], key_dim)     # first party as primary party to generate key
    # key = get_noise_key(X.shape[0], key_dim, key_base_noise)

    Xs = []
    for i in range(n_parties):
        key_i = key + np.random.normal(0, noise_scale, key.shape)
        Xi = np.concatenate((key_i, raw_Xs[i]), axis=1)
        Xs.append(Xi)

    return Xs


def load_data(data_path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load data from a file.
    :param data_path: [str] path to the data file
    :return: [Tuple[np.ndarray, np.ndarray]] data matrix and label vector
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"{data_path} not found.")
    if data_path.endswith('.libsvm'):
        X, y = load_svmlight_file(data_path)
        X = X.toarray()
    elif data_path.endswith('.csv'):
        X = pd.read_csv(data_path).values
        y = X[:, -1]
        X = X[:, :-1]
    else:
        raise NotImplementedError(f"Unknown file format {data_path}")
    return X, y


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fuzzy Splitter')
    parser.add_argument('-d', '--dataset', type=str, default='gisette.libsvm', help='path to the data file')
    parser.add_argument('-p', '--n_parties', type=int, default=2, help='number of parties')
    parser.add_argument('-kd', '--key_dim', type=int, default=4, help='key dimension')
    parser.add_argument('-ns', '--noise_scale', type=float, default=0.0, help='scale of the Gaussian noise')
    # parser.add_argument('-sd', '--save-dir', type=str, default='data/syn/multi_party_dataset/gisette', help='directory to save the split data')
    parser.add_argument('-a', '--alpha', type=float, default=None, help='weight of the importance score')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    args = parser.parse_args()

    if len(args.dataset.split('.')) < 2:
        raise ValueError(f"Invalid dataset name {args.dataset}, should be in the format of 'dataset.format'")
    fmt = args.dataset.split('.')[-1]
    dataset = args.dataset.split('.')[0]
    X, y = load_data(f"data/syn/{dataset}/{dataset}.{fmt}")

    # random shuffle X, y
    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.alpha is None:
        idx = np.random.permutation(X.shape[0])
        X = X[idx]
        y = y[idx]
        np.random.shuffle(X.T)  # shuffle the data along the feature dimension
        Xs = fuzzy_split(X, args.n_parties, key_dim=args.key_dim, noise_scale=args.noise_scale)
    else:
        splitter = ImportanceSplitter(num_parties=args.n_parties, weights=args.alpha, seed=args.seed)
        Xs_no_key = splitter.split(X)

        # add noise
        key = get_fuzzy_key(X, key_dim=args.key_dim)
        Xs = []
        for i in range(args.n_parties):
            key_i = key + np.random.normal(0, args.noise_scale, key.shape)
            Xs.append(np.concatenate((key_i, Xs_no_key[i]), axis=1))

    save_dir = f"data/syn/{dataset}/noise{args.noise_scale}"
    os.makedirs(save_dir, exist_ok=True)
    for i, Xi in enumerate(Xs):
        weight = '100.0' if args.alpha is None else f"{args.alpha:.1f}"     # 100 for balanced split (the real alpha may not be 100, but a large number)
        party_path = os.path.join(save_dir, f"{dataset}_party{args.n_parties}-{i}_imp_weight{weight}_seed{args.seed}.pkl")
        assert Xi[:, :args.key_dim].shape[1] > 0
        assert Xi[:, args.key_dim:].shape[1] > 0
        LocalDataset(Xi[:, args.key_dim:], y, key=Xi[:, :args.key_dim]).to_pickle(party_path)
        print(f"Saved {party_path}")
