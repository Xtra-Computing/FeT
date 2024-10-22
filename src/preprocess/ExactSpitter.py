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


def get_exact_key(n, key_dim=1) -> np.ndarray:
    """
    Generate a (nx1) key matrix with exact values. Each value is in [-1, 1].
    """
    return np.random.uniform(-1, 1, (n, key_dim))



def exact_split(X: np.ndarray, n_parties: int, key_dim: int = 1) -> List[np.ndarray]:
    """
    Split the data into multiple parties.
    Each party has the same key matrix.
    """

    raw_Xs = np.array_split(X, n_parties, axis=1)  # split the data into multiple parties

    key = get_exact_key(len(raw_Xs[0]), key_dim)

    Xs = []
    for i in range(n_parties):
        Xi = np.concatenate((key, raw_Xs[i]), axis=1)
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
    parser = argparse.ArgumentParser(description='Equal Splitter with exact key')
    parser.add_argument('-d', '--dataset', type=str, default='gisette.libsvm', help='path to the data file')
    parser.add_argument('-p', '--n_parties', type=int, default=2, help='number of parties')
    parser.add_argument('-kd', '--key_dim', type=int, default=1, help='key dimension')
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

    idx = np.random.permutation(X.shape[0])
    X = X[idx]
    y = y[idx]
    np.random.shuffle(X.T)  # shuffle the data along the feature dimension

    Xs = exact_split(X, args.n_parties, key_dim=args.key_dim)
    save_dir = f"data/syn/exact_key_dataset/{dataset}/"
    os.makedirs(save_dir, exist_ok=True)
    for i, Xi in enumerate(Xs):
        party_path = os.path.join(save_dir, f"{dataset}_party{args.n_parties}-{i}_imp_weight100.0_seed{args.seed}.pkl")
        LocalDataset(Xi[:, args.key_dim:], y, key=Xi[:, :args.key_dim]).to_pickle(party_path)
        print(f"Saved {party_path}")
