import wget
import os
import vertibench
import pandas as pd
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import MinMaxScaler

from vertibench.Splitter import ImportanceSplitter

os.path.join(os.path.dirname(__file__), '..')
os.path.join(os.path.dirname(__file__), '..', '..')

from src.dataset.LocalDataset import LocalDataset
from src.dataset.VFLRealDataset import VFLRealDataset
from src.utils.BasicUtils import PartyPath

# check if the data is already downloaded
syn_root = "data/syn/"
dataset_paths = {
# 'covtype': 'covtype.libsvm',
# 'gisette': 'gisette.libsvm',
# 'letter': 'letter.libsvm',
# 'radar': 'radar.csv',
    'realsim': 'realsim.libsvm',
}

for dataset, filename in dataset_paths.items():
    data_path = os.path.join(syn_root, dataset, filename)
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"{data_path} not found. Please download the data first.")

    if filename.endswith('.libsvm'):
        X, y = load_svmlight_file(data_path)
        X = X.toarray()
        if dataset in ['gisette']:
            # gisette is a binary classification dataset with labels in {-1, 1}
            scaler = MinMaxScaler((0, 1))
            y = scaler.fit_transform(y.reshape(-1, 1)).flatten()
    elif filename.endswith('.csv'):
        df = pd.read_csv(data_path)
        X = df.values[:, :-1]
        y = df.values[:, -1]
    else:
        raise NotImplementedError(f"Unknown file format {filename}")

    # split the data
    for n_parties in [2, 4, 8]:
        for alpha in [0.1, 1, 10, 100]:
            splitter = ImportanceSplitter(num_parties=n_parties, weights=alpha, seed=0)
            Xs = splitter.split(X)

            # save the data
            for i in range(n_parties):
                party_path = PartyPath(dataset_path=dataset, n_parties=n_parties, party_id=i,
                                       splitter='imp', weight=alpha, beta=0, seed=0, fmt='pkl').data(None)
                party_full_path = os.path.join(syn_root, dataset, party_path)
                LocalDataset(Xs[i], y).to_pickle(party_full_path)
                print(f"Saved {party_full_path}")

