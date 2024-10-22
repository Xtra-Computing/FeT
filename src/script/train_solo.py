import os
import sys
from typing import Callable
import argparse
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18
# from torchsummaryX import summary

import pandas as pd
from tqdm import tqdm

# add src to python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from model.Solo import MLP
from dataset.LocalDataset import LocalDataset
from src.utils import get_device_from_gpu_id, get_metric_from_str, PartyPath
from src.utils import get_metric_positive_from_str
from train.Fit import fit
from dataset.VFLRealDataset import VFLRealDataset
from src.preprocess.nytaxi.ny_loader import NYBikeTaxiLoader
from src.preprocess.hdb.hdb_loader import load_both as load_both_hdb
from src.preprocess.ml_dataset.two_party_loader import TwoPartyLoader as FedSimSynLoader
from src.model.SoloTransformer import SoloTrans

if __name__ == '__main__':
    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help="GPU ID. Set to None if you want to use CPU")

    # parameters for dataset
    parser.add_argument('--dataset', '-d', type=str,
                        help="dataset to use.")
    parser.add_argument('--n_parties', '-p', type=int, default=4,
                        help="number of parties. Should be >=2")
    parser.add_argument('--primary_party', '-pp', type=int, default=0,
                        help="primary party. Should be in [0, n_parties-1]")
    parser.add_argument('--splitter', '-sp', type=str, default='imp')
    parser.add_argument('--weights', '-w', type=float, default=1, help="weights for the ImportanceSplitter")
    parser.add_argument('--beta', '-b', type=float, default=1, help="beta for the CorrelationSplitter")
    parser.add_argument('--key-noise', type=float, default=0.0, help='key noise in FedSim synthetic data')

    # # parameters for model
    # parser.add_argument('--epochs', '-e', type=int, default=50)
    # parser.add_argument('--lr', '-lr', type=float, default=1e-3)
    # parser.add_argument('--weight_decay', '-wd', type=float, default=1e-5)
    # parser.add_argument('--batch_size', '-bs', type=int, default=128)
    # parser.add_argument('--n_classes', '-c', type=int, default=7,
    #                     help="number of classes. 1 for regression, 2 for binary classification,"
    #                          ">=3 for multi-class classification")
    # parser.add_argument('--metric', '-m', type=str, default='acc',
    #                     help="metric to evaluate the model. Supported metrics: [accuracy, rmse]")
    # parser.add_argument('--result-path', '-rp', type=str, default=None,
    #                     help="path to save the result")
    # parser.add_argument('--seed', '-s', type=int, default=0, help="random seed")
    # parser.add_argument('--key-noise', type=float, default=0.0, help="key noise scale")

    # parameters for model
    parser.add_argument('--epochs', '-e', type=int, default=100)
    parser.add_argument('--lr', '-lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-5)
    parser.add_argument('--batch_size', '-bs', type=int, default=128)
    parser.add_argument('--n_classes', '-c', type=int, default=1,
                        help="number of classes. 1 for regression, 2 for binary classification,"
                             ">=3 for multi-class classification")
    parser.add_argument('--metric', '-m', type=str, default='acc',
                        help="metric to evaluate the model. Supported metrics: [accuracy, rmse]")
    parser.add_argument('--result-path', '-rp', type=str, default=None,
                        help="path to save the result")
    parser.add_argument('--seed', '-s', type=int, default=0, help="random seed")
    parser.add_argument('--log-dir', '-ld', type=str, default='log', help='log directory')
    parser.add_argument('--data-embed-dim', '-ded', type=int, default=200, help='data embedding dimension')
    parser.add_argument('--key-embed-dim', '-ked', type=int, default=200, help='key embedding dimension')
    parser.add_argument('--num-heads', '-nh', type=int, default=8, help='number of heads in multi-head attention')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--n-local-blocks', '-nlb', type=int, default=6, help='number of local blocks')
    parser.add_argument('--n-agg-blocks', '-nab', type=int, default=6, help='number of aggregation blocks')
    parser.add_argument('--knn-k', type=int, default=50, help='k for knn')
    parser.add_argument('-v', '--version', type=int, default=3, help='version of the model')

    args = parser.parse_args()

    # print hostname
    print(f"Hostname: {os.uname().nodename}")

    syn_root = "data/syn/"
    real_root = "data/fedsim-data/"
    if args.dataset == 'house':
        house_root = f"{real_root}/beijing/"
        house_dataset = LocalDataset.from_csv(os.path.join(house_root, "house_clean.csv"), header=1, key_cols=2)
        train_dataset, val_dataset, test_dataset = house_dataset.split_train_test(
            val_ratio=0.1, test_ratio=0.2, random_state=args.seed)
    elif args.dataset == 'taxi':
        key_dim = 4
        taxi_root = f"{real_root}/nytaxi/"
        bike_path = "bike_201606_clean_sample_2e5.pkl"
        taxi_path = "taxi_201606_clean_sample_1e5.pkl"
        base_loader = NYBikeTaxiLoader(bike_path=os.path.join(taxi_root, bike_path),
                                       taxi_path=os.path.join(taxi_root, taxi_path), link=True)
        [X1, X2], y = base_loader.load_parties()

        taxi_dataset = LocalDataset(X1, y, key=None)
        train_dataset, val_dataset, test_dataset = taxi_dataset.split_train_test(
            val_ratio=0.1, test_ratio=0.2, random_state=args.seed)
    elif args.dataset == 'hdb':
        key_dim = 2
        hdb_path = f"{real_root}/hdb/hdb_clean.csv"
        school_path = f"{real_root}/hdb/school_clean.csv"
        [X1, X2], y = load_both_hdb(hdb_path, school_path, active_party='hdb')
        hdb_dataset = LocalDataset(X1, y, key=None)
        train_dataset, val_dataset, test_dataset = hdb_dataset.split_train_test(
            val_ratio=0.1, test_ratio=0.2, random_state=args.seed)

    elif args.dataset in ("gisette", "mnist"):
        # multi_party_dataset for scalability
        key_dim = 4
        data_path = PartyPath(dataset_path=args.dataset, n_parties=args.n_parties, party_id=args.primary_party,
                              splitter=args.splitter, weight=args.weights, beta=args.beta, seed=0,
                              fmt='pkl').data(None) # use seed=0 to get the same data for all parties
        syn_dataset_dir = f"data/syn/{args.dataset}/noise{args.key_noise}/"
        solo_dataset = LocalDataset.from_pickle(
            os.path.join(syn_dataset_dir, data_path)
        )

        train_dataset, val_dataset, test_dataset = solo_dataset.split_train_test(
            val_ratio=0.1, test_ratio=0.2, random_state=args.seed)

    else:
        # Note: torch.compile() in torch 2.0 significantly harms the accuracy with little speed up
        data_path = PartyPath(dataset_path=args.dataset, n_parties=args.n_parties, party_id=args.primary_party,
                              splitter=args.splitter, weight=args.weights, beta=args.beta, seed=args.seed,
                              fmt='pkl').data(None)
        solo_dataset = LocalDataset.from_pickle(os.path.join(syn_root, args.dataset, data_path))

        train_dataset, val_dataset, test_dataset = solo_dataset.split_train_test(
            val_ratio=0.1, test_ratio=0.2, random_state=args.seed)

    X_scaler = train_dataset.normalize_()
    if val_dataset is not None:
        val_dataset.normalize_(scaler=X_scaler)
    test_dataset.normalize_(scaler=X_scaler)

    # create the model
    scaler = None
    if args.n_classes == 1:  # regression
        task = 'reg'
        loss_fn = nn.MSELoss()
        out_dim = 1
        out_activation = nn.Sigmoid()
        if args.metric == 'acc':  # if metric is accuracy, change it to rmse
            args.metric = 'rmse'
            warnings.warn("Metric is changed to rmse for regression task")
        # scale the labels to [0, 1]
        scaler = train_dataset.scale_y_()
        if val_dataset is not None:
            val_dataset.scale_y_(scaler=scaler)
        test_dataset.scale_y_(scaler=scaler)
    elif args.n_classes == 2:  # binary classification
        task = 'bin-cls'
        loss_fn = nn.BCELoss()
        out_dim = 1
        out_activation = nn.Sigmoid()
        # make sure the labels are in [0, 1]
        train_dataset.scale_y_()
        if val_dataset is not None:
            val_dataset.scale_y_()
        test_dataset.scale_y_()
    else:  # multi-class classification
        task = 'multi-cls'
        loss_fn = nn.CrossEntropyLoss()
        out_dim = args.n_classes
        out_activation = None  # No need for softmax since it is included in CrossEntropyLoss

    # use SplitSum
    model = MLP(train_dataset.key_X_dim, [400, 400], out_dim, activation=out_activation)
    # model = SoloTrans(data_dim=train_dataset.X.shape[1],
    #                   out_dim=out_dim, data_embed_dim=args.data_embed_dim,
    #                   key_embed_dim=args.key_embed_dim,
    #                   num_heads=args.num_heads, dropout=args.dropout,
    #                   # n_embeddings=len(train_dataset) + len(test_dataset),
    #                   n_embeddings=None, out_activation=out_activation,
    #                   n_local_blocks=args.n_local_blocks, n_agg_blocks=args.n_agg_blocks)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5, verbose=True)

    train_dataset.to_tensor_()
    test_dataset.to_tensor_()
    if val_dataset is not None:
        val_dataset.to_tensor_()
    n_workers = 0
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=n_workers)
    if val_dataset is not None:
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=n_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=n_workers)

    metric_fn = get_metric_from_str(args.metric)
    metric_positive = get_metric_positive_from_str(args.metric)

    test_loss_list, test_score_list = fit(model, optimizer, loss_fn, metric_fn, train_loader, epochs=args.epochs,
                                          gpu_id=args.gpu,
                                          n_classes=args.n_classes, test_loader=test_loader, task=task,
                                          scheduler=scheduler, has_key=True,
                                          val_loader=val_loader, metric_positive=metric_positive, y_scaler=scaler,
                                          solo=True)

    if args.result_path is not None:
        # save test loss and score to a two-column csv file, each row is for one epoch (with pandas)
        test_result = pd.DataFrame({'loss': test_loss_list, 'score': test_score_list})
        test_result.to_csv(args.result_path, index=False)

    print("Done!")
