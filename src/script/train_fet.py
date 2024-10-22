import argparse
import os
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import torch.multiprocessing
import torch.nn as nn
import torch_optimizer as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# add src to python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.dataset.VFLDataset import VFLSynAlignedDataset
from src.dataset.VFLRealDataset import VFLRealDataset
from src.preprocess.hdb.hdb_loader import load_both as load_both_hdb
from src.preprocess.ml_dataset.two_party_loader import TwoPartyLoader as FedSimSynLoader
from src.preprocess.nytaxi.ny_loader import NYBikeTaxiLoader
from src.train.Fit import fit
from src.utils.BasicUtils import (PartyPath, get_metric_from_str, get_metric_positive_from_str)
from src.utils.logger import CommLogger
from src.model.FeT import FeT

# Avoid "Too many open files" error
torch.multiprocessing.set_sharing_strategy('file_system')



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
    parser.add_argument('--num-heads', '-nh', type=int, default=4, help='number of heads in multi-head attention')
    parser.add_argument('--dropout', type=float, default=0.0, help='dropout rate')
    parser.add_argument('--party-dropout', type=float, default=0.0, help='dropout rate for entire party')
    parser.add_argument('--n-local-blocks', '-nlb', type=int, default=6, help='number of local blocks')
    parser.add_argument('--n-agg-blocks', '-nab', type=int, default=6, help='number of aggregation blocks')
    parser.add_argument('--knn-k', type=int, default=100, help='k for knn')
    parser.add_argument('--disable-pe', action='store_true', help='disable positional encoding')
    parser.add_argument('--disable-dm', action='store_true', help='disable dynamic masking')
    parser.add_argument('-paf', '--pe-average-freq', type=int, default=0,
                        help='average frequency for positional encoding on each party')

    # parameters for fedsim synthetic dataset
    parser.add_argument('--key-noise', type=float, default=0.0, help='key noise in FedSim synthetic data')

    # parameters for differential privacy
    parser.add_argument('--dp-noise', type=float, default=None, help='noise scale for differential privacy')
    parser.add_argument('--dp-clip', type=float, default=1.0, help='clip bound for differential privacy')
    parser.add_argument('--dp-sample', type=float, default=None, help='sample rate for differential privacy (privacy amplification)')

    # cache parameters
    parser.add_argument('--flush-cache', action='store_true', help='flush cache')
    parser.add_argument('--disable-cache', action='store_true', help='disable cache', default=True) # default to True
    args = parser.parse_args()

    # print hostname
    print(f"Hostname: {os.uname().nodename}")

    path = PartyPath(f"data/syn/{args.dataset}", args.n_parties, 0, args.splitter, args.weights, args.beta,
                     args.seed, fmt='pkl', comm_root="log")
    comm_logger = CommLogger(args.n_parties, path.comm_log)

    real_root = "data/"
    syn_root = "data/syn"
    cache_root = "cache"
    normalize_key = True
    if args.dataset == 'house':
        house_root = f"{real_root}/house/"
        key_dim = 2
        house_dataset = VFLRealDataset.from_csv([os.path.join(house_root, "house_clean.csv"),
                                                 os.path.join(house_root, "airbnb_clean.csv")], key_cols=key_dim,
                                                header=1,
                                                multi_primary=False, ks=args.knn_k,
                                                cache_key=os.path.join(cache_root, "nbr/house/main.pkl"),
                                                use_cache=not args.disable_cache,
                                                sample_rate_before_topk=args.dp_sample)
        train_dataset, val_dataset, test_dataset = house_dataset.split_train_test_primary(
            val_ratio=0.1, test_ratio=0.2, random_state=args.seed, shuffle=True)
        train_dataset.data_sampler.sample_rate_before_topk = args.dp_sample
    elif args.dataset == 'taxi':
        key_dim = 4
        taxi_root = f"{real_root}/nytaxi/"
        bike_path = "bike_201606_clean_sample_2e5.pkl"
        taxi_path = "taxi_201606_clean_sample_1e5.pkl"
        base_loader = NYBikeTaxiLoader(bike_path=os.path.join(taxi_root, bike_path),
                                       taxi_path=os.path.join(taxi_root, taxi_path), link=True)
        [X1, X2], y = base_loader.load_parties()

        # Append two empty columns to X2 since it feature_dim is smaller than key_dim, which leads to an error in
        # PositionalEncoding
        X2 = np.concatenate([X2, np.zeros([X2.shape[0], 2])], axis=1)

        taxi_dataset = VFLRealDataset(([X1, X2], y), primary_party_id=0, key_cols=key_dim, ks=args.knn_k,
                                      sample_rate_before_topk=args.dp_sample)
        train_dataset, val_dataset, test_dataset = taxi_dataset.split_train_test_primary(
            val_ratio=0.1, test_ratio=0.2, random_state=args.seed, shuffle=True)
    elif args.dataset == 'hdb':
        key_dim = 2
        hdb_path = f"{real_root}/hdb/hdb_clean.csv"
        school_path = f"{real_root}/hdb/school_clean.csv"
        [X1, X2], y = load_both_hdb(hdb_path, school_path, active_party='hdb')
        hdb_dataset = VFLRealDataset(([X1, X2], y), primary_party_id=0, key_cols=key_dim, ks=args.knn_k,
                                     sample_rate_before_topk=args.dp_sample)
        train_dataset, val_dataset, test_dataset = hdb_dataset.split_train_test_primary(
            val_ratio=0.1, test_ratio=0.2, random_state=args.seed, shuffle=True)
    elif args.dataset == 'boone':
        key_dim = 30
        boone_root = f"{real_root}/MiniBooNE_PID.txt_scale_{args.key_noise:.1f}_loader.pkl"
        data_loader = FedSimSynLoader.from_pickle(boone_root)
        [X1, X2], y = data_loader.load_parties()

        # in FedSim, the common features of X1 is at the end, move it to the front
        X1 = np.concatenate([X1[:, -key_dim:], X1[:, :-key_dim]], axis=1)

        # Append 20 empty columns since it feature_dim is smaller than key_dim, which leads to an error in
        # PositionalEncoding
        X1 = np.concatenate([X1, np.zeros([X1.shape[0], 20])], axis=1)
        X2 = np.concatenate([X2, np.zeros([X2.shape[0], 20])], axis=1)

        boone_dataset = VFLRealDataset(([X1, X2], y), primary_party_id=0, key_cols=key_dim, ks=args.knn_k)
        train_dataset, val_dataset, test_dataset = boone_dataset.split_train_test_primary(
            val_ratio=0.1, test_ratio=0.2, random_state=args.seed)
    elif args.dataset == 'frog':
        key_dim = 16
        frog_root = f"{real_root}/Frogs_MFCCs.csv_scale_{args.key_noise:.1f}_loader.pkl"
        data_loader = FedSimSynLoader.from_pickle(frog_root)
        [X1, X2], y = data_loader.load_parties()

        # in FedSim, the common features of X1 is at the end, move it to the front
        X1 = np.concatenate([X1[:, -key_dim:], X1[:, :-key_dim]], axis=1)

        # Append 13 empty columns since it feature_dim is smaller than key_dim, which leads to an error in
        # PositionalEncoding
        X1 = np.concatenate([X1, np.zeros([X1.shape[0], 13])], axis=1)
        X2 = np.concatenate([X2, np.zeros([X2.shape[0], 13])], axis=1)

        frog_dataset = VFLRealDataset(([X1, X2], y), primary_party_id=0, key_cols=key_dim, ks=args.knn_k)
        train_dataset, val_dataset, test_dataset = frog_dataset.split_train_test_primary(
            val_ratio=0.1, test_ratio=0.2, random_state=args.seed)

    elif args.dataset in ("cifar10", "gisette", "mnist", "radar"):
        # multi_party_dataset for scalability
        normalize_key = False

        if args.dataset in ("cifar10", "mnist"):
            key_dim = 4
        elif args.dataset in ("radar"):
            key_dim = 4
        else:
            key_dim = 4

        syn_dataset_dir = f"data/syn/{args.dataset}/noise{args.key_noise}/"

        syn_aligned_dataset = VFLSynAlignedDataset.from_pickle(syn_dataset_dir, args.dataset, args.n_parties,
                                                               primary_party_id=args.primary_party,
                                                               splitter=args.splitter,
                                                               weight=args.weights, beta=args.beta, seed=0,
                                                               type=None)
        syn_dataset = VFLRealDataset.from_syn_aligned(syn_aligned_dataset, ks=args.knn_k, key_cols=key_dim,
                                                      sample_rate_before_topk=args.dp_sample,
                                                      use_cache=not args.disable_cache,
                                                      cache_key=f"{args.dataset}-{args.n_parties}",
                                                      multi_primary=False)
        train_dataset, val_dataset, test_dataset = syn_dataset.split_train_test_primary(
            val_ratio=0.1, test_ratio=0.2, random_state=args.seed)
    else:
        key_dim = 0
        syn_dataset_dir = f"{syn_root}/{args.dataset}"
        print(f"Loading synthetic dataset from {syn_dataset_dir}")
        syn_aligned_dataset = VFLSynAlignedDataset.from_pickle(syn_dataset_dir, args.dataset, args.n_parties,
                                                               primary_party_id=args.primary_party,
                                                               splitter=args.splitter,
                                                               weight=args.weights, beta=args.beta, seed=args.seed,
                                                               type=None)
        for local_dataset in syn_aligned_dataset.local_datasets:
            # local_dataset.key = None
            local_dataset.key = np.arange(len(local_dataset)).reshape(-1, 1)    # debug
        syn_dataset = VFLRealDataset.from_syn_aligned(syn_aligned_dataset, ks=args.knn_k)
        syn_dataset.key_cols = 1    # debug
        train_dataset, val_dataset, test_dataset = syn_dataset.split_train_test_primary(
            val_ratio=0.1, test_ratio=0.2, random_state=args.seed)

    # normalize features
    X_scalers = train_dataset.normalize_(include_key=normalize_key)
    if val_dataset is not None:
        val_dataset.normalize_(scalers=X_scalers, include_key=normalize_key)
    test_dataset.normalize_(scalers=X_scalers, include_key=normalize_key)

    # create the model
    y_scaler = None
    if args.n_classes == 1:  # regression
        task = 'reg'
        loss_fn = nn.MSELoss()
        out_dim = 1
        out_activation = nn.Sigmoid()
        if args.metric == 'acc':  # if metric is accuracy, change it to rmse
            args.metric = 'rmse'
            warnings.warn("Metric is changed to rmse for regression task")
        # scale the labels to [0, 1]
        y_scaler = train_dataset.scale_y_()
        if val_dataset is not None:
            val_dataset.scale_y_(scaler=y_scaler)
        test_dataset.scale_y_(scaler=y_scaler)
    elif args.n_classes == 2:  # binary classification
        task = 'bin-cls'
        loss_fn = nn.BCELoss()
        out_dim = 1
        out_activation = nn.Sigmoid()
        # make sure the labels are in [0, 1]
        train_dataset.scale_y_(0, 1)
        if val_dataset is not None:
            val_dataset.scale_y_(0, 1)
        test_dataset.scale_y_(0, 1)
    else:  # multi-class classification
        task = 'multi-cls'
        loss_fn = nn.CrossEntropyLoss()
        out_dim = args.n_classes
        out_activation = None  # No need for softmax since it is included in CrossEntropyLoss

    model = FeT(key_dims=train_dataset.local_key_channels, data_dims=train_dataset.local_input_channels,
                out_dim=out_dim, data_embed_dim=args.data_embed_dim,
                key_embed_dim=args.key_embed_dim,
                num_heads=args.num_heads, dropout=args.dropout, party_dropout=args.party_dropout,
                # n_embeddings=len(train_dataset) + len(test_dataset),
                n_embeddings=None, out_activation=out_activation,
                n_local_blocks=args.n_local_blocks, n_agg_blocks=args.n_agg_blocks, k=args.knn_k,
                rep_noise=args.dp_noise, max_rep_norm=args.dp_clip, enable_pe=not args.disable_pe,
                enable_dm=not args.disable_dm)
    # model = torch.compile(model)

    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer = optim.Lamb(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5, verbose=True)

    train_dataset.to_tensor_()
    test_dataset.to_tensor_()
    if val_dataset is not None:
        val_dataset.to_tensor_()

    def is_debug():
        gettrace = getattr(sys, 'gettrace', None)
        if gettrace is None:
            return False
        elif gettrace():
            return True
        else:
            return False

    if is_debug():
        print("Debug mode, set num_workers to 0")
        n_workers = 0
    else:
        n_workers = 0   # disable multiprocessing for a multi-process bug in pytorch (some times it will freeze)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=n_workers,
                              drop_last=False)
    if val_dataset is not None:
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=n_workers,
                                drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=n_workers,
                             drop_last=False)

    metric_fn = get_metric_from_str(args.metric)
    metric_positive = get_metric_positive_from_str(args.metric)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=str(os.path.join(args.log_dir, args.dataset)))

    cache_dir = os.path.join("cache", args.dataset)
    os.makedirs(cache_dir, exist_ok=True)
    model_path = os.path.join(cache_dir, f"model_{args.dataset}_party{args.n_parties}_knn{args.knn_k}"
                                           f"_{timestamp}.pt")

    test_loss_list, test_score_list = fit(model, optimizer, loss_fn, metric_fn, train_loader, epochs=args.epochs,
                                          gpu_id=args.gpu,
                                          n_classes=args.n_classes, test_loader=test_loader, task=task,
                                          scheduler=scheduler, has_key=True,
                                          val_loader=val_loader, metric_positive=metric_positive, y_scaler=y_scaler,
                                          solo=False, writer=writer, log_timestamp=timestamp,
                                          visualize=False, model_path=model_path, dataset_name=args.dataset)

    if args.result_path is not None:
        # save test loss and score to a two-column csv file, each row is for one epoch (with pandas)
        test_result = pd.DataFrame({'loss': test_loss_list, 'score': test_score_list})
        test_result.to_csv(args.result_path, index=False)

    print("Done!")
