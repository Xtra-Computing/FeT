import warnings
import numpy as np
import multiprocessing as mp

warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")

import os
import sys
import datetime, pytz

import torch
from torchinfo import summary
from src.model.FeT import FeT


import pandas as pd
from tqdm import tqdm
import deprecated

# add src to python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils.BasicUtils import get_device_from_gpu_id, get_metric_from_str, PartyPath


@deprecated.deprecated(reason="Previously used for handling format, now deprecated")
def preprocess_Xs_y_temp(Xs, y, device, task, solo=False, has_key=False, duplicate_y=None):
    Xs, y = preprocess_Xs_y(Xs, y, device, task, solo=solo, has_key=has_key, duplicate_y=duplicate_y)
    concated_Xs = [Xi[1].squeeze(1) for Xi in Xs] # shape [128, 1, 2317] --> [128, 2317]
    return concated_Xs, y


def preprocess_Xs_y(Xs, y, device, task, solo=False, has_key=False, duplicate_y=None):
    # pdbr.set_trace()
    if not has_key:
        Xs = [Xi.to(device) for Xi in Xs]
        default_keys = torch.arange(Xs[0].shape[0]).reshape(-1, 1).long().to(device)
        Xs = [(default_keys, Xi) for Xi in Xs]
    else:
        if solo:
            Xs = (Xs[0].float().to(device), Xs[1].float().to(device))
        else:
            Xs = [(Xi[0].float().to(device), Xi[1].float().to(device)) for Xi in Xs]
    y = y.to(device)
    y = y.flatten()
    if duplicate_y is not None:
        y = y.repeat(duplicate_y)
    y = y.long() if task == 'multi-cls' else y.float()
    return Xs, y


def summarize_fet_model(model, loader, depth=4):
    sample_Xs = next(iter(loader))[0]
    sample_Xs_shape = [tuple(torch.cat([Xi[0], Xi[1]], dim=-1).shape) for Xi in sample_Xs]
    key_dim = sample_Xs[0][0].shape[-1]

    # wrap the model with a concatenation
    class WrapModel(torch.nn.Module):
        def __init__(self, model, key_dim):
            super().__init__()
            self.model = model
            self.key_dim = key_dim

        def forward(self, *key_Xs_concat):
            key_Xs = [(key_X[:, :, :self.key_dim], key_X[:, :, self.key_dim:]) for key_X in key_Xs_concat]
            return self.model(key_Xs)

    wrap_model = WrapModel(model, key_dim)
    stats = summary(wrap_model, sample_Xs_shape, depth=depth)
    return stats



def fit(model, optimizer, loss_fn, metric_fn, train_loader, test_loader=None, epochs=10, gpu_id=0, n_classes=1,
        task='bin-cls', scheduler=None, has_key=False, val_loader=None, metric_positive=True, y_scaler=None,
        solo=False, writer=None, log_timestamp=None, visualize=False, model_path=None, dataset_name=None, fig_dir='fig',
        average_pe_freq=None):
    device = get_device_from_gpu_id(gpu_id)
    model.to(device)

    if fig_dir is not None:
        os.makedirs(fig_dir, exist_ok=True)
    if isinstance(model, FeT):
        stats = summarize_fet_model(model, train_loader, depth=4)

    test_loss_list = []
    test_score_list = []
    best_epoch = -1
    if metric_positive:
        best_train_score = -np.inf
        best_val_score = -np.inf
        best_test_score = -np.inf
    else:
        best_train_score = np.inf
        best_val_score = np.inf
        best_test_score = np.inf

    for epoch in range(epochs):
        model.train()
        train_pred_y = train_y = torch.zeros([0, 1], device=device)
        train_loss = 0

        for Xs, y in tqdm(train_loader):
            # pdbr.set_trace()
            Xs, y = preprocess_Xs_y(Xs, y, device, task, solo=solo, has_key=has_key)

            optimizer.zero_grad()
            y_pred = model(Xs)

            y_pred = y_pred.flatten() if task in ['reg', 'bin-cls'] else y_pred
            loss = loss_fn(y_pred, y)
            train_loss += loss.item()

            if n_classes == 2:
                y_pred = torch.round(y_pred)
            elif n_classes > 2:
                y_pred = torch.argmax(y_pred, dim=1).reshape(-1, 1)

            y_pred = y_pred.reshape(-1, 1)
            train_pred_y = torch.cat([train_pred_y, y_pred], dim=0)
            train_y = torch.cat([train_y, y.reshape(-1, 1)], dim=0)
            loss.backward()
            optimizer.step()

        train_y_array = train_y.data.cpu().numpy()
        train_pred_y_array = train_pred_y.data.cpu().numpy()
        if y_scaler is not None:
            train_y_array = y_scaler.inverse_transform(train_y_array.reshape(-1, 1)).reshape(-1)
            train_pred_y_array = y_scaler.inverse_transform(train_pred_y_array.reshape(-1, 1)).reshape(-1)
        # pdbr.set_trace()
        train_score = metric_fn(train_y_array, train_pred_y_array)

        timestamp_now = datetime.datetime.now(pytz.timezone('Asia/Shanghai')).strftime("%Y-%m-%d %H:%M:%S")
        print(timestamp_now, f"Epoch: {epoch}, Train Loss: {train_loss / len(train_loader)}, Train Score: {train_score}")
        if hasattr(model, 'comm_logger'):
            model.comm_logger.save_log()
        if writer is not None:
            writer.add_scalars(f"loss", {f'{log_timestamp}/train': train_loss / len(train_loader)}, epoch)
            writer.add_scalars(f"score", {f'{log_timestamp}/train': train_score}, epoch)

        if scheduler is not None and val_loader is None:
            scheduler.step(loss.item() / len(train_loader))

        # if train_loader.dataset.cache_need_update and epoch == 0:
        #     train_loader.dataset.cache_need_update = False
            # train_loader.dataset.cache.save_pkl()

        # if visualize:
        #     model.visualize_positional_encoding()
        # pdbr.set_trace()

        # # debug
        # if hasattr(model, 'save_pe_inout') and epoch % 20 == 0 and epoch != 0:
        #     # inference and save the input and output of pe layer
        #     model.save_pe_inout(val_loader, f"log/{dataset_name}/_pe_inout_independent", device=device)

        if val_loader is not None:
            model.eval()
            with torch.no_grad():
                val_loss = torch.zeros(1, device=device)
                val_y_pred = val_y = torch.zeros([0, 1], device=device)
                for Xs, y in tqdm(val_loader):
                    Xs, y = preprocess_Xs_y(Xs, y, device, task, solo=solo, has_key=has_key)

                    y_pred = model(Xs)
                    y_pred = y_pred.flatten() if task in ['reg', 'bin-cls'] else y_pred
                    val_loss += loss_fn(y_pred, y)

                    if n_classes == 2:
                        y_pred = torch.round(y_pred)
                    elif n_classes > 2:
                        y_pred = torch.argmax(y_pred, dim=1).reshape(-1, 1)

                    if y_pred.isnan().any():
                        warnings.warn("y_pred has nan")
                        y_pred[y_pred.isnan()] = 0

                    y_pred = y_pred.reshape(-1, 1)
                    val_y_pred = torch.cat([val_y_pred, y_pred], dim=0)
                    val_y = torch.cat([val_y, y.reshape(-1, 1)], dim=0)

                val_y_array = val_y.data.cpu().numpy()
                val_y_pred_array = val_y_pred.data.cpu().numpy()

                if y_scaler is not None:
                    scaled_val_y_array = y_scaler.inverse_transform(val_y_array.reshape(-1, 1)).reshape(-1)
                    scaled_val_y_pred_array = y_scaler.inverse_transform(val_y_pred_array.reshape(-1, 1)).reshape(-1)
                else:
                    scaled_val_y_array = val_y_array
                    scaled_val_y_pred_array = val_y_pred_array
                try:
                    val_score = metric_fn(scaled_val_y_array, scaled_val_y_pred_array)
                except ValueError as e:
                    print(f"Error: {e}")
                    raise e

                val_loss_mean = float(val_loss.cpu().numpy() / len(val_loader))
                timestamp_now = datetime.datetime.now(pytz.timezone('Asia/Shanghai')).strftime("%Y-%m-%d %H:%M:%S")
                print(timestamp_now, f"Epoch: {epoch}, Val Loss: {val_loss_mean}, Val Score: {val_score}")

                if writer is not None:
                    writer.add_scalars(f"loss", {f'{log_timestamp}/val': val_loss_mean}, epoch)
                    writer.add_scalars(f"score", {f'{log_timestamp}/val': val_score}, epoch)

                scheduler.step(val_score)

                # if val_loader.dataset.cache_need_update and epoch == 0:
                #     val_loader.dataset.cache_need_update = False
                    # val_loader.dataset.cache.save_pkl()

        if test_loader is not None:
            model.eval()
            with torch.no_grad():
                test_loss = torch.zeros(1, device=device)
                test_y_pred = test_y = torch.zeros([0, 1], device=device)
                for Xs, y in tqdm(test_loader):
                    Xs, y = preprocess_Xs_y(Xs, y, device, task, solo=solo, has_key=has_key)

                    y_pred = model(Xs)
                    y_pred = y_pred.flatten() if task in ['reg', 'bin-cls'] else y_pred
                    test_loss += loss_fn(y_pred, y)

                    if n_classes == 2:
                        y_pred = torch.round(y_pred)
                    elif n_classes > 2:
                        y_pred = torch.argmax(y_pred, dim=1).reshape(-1, 1)

                    if y_pred.isnan().any():
                        warnings.warn("y_pred has nan")
                        y_pred[y_pred.isnan()] = 0

                    y_pred = y_pred.reshape(-1, 1)
                    test_y_pred = torch.cat([test_y_pred, y_pred], dim=0)
                    test_y = torch.cat([test_y, y.reshape(-1, 1)], dim=0)

                test_y_array = test_y.data.cpu().numpy()
                test_y_pred_array = test_y_pred.data.cpu().numpy()

                if y_scaler is not None:
                    scaled_test_y_array = y_scaler.inverse_transform(test_y_array.reshape(-1, 1)).reshape(-1)
                    scaled_test_y_pred_array = y_scaler.inverse_transform(test_y_pred_array.reshape(-1, 1)).reshape(-1)
                else:
                    scaled_test_y_array = test_y_array
                    scaled_test_y_pred_array = test_y_pred_array
                try:
                    test_score = metric_fn(scaled_test_y_array, scaled_test_y_pred_array)
                except ValueError as e:
                    print(f"Error: {e}")
                    raise e
                test_loss_mean = float(test_loss.cpu().numpy() / len(test_loader))
                timestamp_now = datetime.datetime.now(pytz.timezone('Asia/Shanghai')).strftime("%Y-%m-%d %H:%M:%S")
                print(timestamp_now, f"Epoch: {epoch}, Test Loss: {test_loss_mean}, Test Score: {test_score}")
                test_loss_list.append(test_loss_mean)
                test_score_list.append(test_score)

                if writer is not None:
                    writer.add_scalars(f"loss", {f'{log_timestamp}/test': test_loss_mean}, epoch)
                    writer.add_scalars(f"score", {f'{log_timestamp}/test': test_score}, epoch)

                # if test_loader.dataset.cache_need_update and epoch == 0:
                #     test_loader.dataset.cache_need_update = False
                    # test_loader.dataset.cache.save_pkl()

        if visualize:
            model.visualize_positional_encoding(dataset=dataset_name, device=device,
                                                save_path=os.path.join(fig_dir, f"{dataset_name}_epoch{epoch}.png"))

        if val_loader is not None:
            if (metric_positive and val_score > best_val_score) or (not metric_positive and val_score < best_val_score):
                best_val_score = val_score
                best_train_score = train_score
                best_test_score = test_score
                best_epoch = epoch
                if model_path is not None:
                    model.save(model_path)
            print(f"Best epoch: {best_epoch}, Train Score: {best_train_score}, Val Score: {best_val_score}, Test Score: {best_test_score}")

        # Average the positional encoding layer
        if (hasattr(model, 'average_pe_') and (average_pe_freq is not None or average_pe_freq == 0)
                and epoch % average_pe_freq == 0 and epoch != 0):
            model.average_pe_()
    return test_loss_list, test_score_list