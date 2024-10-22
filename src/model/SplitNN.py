import warnings
warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")
from typing import Callable
import argparse
import warnings
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.ops import MLP
from torchvision.models import resnet18
# from torchsummaryX import summary

import pandas as pd
from tqdm import tqdm

# add src to python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dataset.VFLDataset import VFLSynAlignedDataset
from src.utils.BasicUtils import get_device_from_gpu_id, get_metric_from_str, PartyPath
from src.utils.logger import CommLogger
from train.Fit import fit
from train.Evaluate import evaluate


class SplitMLP(nn.Module):
    def __init__(self, local_input_channels, local_hidden_channels, agg_hidden_channels,
                 out_activation=None, comm_logger=None, primary_party=0, **kwargs):
        """
        SplitNN that all layers are fully connected layers (MLP).
        Usage: (f is the number of features)
        -------------------- Examples ----------------------------
        - SplitMLP(local_layers=[[10], [10,20]], agg_layers=[50], output_dim=1)
        | 2 parties, party 0 has 1 hidden layer with 10 neurons, party 1 has 1 hidden layer with 20 neurons,
        | the aggregation layer has 50 neurons, and the output layer has 1 neuron.
        |            50 x 1
        |               |
        |            30 x 50
        |               |
        |            concat
        |            /     \
        |        f x 10   10 x 20
        |           /       \
        |        data1    f x 10
        |                    \
        |                   data2
        |
        - SplitMLP(local_layers=[[10]] * 3, agg_layers=[], output_dim=1)
        | 3 parties, each party has 1 hidden layers with 10 neurons, and the output layer has 1 neuron.
        |            30 x 1
        |               |
        |            concat
        |         /     |      \
        |     f x 10  f x 10  f x 10
        |        /      |       \
        |     data1   data2    data3
        |
        ---------------------------------------------------------

        Parameters
        ----------
        local_input_channels : list[int]
            list of local input channels. Same as the number of features of each party.
        local_hidden_channels : list[list[int]]
            list of local hidden channels. Note that each sublist must be non-empty, because each party must have at
            least one hidden layer to avoid directly transferring the data to the primary party.
        agg_hidden_channels : list[int]
            list of aggregation channels. Note that this list must be non-empty, because the aggregation layer must
            output a prediction result. The last element of this list is the number of output channels.
        output_channel : int
            output channel of the model
        out_activation : Callable
            activation function of the output layer
        comm_logger : CommLogger
            logger of communication size. None if no logging is needed.
        primary_party : int
            ID of the primary party
        kwargs : dict
            other parameters for torchvision.ops.MLP
            For example:
            - hidden_activation: nn.ReLU
            - dropout: 0.0
            ...
        """
        super().__init__()
        self.local_input_channels = local_input_channels
        self.local_hidden_channels = local_hidden_channels
        self.agg_hidden_channels = agg_hidden_channels
        self.out_activation = out_activation
        self.n_parties = len(local_input_channels)
        assert len(local_input_channels) == len(local_hidden_channels), \
            f"The number of parties must be the same. Got {len(local_input_channels)} and {len(local_hidden_channels)}."

        self.local_mlps = nn.ModuleList()
        for i in range(self.n_parties):
            if local_input_channels[i] == 0:
                # if the party has no data, then the local MLP directly outputs zero dimension tensor
                local_mlp = nn.Identity()
            else:
                local_mlp = MLP(local_input_channels[i], local_hidden_channels[i], **kwargs)
            self.local_mlps.append(local_mlp)

        valid_hidden_channels = []
        for i in range(self.n_parties):
            if local_input_channels[i] != 0:
                valid_hidden_channels.append(local_hidden_channels[i])
        self.cut_dim = sum([channel[-1] for channel in valid_hidden_channels])  # the dimension of the cut layer
        self.agg_mlp = MLP(self.cut_dim, agg_hidden_channels, **kwargs)

        self.comm_logger = comm_logger
        self.primary_party = primary_party

        # # register hook for the gradients of cut layers of each party
        # for i in range(self.n_parties):
        #     if i != self.primary_party:
        #         cut_layer = self.local_mlps[i][-2]
        #         cut_layer.register_full_backward_hook(lambda module, grad_input, grad_output:
        #                                               self.get_grad_size_in_cut_layer(grad_output, comm_logger,
        #                                                                               primary_party, i))

    def forward(self, Xs):
        """
        Forward propagation of the model.

        Parameters
        ----------
        Xs : list[torch.Tensor]
            list of local data. Each element is a tensor of shape (batch_size, local_input_channels[i])

        Returns
        -------
        torch.Tensor
            output of the model. Shape: (batch_size, output_channel)
        """
        if isinstance(Xs[0], tuple):
            # print(f"({Xs[0][0].squeeze(1).shape}, {Xs[0][1].squeeze(1).shape})")
            Xs = [torch.cat([Xi[0].squeeze(1), Xi[1].squeeze(1)], dim=-1) for Xi in Xs]

        local_outputs = [mlp(Xi) for mlp, Xi in zip(self.local_mlps, Xs)]

        # communication cost of the forward propagation
        if self.comm_logger is not None:
            for i, local_output in enumerate(local_outputs):
                if i != self.primary_party:
                    comm_size = local_output.nelement() * local_output.element_size()
                    self.comm_logger.comm(i, self.primary_party, comm_size)

                    # the size of the gradient in the cut layer in backward propagation is the same as the size of the
                    # output of the cut layer in forward propagation
                    self.comm_logger.comm(self.primary_party, i, comm_size)

        agg_input = torch.cat(local_outputs, dim=1)
        agg_output = self.agg_mlp(agg_input)
        if self.out_activation is not None:
            return self.out_activation(agg_output)
        else:
            return agg_output

    # @staticmethod
    # def get_grad_size_in_cut_layer(grad_out, comm_logger: CommLogger, primary_party, secondary_party):
    #     """
    #     Get the size of the gradient in the cut layer.
    #
    #     Parameters
    #     ----------
    #     grad_out : list[torch.Tensor]
    #         list of gradients of the cut layer
    #     comm_logger : CommLogger
    #         logger of communication size
    #     primary_party : int
    #         ID of the primary party
    #     secondary_party : int
    #         ID of the secondary party
    #     """
    #
    #     size = 0
    #     print("Hook called!")
    #     for grad in grad_out:
    #         if grad is not None:
    #             print(f"{primary_party} -> {secondary_party}: {grad.nelement() * grad.element_size()}")
    #             size += grad.nelement() * grad.element_size()
    #             comm_logger.comm(primary_party, secondary_party, size)

class SplitSumMLP(nn.Module):
    def __init__(self, local_input_channels, local_hidden_channels, agg_hidden_channels,
                 out_activation=None, comm_logger=None, primary_party=0, **kwargs):
        """
        SplitNN that all layers are fully connected layers (MLP).
        Usage: (f is the number of features)
        -------------------- Examples ----------------------------
        - SplitMLP(local_layers=[[10], [10,20]], agg_layers=[50], output_dim=1)
        | 2 parties, party 0 has 1 hidden layer with 10 neurons, party 1 has 1 hidden layer with 20 neurons,
        | the aggregation layer has 50 neurons, and the output layer has 1 neuron.
        |            50 x 1
        |               |
        |            30 x 50
        |               |
        |              Sum
        |            /     \
        |        f x 10   10 x 20
        |           /       \
        |        data1    f x 10
        |                    \
        |                   data2
        |
        - SplitMLP(local_layers=[[10]] * 3, agg_layers=[], output_dim=1)
        | 3 parties, each party has 1 hidden layers with 10 neurons, and the output layer has 1 neuron.
        |            30 x 1
        |               |
        |            concat
        |         /     |      \
        |     f x 10  f x 10  f x 10
        |        /      |       \
        |     data1   data2    data3
        |
        ---------------------------------------------------------

        Parameters
        ----------
        local_input_channels : list[int]
            list of local input channels. Same as the number of features of each party.
        local_hidden_channels : list[list[int]]
            list of local hidden channels. Note that each sublist must be non-empty, because each party must have at
            least one hidden layer to avoid directly transferring the data to the primary party.
        agg_hidden_channels : list[int]
            list of aggregation channels. Note that this list must be non-empty, because the aggregation layer must
            output a prediction result. The last element of this list is the number of output channels.
        output_channel : int
            output channel of the model
        out_activation : Callable
            activation function of the output layer
        comm_logger : CommLogger
            logger of communication size. None if no logging is needed.
        primary_party : int
            ID of the primary party
        kwargs : dict
            other parameters for torchvision.ops.MLP
            For example:
            - hidden_activation: nn.ReLU
            - dropout: 0.0
            ...
        """
        super().__init__()
        self.local_input_channels = local_input_channels
        self.local_hidden_channels = local_hidden_channels
        self.agg_hidden_channels = agg_hidden_channels
        self.out_activation = out_activation
        self.n_parties = len(local_input_channels)
        assert len(local_input_channels) == len(local_hidden_channels), \
            f"The number of parties must be the same. Got {len(local_input_channels)} and {len(local_hidden_channels)}."

        self.local_mlps = nn.ModuleList()
        for i in range(self.n_parties):
            if local_input_channels[i] == 0:
                # if the party has no data, then the local MLP directly outputs zero dimension tensor
                local_mlp = nn.Identity()
            else:
                local_mlp = MLP(local_input_channels[i], local_hidden_channels[i], **kwargs)
            self.local_mlps.append(local_mlp)

        valid_hidden_channels = []
        for i in range(self.n_parties):
            if local_input_channels[i] != 0:
                valid_hidden_channels.append(local_hidden_channels[i])
        # self.cut_dim = sum([channel[-1] for channel in valid_hidden_channels])  # the dimension of the cut layer
        self.cut_dim = valid_hidden_channels[0][-1]
        self.agg_mlp = MLP(self.cut_dim, agg_hidden_channels, **kwargs)

        self.comm_logger = comm_logger
        self.primary_party = primary_party

    def forward(self, key_Xs):
        """
        Forward propagation of the model.

        Parameters
        ----------
        key_Xs : list[tuple(torch.Tensor, torch.Tensor)]
            tuple of local keys and data. Each data matrix is a tensor of shape (batch_size, local_input_channels[i])

        Returns
        -------
        torch.Tensor
            output of the model. Shape: (batch_size, output_channel)
        """
        local_outputs = [mlp(Xi[1]) for mlp, Xi in zip(self.local_mlps, key_Xs)]

        agg_input = torch.stack(local_outputs, dim=0).sum(dim=0)
        agg_output = self.agg_mlp(agg_input)
        if self.out_activation is not None:
            return self.out_activation(agg_output)
        else:
            return agg_output



class SplitResNet(nn.Module):
    def __init__(self, n_parties, agg_hidden=None, out_activation=None):
        super().__init__()
        self.n_parties = n_parties
        self.out_activation = out_activation
        self.local_resnet_list = nn.ModuleList()
        local_output_dims = []
        for i in range(self.n_parties):
            resnet = resnet18(weights=None)
            local_output_dims.append(resnet.fc.in_features)
            resnet.fc = nn.Identity()
            resnet.conv1 = nn.Conv2d(13, 64, 9, stride=2, padding=3, bias=False)
            self.local_resnet_list.append(resnet)
        self.cut_dim = sum(local_output_dims)
        if agg_hidden is None:
            self.agg_hidden = [100, 1]
        else:
            self.agg_hidden = agg_hidden
        self.agg_mlp = MLP(self.cut_dim, self.agg_hidden)
        if out_activation is None:
            self.out_activation = nn.Identity()
        else:
            self.out_activation = out_activation

    def forward(self, Xs):
        local_outputs = [resnet(Xi) for resnet, Xi in zip(self.local_resnet_list, Xs)]
        agg_input = torch.cat(local_outputs, dim=1)
        agg_output = self.agg_mlp(agg_input)
        return self.out_activation(agg_output)


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
    parser.add_argument('--epochs', '-e', type=int, default=50)
    parser.add_argument('--lr', '-lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-5)
    parser.add_argument('--batch_size', '-bs', type=int, default=128)
    parser.add_argument('--n_classes', '-c', type=int, default=7,
                        help="number of classes. 1 for regression, 2 for binary classification,"
                             ">=3 for multi-class classification")
    parser.add_argument('--metric', '-m', type=str, default='acc',
                        help="metric to evaluate the model. Supported metrics: [accuracy, rmse]")
    parser.add_argument('--model', '-md', type=str, default='mlp',
                        help="model to use. Supported models: [mlp, cnn, resnet18]")
    parser.add_argument('--split-mode', '-sm', type=str, default='concat',
                        help="split mode. Supported modes: [concat, sum]")
    parser.add_argument('--result-path', '-rp', type=str, default=None,
                        help="path to save the result")
    parser.add_argument('--seed', '-s', type=int, default=0, help="random seed")
    args = parser.parse_args()

    path = PartyPath(f"data/syn/{args.dataset}", args.n_parties, 0, args.splitter, args.weights, args.beta,
                     args.seed, fmt='pkl', comm_root="log")
    comm_logger = CommLogger(args.n_parties, path.comm_log)


    # Note: torch.compile() in torch 2.0 significantly harms the accuracy with little speed up
    train_dataset = VFLSynAlignedDataset.from_pickle(f"data/syn/{args.dataset}", f'{args.dataset}', args.n_parties,
                                                  primary_party_id=args.primary_party, splitter=args.splitter,
                                                  weight=args.weights, beta=args.beta, seed=args.seed, type='train')
    test_dataset = VFLSynAlignedDataset.from_pickle(f"data/syn/{args.dataset}", f'{args.dataset}', args.n_parties,
                                                 primary_party_id=args.primary_party, splitter=args.splitter,
                                                 weight=args.weights, beta=args.beta, seed=args.seed, type='test')
    model = 'mlp'

    # create the model
    if args.n_classes == 1:  # regression
        task = 'reg'
        loss_fn = nn.MSELoss()
        out_dim = 1
        out_activation = nn.Sigmoid()
        if args.metric == 'acc':  # if metric is accuracy, change it to rmse
            args.metric = 'rmse'
            warnings.warn("Metric is changed to rmse for regression task")
    elif args.n_classes == 2:  # binary classification
        task = 'bin-cls'
        loss_fn = nn.BCELoss()
        out_dim = 1
        out_activation = nn.Sigmoid()
        # make sure the labels are in [0, 1]
        train_dataset.scale_y_()
        test_dataset.scale_y_()
    else:  # multi-class classification
        task = 'multi-cls'
        loss_fn = nn.CrossEntropyLoss()
        out_dim = args.n_classes
        out_activation = None   # No need for softmax since it is included in CrossEntropyLoss

    if model == 'mlp':
        if args.split_mode == 'concat':
            model = SplitMLP(train_dataset.local_input_channels, [[100, 100]] * args.n_parties, [200, out_dim],
                             out_activation=out_activation, comm_logger=comm_logger, primary_party=args.primary_party)
        elif args.split_mode == 'sum':
            model = SplitSumMLP(train_dataset.local_input_channels, [[100, 100]] * args.n_parties, [200, out_dim],
                                out_activation=out_activation, comm_logger=comm_logger, primary_party=args.primary_party)
        else:
            raise ValueError(f"Unsupported split mode: {args.split_mode}")
    elif model == 'resnet':
        model = SplitResNet(args.n_parties, agg_hidden=[1000, out_dim], out_activation=out_activation)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    metric_fn = get_metric_from_str(args.metric)

    test_loss_list, test_score_list = fit(model, optimizer, loss_fn, metric_fn, train_loader, epochs=args.epochs, gpu_id=args.gpu,
        n_classes=args.n_classes, test_loader=test_loader, task=task, scheduler=scheduler)

    if args.result_path is not None:
        # save test loss and score to a two-column csv file, each row is for one epoch (with pandas)
        test_result = pd.DataFrame({'loss': test_loss_list, 'score': test_score_list})
        test_result.to_csv(args.result_path, index=False)

