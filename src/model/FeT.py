import os
import sys
from typing import Sequence, Callable
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import torch.random

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from .PosEncoding import LearnableFourierPositionalEncoding

from contextlib import contextmanager



class FeedForward(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout, activation):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout_ratio = dropout
        self.activation = activation

        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class SelfAttnBlock(nn.Module):
    def __init__(self, data_embed_dim, num_heads, dropout, activation):
        super().__init__()
        self.data_embed_dim = data_embed_dim
        self.num_heads = num_heads
        self.dropout_ratio = dropout
        self.activation = activation

        self.attn = nn.MultiheadAttention(data_embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x, need_weights=False, key_padding_mask=None):
        x_embed, attn_weights = self.attn(x, x, x, need_weights=need_weights,
                                          average_attn_weights=not need_weights, key_padding_mask=key_padding_mask)
        x_embed = self.attn_dropout(x_embed)
        return x_embed, attn_weights


class AggAttnBlock(nn.Module):
    def __init__(self, data_embed_dim, num_heads, dropout, activation):
        super().__init__()
        self.data_embed_dim = data_embed_dim
        self.num_heads = num_heads
        self.dropout_ratio = dropout
        self.activation = activation

        self.attn = nn.MultiheadAttention(data_embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x, mem, need_weights=False, key_padding_mask=None):
        x_embed, attn_weights = self.attn(x, mem, mem, need_weights=need_weights,
                                          average_attn_weights=not need_weights, key_padding_mask=key_padding_mask)
        x_embed = self.attn_dropout(x_embed)
        return x_embed, attn_weights


class SelfAttnChain(nn.Module):
    def __init__(self, data_embed_dim, num_heads, dropout, activation, n_blocks):
        super().__init__()
        self.data_embed_dim = data_embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation
        self.n_blocks = n_blocks
        self.attn_blocks = nn.ModuleList([SelfAttnBlock(data_embed_dim, num_heads, dropout, activation)
                                          for _ in range(n_blocks)])
        self.ffn_blocks = nn.ModuleList(
            [FeedForward(data_embed_dim, data_embed_dim * 2, data_embed_dim, dropout, activation)
             for _ in range(n_blocks)])
        self.attn_norm = nn.LayerNorm(data_embed_dim)
        self.ffn_norm = nn.LayerNorm(data_embed_dim)

        self.attn_weights = []  # for visualization, not used in forward

    def forward(self, x, need_weights=False, random_mask=False, key_padding_mask=None):
        # random mask with -inf value, used for training. Add the mask to x before the self-attention
        if random_mask and self.training:
            mask_valid_size = torch.randint(1, x.shape[1] + 1, (x.shape[0],))
            # mask_valid_size = D.Exponential(0.2).sample(torch.Size([x.shape[0]])).long() + 1
            mask_valid_size = torch.clamp(mask_valid_size, min=1, max=x.shape[1])
            mask = torch.zeros_like(x)
            for i in range(x.shape[0]):
                mask[i, mask_valid_size[i]:] = -1e-20
            x = x + mask

        for attn_block, ffn_block in zip(self.attn_blocks, self.ffn_blocks):
            x_emb, attn_weight = attn_block(self.attn_norm(x), need_weights=need_weights,
                                            key_padding_mask=key_padding_mask)
            x = x + x_emb
            x = x + ffn_block(self.ffn_norm(x))
            if need_weights:
                self.attn_weights.append(attn_weight)
        return x

    @torch.no_grad()
    def visualize_attention(self, head=None):
        """
        Visualize the attention weights.
        :param head: If None, average the attention weights across all heads. Otherwise, visualize attention of the
                    specified head.
        """
        for idx, attn_weights in enumerate(self.attn_weights):
            if head is not None:
                attn_weights = attn_weights[:, head, :, :][0]  # Select the specified head

            # Average attention weights across all heads if head=None
            else:
                attn_weights = attn_weights.mean(dim=1)[0]

            # Plot the attention weights
            plt.figure(figsize=(10, 10))
            plt.imshow(attn_weights.squeeze(0).cpu().detach().numpy(), cmap='viridis')
            plt.title(f"Attention Weights from Block {idx + 1}")
            plt.colorbar()
            plt.show()


class AggAttnChain(nn.Module):
    def __init__(self, data_embed_dim, num_heads, dropout, activation, n_blocks):
        super().__init__()
        self.data_embed_dim = data_embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation
        self.n_blocks = n_blocks
        self.self_attn_blocks = nn.ModuleList([SelfAttnBlock(data_embed_dim, num_heads, dropout, activation)
                                               for _ in range(n_blocks)])
        self.agg_attn_blocks = nn.ModuleList([AggAttnBlock(data_embed_dim, num_heads, dropout, activation)
                                              for _ in range(n_blocks)])
        self.ffn_blocks = nn.ModuleList([FeedForward(data_embed_dim, data_embed_dim * 2, data_embed_dim, dropout,
                                                     activation) for _ in range(n_blocks)])
        self.attn_norm1 = nn.LayerNorm(data_embed_dim)
        self.attn_norm2 = nn.LayerNorm(data_embed_dim)
        self.ffn_norm = nn.LayerNorm(data_embed_dim)

        self.attn_weights = []  # for visualization, not used in forward

    def forward(self, x, mem, need_weights=False, random_mask=False, key_padding_mask=None):
        # random mask with -inf value, used for training. Add the mask to mem before attention
        if random_mask and self.training:
            mask_valid_size = torch.randint(1, mem.shape[1] + 1, (mem.shape[0],))
            # mask_valid_size = D.Exponential(0.2).sample(torch.Size([mem.shape[0]])).long() + 1
            mask_valid_size = torch.clamp(mask_valid_size, min=1, max=mem.shape[1])
            mask = torch.zeros_like(mem)
            for i in range(mem.shape[0]):
                mask[i, mask_valid_size[i]:] = -1e-20
            mem = mem + mask

        for self_attn_block, agg_attn_block, ffn_block in (
                zip(self.self_attn_blocks, self.agg_attn_blocks, self.ffn_blocks)):
            x_emb, attn_weight = self_attn_block(self.attn_norm1(x), need_weights=need_weights)
            x = x + x_emb
            x_emb, attn_weight = agg_attn_block(self.attn_norm2(x), mem, need_weights=need_weights,
                                                key_padding_mask=key_padding_mask)
            x = x + x_emb
            x = x + ffn_block(self.ffn_norm(x))
            if need_weights:
                self.attn_weights.append(attn_weight)
        return x


class FeT(nn.Module):
    def __init__(self, key_dims: Sequence, data_dims: Sequence, out_dim: int,
                 key_embed_dim: int, data_embed_dim: int,
                 num_heads: int = 1, dropout: float = 0.1, party_dropout: float = 0.0, n_embeddings: int = None,
                 activation: str = 'gelu', out_activation: Callable = None,
                 n_local_blocks: int = 1, n_agg_blocks: int = 1, primary_party_id: int = 0, k=1,
                 rep_noise=None, max_rep_norm=None, enable_pe=True, enable_dm=True):
        """
        FedTrans model.
        :param n_local_blocks:
        :param n_global_blocks:
        :param key_dims: Sequence of key dimensions of each party, e.g., [k1, ..., kn], where n is the number of parties
        :param data_dims: Sequence of data dimensions of each party, e.g., [d1, ..., dn],
                          where n is the number of parties
        :param out_dim: the output dimension (n_classes) of the model, supposed to be a positive integer.
        :param key_embed_dim: the embedding dimension of the key, supposed to be a positive integer.
        :param data_embed_dim: the embedding dimension of the data, supposed to be a positive integer.
        :param num_heads: the number of heads in the multi-head attention, supposed to be a positive integer and should
                          be divisible by key_embed_dim and data_embed_dim.
        :param dropout: the dropout rate, supposed to be a positive float.
        :param n_embeddings: the number of embeddings for each party, supposed to be a positive integer. If None,
                             dense layer is used instead of embedding layer.
        :param activation: the activation function in hidden layers, supposed to be one of
                           ['relu', 'gelu', 'leakyrelu']. Note that the activation function is not applied to the
                           output layer. The activation is case-insensitive.
        :param out_activation: the activation function in the output layer, supposed to be a callable function. None
                                 means no activation function.
        :param n_local_blocks: the number of local blocks in the model, supposed to be a positive integer.
        :param n_agg_blocks: the number of aggregated blocks (on the primary party) in the model, supposed to be
                                a positive integer.
        :param primary_party_id: the ID of the primary party, supposed to be an integer in [0, n_parties).
        :param k: the number of nearest neighbors to be used in the cut layer, supposed to be a positive integer.
        :param rep_noise: the noise added to the representation in the cut layer, supposed to be a positive float.
        :param max_rep_norm: the maximum norm of the representation in the cut layer, supposed to be a positive float.
        :param enable_pe: whether to enable positional encoding, supposed to be a boolean.
        :param enable_dm: whether to enable dynamic masking, supposed to be a boolean.
        """
        super().__init__()
        self.key_dims = key_dims
        self.data_dims = data_dims
        self.out_dim = out_dim
        # self.key_embed_dim = key_embed_dim
        self.data_embed_dim = data_embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.party_dropout = party_dropout
        self.n_embeddings = n_embeddings
        self.activation = activation
        self.out_activation = out_activation
        self.n_local_blocks = n_local_blocks
        self.n_agg_blocks = n_agg_blocks
        self.primary_party_id = primary_party_id
        self.k = k
        self.rep_noise = rep_noise
        self.max_rep_norm = max_rep_norm
        self.enable_pe = enable_pe
        self.enable_dm = enable_dm

        self.n_parties = len(key_dims)
        if activation.lower() == 'relu':
            self.activation = nn.ReLU()
        elif activation.lower() == 'gelu':
            self.activation = nn.GELU()
        elif activation.lower() == 'leakyrelu':
            self.activation = nn.LeakyReLU()
        else:
            raise ValueError(f"Unsupported activation function: {activation}, "
                             f"expecting one of ['relu', 'gelu', 'leakyrelu']")

        # positional encoding
        self.positional_encodings = nn.ModuleList()
        for key_dim, data_dim in zip(key_dims, data_dims):
            self.positional_encodings.append(LearnableFourierPositionalEncoding(G=1, M=key_dim, F_dim=data_embed_dim,
                                                                                H_dim=2*data_embed_dim, D=data_embed_dim,
                                                                                gamma=0.05))

        # self.pe_dim = data_embed_dim // 2
        # start_data_embed_dim = data_embed_dim - self.pe_dim
        # self.positional_encoding = LearnableFourierPositionalEncoding(G=1, M=key_dims[0], F_dim=self.data_embed_dim,
        #                                                               H_dim=self.data_embed_dim * 2,
        #                                                               D=self.data_embed_dim, gamma=0.05)

        # dynamic mask layer
        self.dynamic_mask_layers = nn.ModuleList()
        for i, (key_dim, data_dim) in enumerate(zip(key_dims, data_dims)):
            if not self.enable_dm:
                self.dynamic_mask_layers.append(torch.nn.Identity())
                continue

            if i == primary_party_id:
                layer = nn.Identity()
            else:
                layer = nn.Sequential(
                    nn.Linear(key_dim * k, 2 * key_dim * k),
                    self.activation,
                    nn.Linear(2 * key_dim * k, k),
                )
            self.dynamic_mask_layers.append(layer)

        # X embedding layer for each party
        self.data_embeddings = nn.ModuleList()
        for key_dim, data_dim in zip(key_dims, data_dims):
            # self.data_embeddings.append(nn.Linear(data_dim, start_data_embed_dim))
            self.data_embeddings.append(nn.Linear(key_dim + data_dim, data_embed_dim))  # sum embedding

        # self-attention for each party
        # Query: key_embed, Key: key_embed, Value: data
        self.self_attns = nn.ModuleList()
        for i in range(self.n_parties):
            if i == self.primary_party_id:
                # self.self_attns.append(torch.nn.Identity())
                self.self_attns.append(SelfAttnChain(data_embed_dim, num_heads, dropout,
                                                        self.activation, n_local_blocks))
            else:
                self.self_attns.append(SelfAttnChain(data_embed_dim, num_heads, dropout,
                                                     self.activation, n_local_blocks))

        # self.self_attns[0].visualize_attention()

        self.agg_attn = AggAttnChain(data_embed_dim, num_heads, dropout,
                                     self.activation, n_agg_blocks)

        # final output dense layer
        self.output_layer = nn.Linear(data_embed_dim, out_dim)

    def check_args(self, key_Xs):
        for key_X in key_Xs:
            if not isinstance(key_X, Sequence):
                raise ValueError(f"Each element of the input should be a sequence, but got {type(key_X)}")
            if len(key_X) != 2:
                raise ValueError(f"Each element of the input should be a sequence of length 2, first is the key and "
                                 f"second is the feature tensor, but got a sequence of length {len(key_X)}")
        for i, ((key, X), key_dim, X_dim) in enumerate(zip(key_Xs, self.key_dims, self.data_dims)):
            if not isinstance(key, torch.Tensor):
                raise ValueError(f"The {i}-th key should be a tensor, but got {type(key)}")
            if not isinstance(X, torch.Tensor):
                raise ValueError(f"The {i}-th feature tensor should be a tensor, but got {type(X)}")
            if key.shape[0] != X.shape[0]:
                raise ValueError(f"The first dimension of the {i}-th key tensor {key.shape[0]} should be the same as "
                                 f"the first dimension of the feature tensor {X.shape[0]} of the same party.")
            if key_dim > 0 and key.shape[-1] != key_dim:
                raise ValueError(f"The #features of the {i}-th key tensor {key.shape[-1]} should be the same as "
                                 f"the key dimension {key_dim} of the same party.")
            if X_dim > 0 and X.shape[-1] != X_dim:
                raise ValueError(f"The #features of the {i}-th feature tensor {X.shape[-1]} should be the same as "
                                 f"the feature dimension {X_dim} of the same party.")

    def forward(self, key_Xs, visualize=False):
        """
        Forward function of FedTrans
        :param key_Xs: [(k1, X1), ..., (kn, Xn)], where n is the number of parties. ki is the key and Xi is the
                       feature tensor of the i-th party. ki and Xi are supposed to be tensors with the same first
                       dimension. The size of ki is supposed to be (batch_size, key_dim) and the size of Xi is
                       supposed to be (batch_size, data_dim). The batch_size of ki and Xi are regarded as sequence
                       length of the self-attention (i.e., the self-attention takes non-batched data).
        :return:
        """
        self.check_args(key_Xs)

        keys = []
        Xs = []
        for key_X in key_Xs:
            if len(key_X[0].shape) == 2:
                keys.append(key_X[0].unsqueeze(1))
                Xs.append(key_X[1].unsqueeze(1))
            else:
                keys.append(key_X[0])
                Xs.append(key_X[1])

        # dynamic masking
        masks = []
        for i in range(self.n_parties):
            if not self.enable_dm:
                mask = None
            elif i == self.primary_party_id:
                mask = None
            else:
                mask = self.dynamic_mask_layers[i](keys[i].reshape(keys[i].shape[0], -1))
            masks.append(mask)

        # X embedding and positional encoding
        key_Xs = [torch.cat([keys[i], Xs[i]], dim=-1) for i in range(self.n_parties)]
        key_X_embeds = []
        for i in range(self.n_parties):
            key_X_embed = self.data_embeddings[i](key_Xs[i])
            key_dim = self.key_dims[i]
            if self.enable_pe:
                pe = self.positional_encodings[i](keys[i].view(-1, 1, key_dim))
                # pe = self.positional_encoding(keys[i].view(-1, 1, key_dim))  # share the same positional encoding

                key_X_embed += pe.view(key_X_embed.shape)
                # pe = pe.view(key_X_embed.shape[0], -1, self.pe_dim)
                # key_X_embed = torch.cat([key_X_embed, pe], dim=-1)  # concatenate the positional encoding

            key_X_embeds.append(key_X_embed)

        # Self-attention of secondary parties, keep the primary party's key and data unchanged
        key_X_embeds = [self.self_attns[i](key_X_embeds[i], need_weights=visualize, key_padding_mask=masks[i])
                        if i != self.primary_party_id else key_X_embeds[i]
                        for i in range(self.n_parties)]

        # # debug
        # self.self_attns[1].visualize_attention()

        # sum keys and data respectively, the result should be of the same size as the single-party cut layer keys and
        # data. All the dimensions of the cut layer keys and data should be the same
        # todo: replace with cross-party secure multi-party aggregation
        primary_key_X_embed = key_X_embeds[self.primary_party_id]
        # key_X_embeds[self.primary_party_id] = primary_key_X_embed.repeat(1, self.k, 1)
        secondary_key_X_embeds = [key_X_embeds[i] for i in range(self.n_parties) if i != self.primary_party_id]

        # dropout self.dropout number of parties
        if self.training and not np.isclose(self.party_dropout, 0):
            n_drop_parties = int((self.n_parties - 1) * self.party_dropout)
            drop_party_indices = torch.randperm(self.n_parties - 1)[:n_drop_parties]
            for i in range(self.n_parties - 1):
                if i in drop_party_indices:
                    self.self_attns[i].requires_grad_(False)
                else:
                    self.self_attns[i].requires_grad_(True)
            drop_mask = torch.ones(self.n_parties - 1)
            drop_mask[drop_party_indices] = 0.
            drop_mask = drop_mask.to(primary_key_X_embed.device)
            secondary_key_X_embeds = [drop_mask[i] * secondary_key_X_embeds[i]
                                      for i in range(self.n_parties - 1)]
        else:
            n_drop_parties = 0

        if self.rep_noise is not None and self.max_rep_norm is not None:
            # norm cut for each secondary party, ensure the total norm is less than max_rep_norm
            max_rep_norm_per_party = self.max_rep_norm / (self.n_parties - 1)
            secondary_reps = []
            for secondary_key_X_embed in secondary_key_X_embeds:
                cut_layer_i = torch.tanh(secondary_key_X_embed)  # reduce norm by activation
                cut_layer_i_flat = cut_layer_i.reshape(cut_layer_i.shape[0], -1)
                per_sample_norm = torch.norm(cut_layer_i_flat, dim=1, p=2)
                clip_coef = max_rep_norm_per_party / (per_sample_norm + 1e-6)
                clip_coef_clamped = torch.clamp(clip_coef, max=1)
                rep = cut_layer_i_flat * clip_coef_clamped.unsqueeze(-1)
                secondary_reps.append(rep)

            # sum of the cut layers
            cut_layer_key_X_flat = torch.sum(torch.stack(secondary_reps), dim=0)

            noise = (torch.normal(0, self.rep_noise * self.max_rep_norm, cut_layer_key_X_flat.shape)
                     .to(cut_layer_key_X_flat.device))
            cut_layer_key_X_flat += noise
            cut_layer_key_X = cut_layer_key_X_flat.reshape(cut_layer_key_X_flat.shape[0], -1, self.data_embed_dim)
        else:
            cut_layer_key_X = torch.sum(torch.stack(secondary_key_X_embeds), dim=0)

        # primary party aggregates the embedding with primary_attn from summed cut keys and data
        agg_key_X_embed = self.agg_attn(primary_key_X_embed,
                                        (cut_layer_key_X) / (self.n_parties - n_drop_parties - 1),
                                        need_weights=visualize, key_padding_mask=masks[self.primary_party_id])

        # # debug: ignore secondary parties
        # agg_key_X_embed = self.agg_attn(primary_key_X_embed, primary_key_X_embed, need_weights=visualize)

        # output layer
        output = self.output_layer(agg_key_X_embed.reshape(agg_key_X_embed.shape[0], -1))
        if self.out_activation is not None:
            output = self.out_activation(output)

        return output

    def average_pe_(self):
        """
        Average all the trainable parameters of the positional encoding
        :return:
        """
        mean_pe = self.positional_encodings[0]
        for name, param in mean_pe.named_parameters():
            for pe in self.positional_encodings[1:]:
                param.data += pe.state_dict()[name]
            param.data /= len(self.positional_encodings)

        for pe in self.positional_encodings[1:]:
            pe.load_state_dict(mean_pe.state_dict())

    @torch.no_grad()
    def visualize_positional_encoding(self, pivot=(0, 0), sample_size=50000, scale=1.0, dataset=None,
                                      save_path=None, device='cpu'):
        """
        Visualize the positional encoding
        :return: None
        """
        self.eval()
        plt.rcParams.update({'font.size': 24})
        if not isinstance(pivot, torch.Tensor):
            pivot = torch.tensor(pivot).float().to(device)

        # keys = D.Normal(pivot, scale).sample(torch.Size([sample_size]))
        keys = D.Uniform(pivot - 4 * scale, pivot + 4 * scale).sample(torch.Size([sample_size])).to(device)

        # get the positional encoding
        # multiple subfigures in the same row
        fig, axs = plt.subplots(1, len(self.positional_encodings), figsize=(25, 10))
        for i, pe_layer in enumerate(self.positional_encodings):
            key_encoding = pe_layer(keys.view(-1, self.key_dims[i], 1))
            pivot_encoding = pe_layer(pivot.view(-1, self.key_dims[i], 1))
            dist_enc = torch.norm(key_encoding - pivot_encoding, dim=-1).detach().cpu().numpy()

            # plot the positional encoding
            if pivot.detach().cpu().numpy().size > 2:
                raise NotImplementedError
            keys_array = keys.detach().cpu().numpy()

            axs[i].set_title(f"Party {i}")
            sc = axs[i].scatter(keys_array[:, 0], keys_array[:, 1], c=dist_enc, cmap='viridis_r', s=30)
            fig.colorbar(sc, ax=axs[i])
        plt.tight_layout()
        if save_path is None:
            plt.show()
        else:
            plt.savefig(save_path)
        plt.close()

    @torch.no_grad()
    def save_pe_inout(self, dataloader, save_dir, device='cpu'):
        """
        Save the positional encoding of the input and output
        :param dataloader: the dataloader to be used to generate the input and output positional encoding
        :param save_dir: the path to save the positional encoding
        :param device: the device of the model
        :return: None
        """
        self.eval()
        # get the positional encoding
        pe_in_all_parties = [[] for _ in range(self.n_parties)]
        pe_out_all_parties = [[] for _ in range(self.n_parties)]

        for key_Xs, _ in dataloader:
            keys = [X[0] for X in key_Xs]  # keys for each party
            for i in range(self.n_parties):
                if len(self.positional_encodings) == 0:
                    pe_layer = self.positional_encoding
                else:
                    pe_layer = self.positional_encodings[i]
                key = keys[i].to(device)
                key_encoding = pe_layer(key.view(-1, 1, self.key_dims[i])).view(key.shape[0], -1, self.data_embed_dim)
                pe_in_all_parties[i].append(key.detach().cpu())
                pe_out_all_parties[i].append(key_encoding.detach().cpu())

        os.makedirs(save_dir, exist_ok=True)

        for i in range(self.n_parties):
            pe_in_all_parties[i] = torch.cat(pe_in_all_parties[i], dim=0)
            torch.save(pe_in_all_parties[i], f"{save_dir}/pe-in-party{i}.pt")
            print(f"Saved tensor of size {pe_in_all_parties[i].shape} to {save_dir}/pe-in-party{i}.pt")
            pe_out_all_parties[i] = torch.cat(pe_out_all_parties[i], dim=0)
            torch.save(pe_out_all_parties[i], f"{save_dir}/pe-out-party{i}.pt")
            print(f"Saved tensor of size {pe_out_all_parties[i].shape} to {save_dir}/pe-out-party{i}.pt")

    def visualize_attention(self):
        """
        Visualize the attention
        :return: None
        """
        pass

    def visualize_dynamic_mask(self, key_Xs, key_scalers=None, title=None, save_path=None):
        """
        Visualize the dynamic mask
        Args:
            key_Xs: the input tensors on each party for visualization

        Returns: None

        """
        keys = []
        Xs = []
        for key_X in key_Xs:
            if len(key_X[0].shape) == 2:
                keys.append(key_X[0].unsqueeze(1))
                Xs.append(key_X[1].unsqueeze(1))
            else:
                keys.append(key_X[0])
                Xs.append(key_X[1])

        # dynamic masking
        masks = []
        for i in range(self.n_parties):
            if not self.enable_dm or i == self.primary_party_id:
                mask = None
            else:
                mask = self.dynamic_mask_layers[i](keys[i].view(keys[i].shape[0], -1))
            masks.append(mask)

        # obtain the original keys
        real_masks = []
        original_keys = []
        if key_scalers is not None:
            for i in range(self.n_parties):
                key_X_numpy = torch.cat([keys[i], Xs[i]], dim=-1).detach().cpu().numpy()
                length = key_X_numpy.shape[1]
                original_key_X = key_scalers[i].inverse_transform(key_X_numpy.reshape(-1, key_X_numpy.shape[-1]))
                original_key = original_key_X[:, :self.key_dims[i]]
                original_keys.append(original_key.reshape(keys[i].shape[0], length, -1))

                if i != self.primary_party_id:
                    mask = masks[i].detach().cpu().numpy()
                    mask_scaled = mask / np.mean(key_scalers[i].scale_[:self.key_dims[i]])
                    real_masks.append(mask_scaled)
                else:
                    real_masks.append(None)
        else:
            real_masks = [mask.detach().cpu().numpy() if mask is not None else None for mask in masks]
            original_keys = [keys[i].detach().cpu().numpy() for i in range(self.n_parties)]

        # visualize the dynamic mask
        key_dim = self.key_dims[0]
        if any([kd != key_dim for kd in self.key_dims]):
            raise NotImplementedError("Visualization over different key dimensions is not implemented yet")

        if self.n_parties > 2:
            # use PCA to reduce the dimension to 2
            pca = PCA(n_components=2)
            pca.fit(np.concatenate([key.reshape(-1, key_dim) for key in original_keys], axis=0))
            original_keys = [pca.transform(original_key.reshape(-1, key_dim)).reshape(original_key.shape[0], -1, 2)
                             for original_key in original_keys]

        # multiple subfigures in the same row
        n_samples = original_keys[0].shape[0]


        # concat keys from secondary parties, duplicate the primary party's key
        primary_party_key = original_keys[self.primary_party_id]
        secondary_party_keys = np.concatenate([original_keys[i]
                                          for i in range(self.n_parties) if i != self.primary_party_id], axis=1)
        secondary_masks = np.concatenate([real_masks[i]
                                          for i in range(self.n_parties) if i != self.primary_party_id], axis=1)

        primary_party_id = 0
        secondary_party_id = 1
        original_keys = [primary_party_key, secondary_party_keys]

        for i in range(n_samples):
            # mask = masks[secondary_party_id][i].detach().cpu().numpy()
            mask = secondary_masks[i]
            primary_key = original_keys[primary_party_id][i]
            secondary_key = original_keys[secondary_party_id][i]

            plt.figure(figsize=(6, 4))

            # Normalize mask for alpha values: higher mask values are more opaque
            alphas = np.interp(mask, [mask.min(), mask.max()], [0.15, 1])
            # colors[:, -1] = np.interp(mask, [mask.min(), mask.max()], [1, 1])

            scatter = plt.scatter(secondary_key[:, 0], secondary_key[:, 1], c=mask, cmap='coolwarm',
                                  label='Keys - Secondary', s=2, alpha=alphas)
            plt.scatter(primary_key[:, 0], primary_key[:, 1], marker='*', color='r', label='Key - Primary', s=50)
            # use 2 decimal for x and y ticks
            plt.gca().xaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
            plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
            plt.colorbar(scatter, label='Dynamic Mask Value')
            plt.legend()
            if title is not None:
                plt.title(title)

            if save_path is not None:
                plt.savefig(save_path + f"-{i}.png", bbox_inches='tight', pad_inches=0, dpi=600)
                plt.close()
            else:
                plt.show()
            pass
        pass


    def save(self, path):
        torch.save(self.state_dict(), path)

    def save_pkl(self, path):
        torch.save(self, path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
        return self

    @classmethod
    def load_pkl(cls, path):
        model = torch.load(path)
        return model
