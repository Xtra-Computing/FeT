import abc
from typing import Sequence, Union
from copy import deepcopy
from collections import defaultdict
from functools import wraps

from cachetools import cached, LRUCache
from cachetools.keys import hashkey
import numpy as np
import nmslib


class DataSampler(abc.ABC):
    return_multi = False
    def __init__(self, keys: Sequence, primary_party_id: int = 0, seed=None, **kwargs):
        """
        Sample the indices from a sequence of dataset according to keys
        :param keys: keys of each dataset
        :param primary_party_id: primary party id
        """
        self.keys = deepcopy(keys)
        self.primary_party_id = primary_party_id
        self.seed = seed
        np.random.seed(seed)
        self.n_datasets = len(keys)

    def initialize(self, keys, **kwargs):
        pass

    @abc.abstractmethod
    def sample(self, p_id):
        """
        Sample one or multiple secondary indices from one primary ID
        """
        raise NotImplementedError


class SimSampler(DataSampler, abc.ABC):
    def __init__(self, keys: Sequence, primary_party_id: int = 0, seed=None, indices=None, keep_primary=False,
                 **kwargs):
        """
        Sample the indices from a sequence of dataset according to keys
        :param keys: keys of each party
        :param primary_party_id: primary party id
        :param seed: random see
        :param indices: If None, initialize the indices from keys. Otherwise, use the given indices.
        :param kwargs: other parameters for self.initialize
        """
        super().__init__(keys, primary_party_id, seed, **kwargs)
        self.indices = None  # indices for querying
        self.keep_primary = keep_primary
        if indices is None:
            self.initialize(keys, **kwargs)
        else:
            self.indices = indices

    def initialize(self, keys, method='hnsw', space='l2'):
        self.indices = []
        for i, key in enumerate(keys):
            if (not self.keep_primary) and i == self.primary_party_id:
                self.indices.append(None)  # placeholder for primary party
                continue

            index = nmslib.init(method=method, space=space, data_type=nmslib.DataType.DENSE_VECTOR)
            index.addDataPointBatch(key)
            index.createIndex()
            self.indices.append(index)


class Top1Sampler(SimSampler):
    def __init__(self, keys: Sequence, primary_party_id: int = 0, seed=None, indices=None, **kwargs):
        super().__init__(keys, primary_party_id, seed, indices, **kwargs)

    def sample(self, idx):
        """
        Sample most similar secondary indices from one primary ID (pid)
        :param idx: [Int] sample ID on the primary party
        :return: List[Int] sampled indices for all parties. The i-th element is the sampled index on the i-th party.
                The total length should be `self.n_datasets`.
        """

        primary_key = self.keys[self.primary_party_id][idx]

        sampled_indices = []
        for data_id in range(self.n_datasets):
            if data_id == self.primary_party_id:
                sampled_indices.append(idx)
            else:
                nbr_ids, dist = self.indices[data_id].knnQuery(primary_key, k=1)
                sampled_indices.append(nbr_ids[0])

        return sampled_indices


class TopkUniformSampler(SimSampler):
    def __init__(self, keys: Sequence, ks: Union[Sequence, int], primary_party_id: int = 0, seed=None, **kwargs):
        super().__init__(keys, primary_party_id, seed, **kwargs)
        if isinstance(ks, int):
            self.ks = [ks] * self.n_datasets
        else:
            self.ks = ks

        if len(self.ks) != self.n_datasets:
            raise ValueError(f"The length of ks {len(self.ks)} should be the same as the number of parties "
                             f"{self.n_datasets}")

    def sample(self, idx):
        """
        Uniformly sample secondary indices among topk-most-similar ones from one primary ID (pid)
        :param idx: [Int] sample ID on the primary party
        :return: List[Int] sampled indices for all parties. The i-th element is the sampled index on the i-th party.
                The total length should be `self.n_datasets`.
        """
        primary_key = self.keys[self.primary_party_id][idx]

        sampled_indices = []
        for data_id in range(self.n_datasets):
            if data_id == self.primary_party_id:
                sampled_indices.append(idx)
            else:
                nbr_ids, dist = self.indices[data_id].knnQuery(primary_key, k=self.ks[data_id])
                sampled_indices.append(np.random.choice(nbr_ids))

        return sampled_indices


class TopkSimAsProbSampler(SimSampler):
    def __init__(self, keys: Sequence, ks: Union[Sequence, int], primary_party_id: int = 0, seed=None, **kwargs):
        super().__init__(keys, primary_party_id, seed, **kwargs)
        if isinstance(ks, int):
            self.ks = [ks] * self.n_datasets
        else:
            self.ks = ks

        if len(self.ks) != self.n_datasets:
            raise ValueError(f"The length of ks {len(self.ks)} should be the same as the number of parties "
                             f"{self.n_datasets}")

    def sample(self, idx):
        """
        Sample secondary indices among topk-most-similar ones from one primary ID (pid). The probability of sampling
        is proportional to the exponential negative distance.
        :param idx: [Int] sample ID on the primary party
        :return: List[Int] sampled indices for all parties. The i-th element is the sampled index on the i-th party.
                The total length should be `self.n_datasets`.
        """
        primary_key = self.keys[self.primary_party_id][idx]

        sampled_indices = []
        for data_id in range(self.n_datasets):
            if data_id == self.primary_party_id:
                sampled_indices.append(idx)
            else:
                nbr_ids, dist = self.indices[data_id].knnQuery(primary_key, k=self.ks[data_id])
                # make dist as probability
                scaled_dist = np.exp(-dist) / np.sum(np.exp(-dist))
                sampled_indices.append(np.random.choice(nbr_ids, p=scaled_dist))

        return sampled_indices

# def diskcache(func):
#     def wrapper(self, idx):
#         key = idx
#         if self.cache_path is not None:
#             if key in self.cache:
#                 return self.cache[key]
#             else:
#                 result = func(self, idx)
#                 self.cache[key] = result
#                 return result
#         else:
#             return func(self, idx)
#     return wrapper


def conditional_cached(cache_key, cache=LRUCache(maxsize=10**7)):
    def decorator(func):
        if cache_key is not None:
            # Apply caching if cache_key is not None
            return cached(cache=cache, key=lambda self, idx: hashkey((cache_key, idx)))(func)
        else:
            # Return the original function unmodified if cache_key is None
            @wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            return wrapper
    return decorator


class TopkSampler(SimSampler):
    return_multi = True     # return multiple indices for secondary parties
    def __init__(self, keys: Sequence, ks: Union[Sequence, int], primary_party_id: int = 0, seed=None, indices=None,
                 multi_primary=False, sample_rate_before_topk=None, cache_key=None, **kwargs):
        super().__init__(keys, primary_party_id, seed, method='brute_force', space='l2', indices=indices,
                         keep_primary=multi_primary, **kwargs)
        self.multi_primary = multi_primary
        self.sample_rate_before_topk = sample_rate_before_topk
        if isinstance(ks, int):
            self.ks = [ks] * self.n_datasets
        else:
            self.ks = ks

        if len(self.ks) != self.n_datasets:
            raise ValueError(f"The length of ks {len(self.ks)} should be the same as the number of parties "
                             f"{self.n_datasets}")
        self.cache_key = cache_key
        # self.cache_path = cache_path
        # if self.cache_path is not None:
        #     self.cache = Cache(self.cache_path)

    # @conditional_cached(cache_key=lambda self: self.cache_key)
    def sample(self, idx):
        """
        Sample secondary indices among topk-most-similar ones from one primary ID (pid). The probability of sampling
        is proportional to the exponential negative distance.
        :param idx: [Int] sample ID on the primary party
        :param multi_primary: [Bool] whether to return multiple indices for primary party
        :return: List[List[Int]] sampled indices for all parties. The i-th element is the sampled index set on the i-th
                 The total length should be `self.n_datasets`. The length of the i-th element is `self.ks[i]`.
        """

        primary_key = self.keys[self.primary_party_id][idx]

        sampled_indices = []
        for data_id in range(self.n_datasets):
            if (not self.multi_primary) and data_id == self.primary_party_id:
                sampled_indices.append(idx)
                continue

            if self.sample_rate_before_topk:
                index = nmslib.init(method='brute_force', space='l2', data_type=nmslib.DataType.DENSE_VECTOR)
                sample_size = int(self.sample_rate_before_topk * len(self.keys[data_id]))
                sample_idx = np.random.choice(len(self.keys[data_id]), sample_size, replace=False)
                sample_key = self.keys[data_id][sample_idx]
                index.addDataPointBatch(sample_key)
                index.createIndex()
                nbr_ids_i, dist = index.knnQuery(primary_key, k=self.ks[data_id])
                nbr_ids = sample_idx[nbr_ids_i]
                if len(nbr_ids) < self.ks[data_id]:
                    # repeat nbr_ids if not enough
                    nbr_ids = np.concatenate([nbr_ids, np.random.choice(nbr_ids, self.ks[data_id] - len(nbr_ids))])
            else:
                nbr_ids, dist = self.indices[data_id].knnQuery(primary_key, k=self.ks[data_id])

            # # debug: can be optimized. to shuffle the secondary indices
            # if data_id != self.primary_party_id:
            #     np.random.shuffle(nbr_ids)

            # # debug: can be optimized. to ensure the first element in primary neighbors is itself
            # if data_id == self.primary_party_id:
            #     if idx in nbr_ids:
            #         nbr_ids = np.concatenate([np.array([idx]), nbr_ids[nbr_ids != idx]])
            #     else:
            #         nbr_ids = np.concatenate([np.array([idx]), nbr_ids[:-1]])

            sampled_indices.append(nbr_ids)

        return sampled_indices


class RandomSampler:
    return_multi = True     # return multiple indices for secondary parties
    def __init__(self, sizes, n_samples, primary_party_id=0, seed=None):
        self.sizes = sizes
        self.n_samples = n_samples
        self.primary_party_id = primary_party_id
        self.seed = seed
        np.random.seed(seed)

    def sample(self, idx):
        """
        Sample one or multiple secondary indices from one primary ID
        :param p_id: primary party ID
        :return: List[List[Int]] sampled indices for all parties. The i-th element is the sampled index set on the i-th
                 The total length should be `self.n_datasets`. The length of the i-th element is `self.ks[i]`.
        """
        sampled_indices = []
        for data_id in range(len(self.sizes)):
            if data_id == self.primary_party_id:
                sampled_indices.append(idx)
                continue
            sampled_indices.append(np.random.choice(self.sizes[data_id], self.n_samples, replace=False))

        return sampled_indices
