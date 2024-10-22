from numbers import Real
import warnings

import cachetools

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import torch
import torch.linalg
from torchmetrics.functional import spearman_corrcoef

from pymoo.algorithms.soo.nonconvex.brkga import BRKGA
from pymoo.core.duplicate import ElementwiseDuplicateElimination
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.termination.default import DefaultSingleObjectiveTermination
from pymoo.core.problem import StarmapParallelization
from multiprocessing.pool import ThreadPool

from .FeatureEvaluator import ImportanceEvaluator, CorrelationEvaluator


class ImportanceSplitter:
    def __init__(self, num_parties, weights=1, seed=None):
        """
        Split a 2D dataset by feature importance under dirichlet distribution (assuming the features are independent).
        :param num_parties: [int] number of parties
        :param weights: [int | list with size num_parties]
                        If weights is an int, the weight of each party is the same.
                        If weights is an array, the weight of each party is the corresponding element in the array.
                        The weights indicate the expected sum of feature importance of each party.
                        Meanwhile, larger weights mean less bias on the feature importance.
        :param seed: [int] random seed
        """
        self.num_parties = num_parties
        self.weights = weights
        self.seed = seed
        np.random.seed(seed)
        if isinstance(self.weights, Real):
            self.weights = [self.weights for _ in range(self.num_parties)]

        self.check_params()

    def check_params(self):
        """
        Check if the parameters are valid
        """
        assert len(self.weights) == self.num_parties, "The length of weights should equal to the number of parties"

    def split_indices(self, X, allow_empty_party=False):
        """
        Split the indices of X by feature importance.
        :param allow_empty_party: [bool] whether to allow parties with zero features
        :param X: [np.ndarray] 2D dataset
        :return: [list] list of indices of each party
        """
        # Generate the probabilities of being assigned to each party
        # All the features share the same ratio
        probs = np.random.dirichlet(self.weights)

        if allow_empty_party:
            # Assign each feature to a party
            party_to_feature = [[] for _ in range(self.num_parties)]
            party_ids = np.random.choice(self.num_parties, size=X.shape[1], p=probs)
            for feature_id in range(X.shape[1]):
                party_id = party_ids[feature_id]
                party_to_feature[party_id].append(feature_id)
        else:
            # uniform-randomly select one feature for each party
            preassigned_feature_ids = np.random.choice(X.shape[1], size=self.num_parties, replace=False)
            party_to_feature = [[feature_id] for feature_id in preassigned_feature_ids]

            # Assign the remaining features to the parties
            preassigned_feature_id_set = set(preassigned_feature_ids)
            party_ids = np.random.choice(self.num_parties, size=X.shape[1], p=probs)
            for feature_id in range(X.shape[1]):
                if feature_id not in preassigned_feature_id_set:
                    party_id = party_ids[feature_id]
                    party_to_feature[party_id].append(feature_id)

            # no empty party
            assert np.count_nonzero([len(party) for party in party_to_feature]) == self.num_parties, \
                "There should be no empty party"
            assert np.sum([len(party) for party in party_to_feature]) == X.shape[1], \
                "The number of features should be the same as the number of columns of X"

        return party_to_feature

    def split(self, X, *args, indices=None, allow_empty_party=False):
        """
        Split X by feature importance.
        :param allow_empty_party: [bool] whether to allow parties with zero features
        :param X: [np.ndarray] 2D dataset
        :param args: [np.ndarray] other datasets with the same number of columns as X (X1, X2, ..., Xn)
        :param indices: [list] indices of features on each party. If not given, the indices will be generated randomly.
        :return: (X1, X2, ..., Xn) [np.ndarray, ...] where n is the number of parties
        """
        if indices is None:
            party_to_feature = self.split_indices(X, allow_empty_party=allow_empty_party)
        else:
            party_to_feature = indices

        # Split the dataset according to party_to_feature
        Xs = []
        for party_id in range(self.num_parties):
            Xs.append(X[:, party_to_feature[party_id]])

        # Split the other datasets
        other_Xs_list = []
        for other_X in args:
            assert other_X.shape[1] == X.shape[1], "The number of columns of other datasets should be the same as X"
            other_Xs = []
            for party_id in range(self.num_parties):
                other_Xs.append(other_X[:, party_to_feature[party_id]])
            other_Xs_list.append(other_Xs)

        if len(other_Xs_list) == 0:
            return tuple(Xs)
        else:
            return tuple(Xs), *tuple(other_Xs_list)


class CorrelationSplitter:

    def __init__(self, num_parties: int, evaluator: CorrelationEvaluator = None, seed=None, gpu_id=None, n_jobs=1):
        """
        Split a 2D dataset by feature correlation (assuming the features are equally important).
        :param num_parties: [int] number of parties
        :param evaluator: [CorrelationEvaluator] the evaluator to evaluate the correlation
        :param seed: [int] random seed
        :param gpu_id: [int] GPU id
        """
        self.num_parties = num_parties
        self.evaluator = evaluator
        self.seed = seed
        self.gpu_id = gpu_id
        if self.gpu_id is not None:
            assert evaluator.gpu_id == self.gpu_id, "The gpu_id of the evaluator should be the same as the gpu_id of the splitter"
            self.device = torch.device("cuda:{}".format(self.gpu_id))
        else:
            self.device = torch.device("cpu")
            warnings.warn("The device is set to CPU. This may cause the performance to be slow.")

        self.pool = ThreadPool(n_jobs)
        self.runner = StarmapParallelization(self.pool.starmap)

        # split result of the last call of fit()
        # self.corr = None      # use evaluator.corr instead
        # self.n_features_on_party = None   # use evaluator.n_features_on_party instead
        self.min_mcor = None
        self.max_mcor = None

        # best split result of all calls of split()
        self.best_mcor = None
        self.best_error = None
        self.best_feature_per_party = None
        self.best_permutation = None

    @staticmethod
    def sort_order_by_party(order, n_features_on_party):
        order_cut_points = np.cumsum(n_features_on_party)
        order_cut_points = np.insert(order_cut_points, 0, 0)
        sorted_order_by_party = []
        for i in range(1, len(order_cut_points)):
            sorted_order_by_party.append(tuple(sorted(order[order_cut_points[i - 1]:order_cut_points[i]])))
        return sorted_order_by_party

    # Nested class for BRKGA solver: duplicate elimination of permutations
    class DuplicationElimination(ElementwiseDuplicateElimination):
        def is_equal(self, perm_a, perm_b):
            return perm_a.get("hash") == perm_b.get("hash")

    # Nested class for BRKGA solver: max-mcor problem definition
    class CorrMaxProblem(ElementwiseProblem):
        def __init__(self, corr, n_features_on_party, evaluator: CorrelationEvaluator = None, runner=None):
            super().__init__(n_var=corr.shape[1], n_obj=1, n_constr=0, xl=-1, xu=1, elementwise_runner=runner)
            self.corr = corr
            self.n_features_on_party = n_features_on_party
            self.evaluator = evaluator

        # @cachetools.cached(cache=cachetools.TTLCache(maxsize=1000, ttl=60),
        #                    key=lambda self, corr, order: hash(tuple(
        #                        CorrelationSplitter.sort_order_by_party(order, self.n_features_on_party))))
        def corr_score(self, corr, order):
            corr_perm = corr[order, :][:, order]
            return self.evaluator.overall_corr_score(corr_perm, self.n_features_on_party)

        def _evaluate(self, x, out, *args, **kwargs):
            order = np.argsort(x)
            out['F'] = -self.corr_score(self.corr, order)
            out['order'] = order
            # sorted_order_by_party = CorrelationSplitter.sort_order_by_party(order, self.n_features_on_party)
            out['hash'] = hash(tuple(order))

    # Nested class for BRKGA solver: min-mcor problem definition
    class CorrMinProblem(ElementwiseProblem):
        def __init__(self, corr, n_features_on_party, evaluator: CorrelationEvaluator = None, runner=None):
            super().__init__(n_var=corr.shape[1], n_obj=1, n_constr=0, xl=-1, xu=1, elementwise_runner=runner)
            self.corr = corr
            self.n_features_on_party = n_features_on_party
            self.evaluator = evaluator

        # @cachetools.cached(cache=cachetools.TTLCache(maxsize=1000, ttl=60),
        #                    key=lambda self, corr, order: hash(tuple(
        #                        CorrelationSplitter.sort_order_by_party(order, self.n_features_on_party))))
        def corr_score(self, corr, order):
            corr_perm = corr[order, :][:, order]
            return self.evaluator.overall_corr_score(corr_perm, self.n_features_on_party)

        def _evaluate(self, x, out, *args, **kwargs):
            order = np.argsort(x)
            out['F'] = self.corr_score(self.corr, order)
            out['order'] = order
            # sorted_order_by_party = CorrelationSplitter.sort_order_by_party(order, self.n_features_on_party)
            out['hash'] = hash(tuple(order))

    # Nested class for BRKGA solver: best-matched-mcor problem definition
    class CorrBestMatchProblem(ElementwiseProblem):
        def __init__(self, corr, n_features_on_party, beta, min_mcor, max_mcor, evaluator: CorrelationEvaluator = None,
                     runner=None):
            super().__init__(n_var=corr.shape[1], n_obj=1, n_constr=0, xl=-1, xu=1, elementwise_runner=runner)
            assert min_mcor < max_mcor, f"min_mcor {min_mcor} should be smaller than max_mcor {max_mcor}"
            self.corr = corr
            self.n_features_on_party = n_features_on_party
            self.beta = beta
            self.max_mcor = max_mcor
            self.min_mcor = min_mcor
            self.evaluator = evaluator

        # @cachetools.cached(cache=cachetools.TTLCache(maxsize=10000, ttl=60),
        #                    key=lambda self, corr, order: hash(tuple(
        #                        CorrelationSplitter.sort_order_by_party(order, self.n_features_on_party))))
        def corr_score(self, corr, order):
            corr_perm = corr[order, :][:, order]
            return self.evaluator.overall_corr_score(corr_perm, self.n_features_on_party)

        def _evaluate(self, x, out, *args, **kwargs):
            order = np.argsort(x)
            mcor = self.corr_score(self.corr, order)
            target_mcor = self.beta * self.max_mcor + (1 - self.beta) * self.min_mcor

            out['F'] = np.abs(mcor - target_mcor)
            out['mcor'] = mcor
            out['order'] = order
            # sorted_order_by_party = CorrelationSplitter.sort_order_by_party(order, self.n_features_on_party)
            out['hash'] = hash(tuple(order))

    @staticmethod
    def split_num_features_equal(n_features, n_parties):
        """
        Split n_features into n_parties equally. The first party may have more features.
        :param n_features: (int) number of features
        :param n_parties: (int) number of parties
        :return: (list) number of features on each party
        """
        n_features_on_party = [n_features // n_parties] * n_parties
        n_features_on_party[0] += n_features % n_parties
        assert sum(n_features_on_party) == n_features
        return n_features_on_party

    def check_fit_data(self):
        """
        Check if the required members are calculated by fit()
        """
        assert self.evaluator.corr is not None, "self.evaluator.corr is None. Please call fit() first."
        assert self.evaluator.n_features_on_party is not None, "self.evaluator.n_features_on_party is None. Please call fit() first."
        assert self.min_mcor is not None, "self.min_mcor is None. Please call fit() first."
        assert self.max_mcor is not None, "self.max_mcor is None. Please call fit() first."

    def fit(self, X, n_elites=20, n_offsprings=70, n_mutants=10, n_gen=100, bias=0.7, verbose=False):
        """
        Calculate the min and max mcor of the overall correlation score.
        Required parameters:
        :param X: [np.ndarray] 2D dataset

        Optional parameters: (BRKGA parameters)
        :param n_elites: (int) number of elites in BRKGA
        :param n_offsprings: (int) number of offsprings in BRKGA
        :param n_mutants: (int) number of mutants in BRKGA
        :param n_gen: (int) number of generations in BRKGA
        :param bias: (float) bias of BRKGA
        :param verbose: (bool) whether to print the progress
        """
        self.evaluator.corr = self.evaluator.corr_func(X)
        self.evaluator.n_features_on_party = self.split_num_features_equal(X.shape[1], self.num_parties)

        algorithm = BRKGA(
            n_elites=n_elites,
            n_offsprings=n_offsprings,
            n_mutants=n_mutants,
            bias=bias,
            eliminate_duplicates=self.DuplicationElimination(),
        )

        # calculate the min and max mcor
        if verbose:
            print("Calculating the min mcor of the overall correlation score...")
        res_min = minimize(
            self.CorrMinProblem(self.evaluator.corr, self.evaluator.n_features_on_party, evaluator=self.evaluator, runner=self.runner),
            algorithm,
            ('n_gen', n_gen),
            seed=self.seed,
            verbose=verbose,
        )
        self.min_mcor = res_min.F[0]
        print(f"min_mcor: {self.min_mcor}")
        if verbose:
            print("Calculating the max mcor of the overall correlation score...")
        res_max = minimize(
            self.CorrMaxProblem(self.evaluator.corr, self.evaluator.n_features_on_party, evaluator=self.evaluator, runner=self.runner),
            algorithm,
            ('n_gen', n_gen),
            seed=self.seed,
            verbose=verbose,
        )
        self.max_mcor = -res_max.F[0]
        print(f"max_mcor: {self.max_mcor}")

    def split(self, X, n_elites=20, n_offsprings=70, n_mutants=10, n_gen=100, bias=0.7, verbose=False,
              beta=0.5, term_tol=1e-4, term_period=10):
        """
        Use BRKGA to find the best order of features that minimizes the difference between the mean of mcor and the
        target. split() assumes that the min and max mcor have been calculated by fit().
        Required parameters:
        :param X: [np.ndarray] 2D dataset

        Optional parameters: (BRKGA parameters)
        :param n_elites: (int) number of elites in BRKGA
        :param n_offsprings: (int) number of offsprings in BRKGA
        :param n_mutants: (int) number of mutants in BRKGA
        :param n_gen: (int) number of generations in BRKGA
        :param bias: (float) bias of BRKGA
        :param verbose: (bool) whether to print the progress
        :param beta: [float] the tightness of inner-party correlation. Larger beta means more inner-party correlation
                             and less inter-party correlation.
        :param term_tol: (float) If out['F'] < term_tol after term_period generations, the algorithm terminates.
        :param term_period: (int) Check the termination condition every term_period generations

        :return: (np.ndarray) indices of features in the order of importance
        """
        self.check_fit_data()

        # termintation by number of generations or the error is less than 1e-6
        termination = DefaultSingleObjectiveTermination(ftol=term_tol, n_max_gen=n_gen, period=term_period)
        algorithm = BRKGA(
            n_elites=n_elites,
            n_offsprings=n_offsprings,
            n_mutants=n_mutants,
            bias=bias,
            eliminate_duplicates=self.DuplicationElimination(),
        )

        # find the best permutation order that makes the mcor closest to the target mcor
        # target_mcor = beta * max_mcor + (1 - beta) * min_mcor
        res_beta = minimize(
            self.CorrBestMatchProblem(self.evaluator.corr, self.evaluator.n_features_on_party, beta, self.min_mcor, self.max_mcor,
                                      evaluator=self.evaluator, runner=self.runner),
            algorithm,
            termination,
            seed=self.seed,
            verbose=verbose,
        )
        self.best_permutation = res_beta.opt.get('order')[0].astype(int)
        self.best_mcor = res_beta.opt.get('mcor')[0]
        self.best_error = res_beta.F[0]
        # print(f"Best permutation order: {permute_order}")
        # print(f"Beta {self.beta}, Best match mcor: {best_match_mcor}")

        # summarize the feature ids on each party
        party_cut_points = np.cumsum(self.evaluator.n_features_on_party)
        party_cut_points = np.insert(party_cut_points, 0, 0)
        self.best_feature_per_party = []
        for i in range(len(party_cut_points) - 1):
            start = party_cut_points[i]
            end = party_cut_points[i + 1]
            self.best_feature_per_party.append(np.sort(self.best_permutation[start:end]))
        assert (np.sort(np.concatenate(self.best_feature_per_party)) == np.arange(X.shape[1])).all()
        # print(f"Feature ids on each party: {feature_ids_on_party}")

        # split X according to the permutation order
        X_split = []
        for feature_ids in self.best_feature_per_party:
            X_split.append(X[:, feature_ids])
        return tuple(X_split)

    def fit_split(self, X, n_elites=20, n_offsprings=70, n_mutants=10, n_gen=100, bias=0.7, verbose=False, beta=1.,
                  term_tol=1e-4, term_period=10):
        """
        Calculate the min and max mcor of the overall correlation score. Then use BRKGA to find the best order of
        features that minimizes the difference between the mean of mcor and the target mcor.
        Required parameters:
        :param X: [np.ndarray] 2D dataset

        Optional parameters: (BRKGA parameters)
        :param n_elites: (int) number of elites in BRKGA
        :param n_offsprings: (int) number of offsprings in BRKGA
        :param n_mutants: (int) number of mutants in BRKGA
        :param n_gen: (int) number of generations in BRKGA
        :param bias: (float) bias of BRKGA
        :param seed: (int) seed of BRKGA
        :param verbose: (bool) whether to print the progress of BRKGA optimization
        :param beta: [float] the tightness of inner-party correlation. Larger beta means more inner-party correlation
                                and less inter-party correlation.
        :param term_tol: (float) If out['F'] < term_tol after term_period generations, the algorithm terminates.
        :param term_period: (int) Check the termination condition every term_period generations

        :return: (X1, X2, ..., Xn) [np.ndarray, ...] where n is the number of parties
        """
        self.fit(X, n_elites, n_offsprings, n_mutants, n_gen, bias, verbose)
        return self.split(X, n_elites, n_offsprings, n_mutants, n_gen, bias, verbose, beta, term_tol, term_period)

    def visualize(self, *args, **kwargs):
        return self.evaluator.visualize(*args, **kwargs)

    def evaluate_beta(self, score):
        """
        Evaluate the beta value that makes the score closest to the target score.
        :param score: [float] the score to be evaluated
        :return: [float] the beta value that makes the score closest to the target score
        """
        if self.min_mcor is None or self.max_mcor is None:
            raise ValueError("The min and max mcor have not been calculated. Please call fit() first.")
        if not (self.min_mcor <= score <= self.max_mcor):
            warnings.warn(f"The score {score} is out of range [{self.min_mcor}, {self.max_mcor}].")
        return (score - self.min_mcor) / (self.max_mcor - self.min_mcor)

    def evaluate_alpha(self, scores):
        """
        Evaluate the alpha value of a symmetric Dirichlet distribution that has the closest variance with the given
        Dirichlet distribution using scores as the parameters.
        :param scores: the importance scores of each party
        :return: [float] the alpha value that makes the score closest to the target score
        """
        assert len(scores) == self.num_parties
        scores = scores / np.sum(scores)    # normalize the scores
        score_var = np.var(scores)
        est_alpha = (self.num_parties - 1 - self.num_parties ** 2 * score_var) / (self.num_parties ** 3 * score_var)
        return est_alpha







