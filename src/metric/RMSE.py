import numpy as np

class RMSE:
    def __init__(self, is_robust=True):
        self.is_robust = is_robust

    def __call__(self, label, pred):
        if self.is_robust:
            return np.sqrt(np.nanmean((label - pred) ** 2))
        else:
            return np.sqrt(np.mean((label - pred) ** 2))
