"""Random inpainting: 70% of the pixels are missing.

As in the original PnP-Flow code, the mask is drawn from a NumPy generator seeded with 42 for a whole batch of shape
(B, H, W), so image i of a batch always carries mask i of that draw: with batches of 4, every fourth image shares
a mask. The mask is identical in every call (observation, data term and adjoint).
"""
import numpy as np
import torch

from .base import Degradation


class RandomInpainting(Degradation):
    def __init__(self, p_missing=0.7, seed=42):
        self.p, self.seed = float(p_missing), int(seed)
        self._cache = {}

    def mask(self, x):
        key = (x.shape[0], x.shape[2], x.shape[3], str(x.device))
        if key not in self._cache:
            keep = np.random.RandomState(self.seed).binomial(n=1, p=1 - self.p, size=key[:3])
            self._cache[key] = torch.from_numpy(keep).to(x.device).unsqueeze(1)
        return self._cache[key]

    def H(self, x):
        return self.mask(x) * x

    H_adj = H
