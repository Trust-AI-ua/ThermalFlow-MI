"""Box inpainting: a 40 x 40 box at the image centre is set to zero (half size 20 at 128 x 128)."""
import torch

from .base import Degradation


class BoxInpainting(Degradation):
    def __init__(self, half_size=20):
        self.half_size = int(half_size)

    def mask(self, x):
        d, h = x.shape[2] // 2, self.half_size
        m = torch.ones_like(x)
        m[:, :, d - h:d + h, d - h:d + h] = 0
        return m

    def H(self, x):
        return self.mask(x) * x

    H_adj = H                                      # a mask is its own adjoint
