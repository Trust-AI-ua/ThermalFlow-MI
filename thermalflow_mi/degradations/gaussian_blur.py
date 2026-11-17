"""Gaussian blur, sigma 1 px, 61 x 61 kernel (paper setting)."""
import torch

from .base import FFTConvolution


def gaussian_kernel(sigma, size):
    """Normalised 2-D Gaussian kernel of odd size."""
    r = torch.arange(-size // 2 + 1., size // 2 + 1.)
    x, y = torch.meshgrid(r, r, indexing="ij")
    k = torch.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    return k / k.sum()


class GaussianBlur(FFTConvolution):
    def __init__(self, sigma_blur=1.0, kernel_size=61, num_channels=1, image_size=128, device="cuda"):
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        self.sigma_blur = float(sigma_blur)
        super().__init__(gaussian_kernel(self.sigma_blur, int(kernel_size)), num_channels, image_size, device)
