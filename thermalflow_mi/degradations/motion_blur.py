"""Linear motion blur, length 15 px, angle 0 deg (horizontal) in the paper."""
import math

import torch

from .base import FFTConvolution


def motion_kernel(length, angle_degrees, device="cpu", dtype=torch.float32):
    """Normalised length x length line kernel through the centre at the given angle (0 = horizontal,
    90 = vertical). Points sampled densely along the line are splatted bilinearly onto the pixel grid."""
    if length <= 0 or length % 2 == 0:
        raise ValueError("length must be a positive odd integer")
    kernel = torch.zeros((length, length), device=device, dtype=dtype)
    center = (length - 1) / 2.0
    theta = math.radians(float(angle_degrees))
    num_points = max(2 * length + 1, 33)
    t = torch.linspace(-center, center, steps=num_points, device=device, dtype=dtype)
    xs, ys = center + t * math.cos(theta), center + t * math.sin(theta)
    x0, y0 = torch.floor(xs).long(), torch.floor(ys).long()
    wx1, wy1 = xs - x0.to(dtype), ys - y0.to(dtype)
    wx0, wy0 = 1.0 - wx1, 1.0 - wy1
    for xi, yi, w in ((x0, y0, wx0 * wy0), (x0 + 1, y0, wx1 * wy0), (x0, y0 + 1, wx0 * wy1), (x0 + 1, y0 + 1, wx1 * wy1)):
        ok = (xi >= 0) & (xi < length) & (yi >= 0) & (yi < length)
        if ok.any():
            kernel.index_put_((yi[ok], xi[ok]), w[ok], accumulate=True)
    return kernel / kernel.sum()


class MotionBlur(FFTConvolution):
    def __init__(self, length=15, angle=0.0, num_channels=1, image_size=128, device="cuda"):
        self.length, self.angle = int(length), float(angle)
        super().__init__(motion_kernel(self.length, self.angle, device=device), num_channels, image_size, device)
