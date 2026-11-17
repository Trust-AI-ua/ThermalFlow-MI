"""Image conversions and saving."""
import numpy as np
import torch
from PIL import Image
from torchvision.utils import save_image


def to_unit(x):
    """[-1, 1] pipeline tensors -> [0, 1], clamped."""
    return torch.clamp(x * 0.5 + 0.5, 0.0, 1.0)


def save_png(x, path):
    """Save one image tensor in [-1, 1] as an 8-bit PNG."""
    save_image(to_unit(x.detach()).cpu(), str(path))


def load_gray(path):
    """8-bit grayscale image -> float32 array in [0, 1]."""
    with Image.open(path) as img:
        return np.asarray(img.convert("L"), dtype=np.float32) / 255.0
