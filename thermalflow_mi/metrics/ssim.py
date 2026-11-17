"""Structural similarity (Wang et al., 2004) between a reconstruction and the clean image (scikit-image,
7 x 7 uniform window, data range 1)."""
from skimage.metrics import structural_similarity

from ..utils.images import load_gray


def ssim(reference, candidate):
    a = load_gray(reference) if isinstance(reference, str) else reference
    b = load_gray(candidate) if isinstance(candidate, str) else candidate
    return float(structural_similarity(a, b, data_range=1.0))
