"""Frechet Inception Distance between two image sets (Heusel et al., NeurIPS 2017).

Grayscale images are repeated to three channels. Code adapted from https://github.com/mseitzer/pytorch-fid.
In the paper, FID compares the reconstructions of a run with the clean private images of the same subset.
"""
import numpy as np
import torch
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d

from ..utils.images import load_gray
from .inception import InceptionV3

_MODEL = {}


def inception(device):
    key = str(device)
    if key not in _MODEL:
        _MODEL[key] = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(device).eval()
    return _MODEL[key]


def load_rgb_array(paths):
    """N x 3 x H x W float array in [0, 1] from grayscale image files."""
    return np.stack([np.repeat(load_gray(p)[None], 3, axis=0) for p in paths]).astype(np.float32)


@torch.no_grad()
def activations(images, device, batch_size=50):
    """pool3 features (N x 2048) of an N x 3 x H x W array in [0, 1]."""
    model = inception(device)
    batch_size = min(batch_size, len(images))
    out = np.empty((len(images), 2048))
    start = 0
    for batch in torch.utils.data.DataLoader(images, batch_size=batch_size, drop_last=False):
        pred = model(batch.to(device))[0]
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
        pred = pred.squeeze(3).squeeze(2).cpu().numpy()
        out[start:start + len(pred)] = pred
        start += len(pred)
    return out


def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            raise ValueError(f"imaginary component {np.max(np.abs(covmean.imag))}")
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean))


def fid_from_arrays(real, fake, device="cuda", batch_size=50):
    a, b = activations(real, device, batch_size), activations(fake, device, batch_size)
    return frechet_distance(a.mean(0), np.cov(a, rowvar=False), b.mean(0), np.cov(b, rowvar=False))


def fid(reference_paths, candidate_paths, device="cuda", batch_size=50):
    """FID between two lists of image files (at least two images each)."""
    if len(reference_paths) < 2 or len(candidate_paths) < 2:
        raise ValueError("FID needs at least two images per set")
    bs = min(batch_size, len(reference_paths), len(candidate_paths))
    return fid_from_arrays(load_rgb_array(reference_paths), load_rgb_array(candidate_paths), device, bs)
