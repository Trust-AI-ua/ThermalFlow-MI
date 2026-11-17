"""Kernel Inception Distance (Binkowski et al., "Demystifying MMD GANs", ICLR 2018).

Unbiased estimate of MMD^2 between InceptionV3 pool3 features with the cubic polynomial kernel
k(a, b) = (a.b / d + 1)^3, averaged over 100 random subsets of 100 images (seed 0). Unlike FID it is unbiased
for small sets, which is why the sensitivity study (200 images per run) reports it.
"""
import numpy as np

from .fid import activations, load_rgb_array


def kid_from_features(fx, fy, subsets=100, subset_size=100, seed=0):
    """(mean, std) of the unbiased MMD^2 over random subsets."""
    rng = np.random.default_rng(seed)
    d = fx.shape[1]
    m = min(subset_size, len(fx), len(fy))
    vals = []
    for _ in range(subsets):
        x = fx[rng.choice(len(fx), m, replace=False)]
        y = fy[rng.choice(len(fy), m, replace=False)]
        kxx, kyy, kxy = (x @ x.T / d + 1) ** 3, (y @ y.T / d + 1) ** 3, (x @ y.T / d + 1) ** 3
        vals.append((kxx.sum() - np.trace(kxx)) / (m * (m - 1)) + (kyy.sum() - np.trace(kyy)) / (m * (m - 1))
                    - 2 * kxy.mean())
    return float(np.mean(vals)), float(np.std(vals))


def kid(reference_paths, candidate_paths, device="cuda", **kw):
    fx = activations(load_rgb_array(reference_paths), device)
    fy = activations(load_rgb_array(candidate_paths), device)
    return kid_from_features(fx, fy, **kw)
