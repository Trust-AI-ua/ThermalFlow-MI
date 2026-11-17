"""LPIPS perceptual distance (Zhang et al., CVPR 2018) with the AlexNet backbone. Grayscale images are repeated to
three channels and scaled to [-1, 1]."""
import torch

from ..utils.images import load_gray

_MODEL = {}


def _model(device):
    key = str(device)
    if key not in _MODEL:
        import lpips as _lpips
        m = _lpips.LPIPS(net="alex", verbose=False).to(device).eval()
        for p in m.parameters():
            p.requires_grad_(False)
        _MODEL[key] = m
    return _MODEL[key]


@torch.no_grad()
def lpips(reference, candidate, device="cuda"):
    def t(v):
        a = load_gray(v) if isinstance(v, str) else v
        return torch.from_numpy(a).float()[None, None].repeat(1, 3, 1, 1).to(device) * 2.0 - 1.0
    return float(_model(device)(t(reference), t(candidate)).mean().item())
