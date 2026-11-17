"""Attack accuracy: the share of reconstructions that a classifier assigns to the true identity (top-1).

Reconstructions are read from their 8-bit PNGs, so the classifier sees exactly what was saved. Two input pipelines:
    resize="pil"     PIL resize 128 -> 224, as the classifier was trained (main result tables)
    resize="tensor"  bilinear antialiased tensor resize 128 -> 224 (defence study)
Both normalise with mean 0.5 / std 0.5.
"""
import torch
import torch.nn.functional as F
from PIL import Image

from ..target_models import INPUT_SIZE, classifier_transform
from ..utils.images import load_gray


@torch.no_grad()
def predictions(classifier, paths, device="cuda", resize="pil", batch_size=64):
    if resize == "pil":
        tf = classifier_transform()
        x = torch.stack([tf(Image.open(p).convert("L")) for p in paths])
    elif resize == "tensor":
        x = torch.stack([torch.from_numpy(load_gray(p))[None] for p in paths])
    else:
        raise ValueError("resize must be pil or tensor")
    out = []
    for i in range(0, len(x), batch_size):
        b = x[i:i + batch_size].to(device)
        if resize == "tensor":
            b = F.interpolate(b, size=(INPUT_SIZE, INPUT_SIZE), mode="bilinear", align_corners=False,
                              antialias=True)
            b = (b - 0.5) / 0.5
        out.append(classifier(b).argmax(1).cpu())
    return torch.cat(out)


def attack_accuracy(classifier, paths, labels, device="cuda", resize="pil"):
    """Top-1 accuracy (in [0, 1]) and the per-image correctness."""
    pred = predictions(classifier, paths, device, resize)
    correct = pred == torch.as_tensor(labels)
    return float(correct.float().mean()), correct.int().tolist()
