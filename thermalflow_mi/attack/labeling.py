"""Stage 1a of the targeted attack: label the attacker's prior data with the target model.

The attacker only queries the deployed classifier for its top-1 label (label-only access). The prior is then
trained on (image, predicted identity) pairs. For out-of-distribution prior data, every image belongs to some
other person and receives the identity the target model confuses it with.
"""
import csv
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from ..target_models import classifier_transform


class _Paths(Dataset):
    def __init__(self, samples, transform):
        self.samples, self.transform = samples, transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        return self.transform(Image.open(path).convert("L")), int(label)


@torch.no_grad()
def query_labels(classifier, samples, device="cuda", batch_size=64, num_workers=4):
    """Top-1 predictions and confidences of the classifier on the images of `samples` ([(path, label)])."""
    dl = DataLoader(_Paths(samples, classifier_transform()), batch_size=batch_size, shuffle=False,
                    num_workers=num_workers, pin_memory=device != "cpu")
    preds, confs = [], []
    for x, _ in dl:
        p = torch.softmax(classifier(x.to(device)).float(), dim=1)
        c, k = p.max(dim=1)
        preds += k.cpu().tolist()
        confs += c.cpu().tolist()
    return preds, confs


def corrupt_labels(labels, fraction, num_classes, seed=0):
    """Replace a fixed fraction of the labels by a wrong identity (target-accuracy study). One seeded permutation
    is used, so the images changed at a small fraction are also changed at every larger one."""
    n = len(labels)
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    shift = [rng.randint(1, num_classes - 1) for _ in range(n)]
    flip = set(order[:round(fraction * n)])
    return [(y + shift[i]) % num_classes if i in flip else y for i, y in enumerate(labels)]


class RelabeledDataset(Dataset):
    """The prior dataset with its labels replaced (e.g. by the target model's predictions)."""

    def __init__(self, base, labels):
        if len(base) != len(labels):
            raise ValueError("one label per image is required")
        self.base, self.labels = base, [int(y) for y in labels]
        self.samples = [(p, y) for (p, _), y in zip(base.samples, self.labels)]
        self.classes = getattr(base, "classes", None)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        return self.base[i][0], self.labels[i]


def save_labels(path, samples, preds, confs, has_identity_labels):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "true_label", "target_prediction", "confidence"])
        for (p, y), q, c in zip(samples, preds, confs):
            w.writerow([p, y if has_identity_labels else "", q, f"{c:.6f}"])
