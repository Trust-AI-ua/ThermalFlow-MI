"""Training of the target classifiers: 40 epochs, AdamW (lr 1e-4, weight decay 1e-4), ReduceLROnPlateau on the
training loss (factor 0.5, patience 3), mixed precision, batch 32, random horizontal flip and +-5 deg rotation.

The loss is plain cross-entropy unless a defence supplies another one (thermalflow_mi.defenses). The test split is
used once, after training.
"""
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .factory import INPUT_SIZE

DEFAULTS = dict(epochs=40, learning_rate=1e-4, weight_decay=1e-4, scheduler_factor=0.5, scheduler_patience=3,
                amp=True, batch_size=32, num_workers=4, seed=42)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ClassifierDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples, self.transform = list(samples), transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        return self.transform(Image.open(path).convert("L")), int(label), path


def classifier_loaders(splits, batch_size=32, num_workers=4, size=INPUT_SIZE):
    norm = [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
    train_tf = transforms.Compose([transforms.Resize((size, size)), transforms.RandomHorizontalFlip(),
                                   transforms.RandomRotation(5), *norm])
    test_tf = transforms.Compose([transforms.Resize((size, size)), *norm])
    kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    return (DataLoader(ClassifierDataset(splits["train"], train_tf), shuffle=True, **kw),
            DataLoader(ClassifierDataset(splits["test"], test_tf), shuffle=False, **kw))


def _metrics(y_true, y_pred):
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return {"accuracy": float(accuracy_score(y_true, y_pred)), "macro_precision": float(p),
            "macro_recall": float(r), "macro_f1": float(f1)}


def train_one_epoch(model, loader, criterion, optimizer, device, amp=True):
    model.train()
    scaler = torch.amp.GradScaler("cuda", enabled=amp and device.type == "cuda")
    total, n, y_true, y_pred = 0.0, 0, [], []
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            logits = model(x)
            loss = criterion(logits, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total += loss.item() * y.size(0)
        n += y.size(0)
        y_true += y.tolist()
        y_pred += logits.argmax(1).tolist()
    return {**_metrics(y_true, y_pred), "loss": total / n}


@torch.no_grad()
def evaluate(model, loader, device, amp=True):
    model.eval()
    rows, y_true, y_pred = [], [], []
    for x, y, paths in loader:
        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            logits = model(x.to(device))
        conf, pred = torch.softmax(logits.float(), 1).max(1)
        for p, t, q, c in zip(paths, y.tolist(), pred.tolist(), conf.tolist()):
            rows.append({"image_path": p, "label": t, "prediction": q, "confidence": c, "correct": int(t == q)})
        y_true += y.tolist()
        y_pred += pred.tolist()
    return _metrics(y_true, y_pred), rows


def train_classifier(model, train_loader, test_loader, out_dir, device, criterion=None, epochs=40,
                     learning_rate=1e-4, weight_decay=1e-4, scheduler_factor=0.5, scheduler_patience=3, amp=True,
                     epoch_callback=None, **_):
    """Train, save <out_dir>/final_model.pt, then evaluate once on the test split (metrics.json, predictions.csv).
    epoch_callback(epoch, epochs) runs before every epoch (used by defences with a loss schedule)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = model.to(device)
    criterion = criterion or nn.CrossEntropyLoss()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=scheduler_factor,
                                                       patience=scheduler_patience)
    history, t0 = [], time.time()
    for epoch in range(1, epochs + 1):
        if epoch_callback:
            epoch_callback(epoch, epochs)
        m = train_one_epoch(model, train_loader, criterion, opt, device, amp)
        sched.step(m["loss"])
        history.append({"epoch": epoch, **m, "lr": opt.param_groups[0]["lr"]})
        print(f"epoch {epoch:03d}/{epochs} loss {m['loss']:.4f} acc {m['accuracy']:.4f}", flush=True)
    torch.save(model.state_dict(), out_dir / "final_model.pt")

    test, rows = evaluate(model, test_loader, device, amp)
    test.update(training_seconds=time.time() - t0, epochs=epochs)
    (out_dir / "metrics.json").write_text(json.dumps(test, indent=2))
    with open(out_dir / "history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0]))
        w.writeheader()
        w.writerows(history)
    with open(out_dir / "test_predictions.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"test accuracy {test['accuracy']:.4f} -> {out_dir}")
    return test
