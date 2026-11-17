#!/usr/bin/env python3
"""Train a target classifier on the train split, optionally with a model-side defence.

    python scripts/train_classifier.py --dataset cftd --model resnet18
    python scripts/train_classifier.py --dataset cftd --model resnet50 --defense nls      [--nls-alpha -0.05]
    python scripts/train_classifier.py --dataset cftd --model resnet50 --defense bido     [--lambda-x 5 --lambda-y 50]
    python scripts/train_classifier.py --dataset cftd --model resnet50 --defense dp --epsilon 8

Writes checkpoints/<dataset>/classifiers[_<defence>]/<model>/final_model.pt and metrics.json (test accuracy).
"""
import argparse
import json

import _bootstrap  # noqa: F401
import torch

from thermalflow_mi.config import classifier_checkpoint, load_dataset_config
from thermalflow_mi.data import read_split
from thermalflow_mi.defenses import BiDOLoss, NLSLoss, train_dp
from thermalflow_mi.target_models import (DEFAULTS, MODELS, build_model, classifier_loaders, seed_all,
                                          train_classifier)
from thermalflow_mi.target_models.train import evaluate


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="cftd, charlotte, thermvision or a config path")
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--defense", default="none", choices=["none", "nls", "bido", "dp"])
    ap.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    ap.add_argument("--nls-alpha", type=float, default=-0.05)
    ap.add_argument("--lambda-x", type=float, default=5.0)
    ap.add_argument("--lambda-y", type=float, default=50.0)
    ap.add_argument("--epsilon", type=float, default=None, help="DP-SGD privacy budget (delta 1e-5)")
    ap.add_argument("--dp-lr", type=float, default=1e-2)
    ap.add_argument("--dp-train", default="fc", choices=["fc", "layer4", "all"])
    ap.add_argument("--out", default=None, help="output folder (default: checkpoints/...)")
    a = ap.parse_args()

    cfg = load_dataset_config(a.dataset)
    tag = a.defense if a.defense != "dp" else f"dp_eps{a.epsilon:g}"
    out = classifier_checkpoint(cfg.name, a.model, tag).parent if a.out is None else a.out
    seed_all(DEFAULTS["seed"])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits, classes = read_split(cfg.split_csv, cfg.data_root)
    train_loader, test_loader = classifier_loaders(splits, DEFAULTS["batch_size"], DEFAULTS["num_workers"])
    model = build_model(a.model, len(classes))

    if a.defense == "dp":
        if a.epsilon is None:
            ap.error("--defense dp needs --epsilon")
        info = train_dp(model, train_loader.dataset, dev, epsilon=a.epsilon, epochs=a.epochs, lr=a.dp_lr,
                        train=a.dp_train, seed=DEFAULTS["seed"])
        from pathlib import Path
        Path(out).mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), Path(out) / "final_model.pt")
        test, _ = evaluate(model, test_loader, dev, amp=False)
        (Path(out) / "metrics.json").write_text(json.dumps({**test, **info}, indent=2))
        print(f"test accuracy {test['accuracy']:.4f} -> {out}")
        return

    criterion, callback = None, None
    if a.defense == "nls":
        criterion = NLSLoss(a.nls_alpha)
        callback = criterion.schedule
    elif a.defense == "bido":
        criterion = BiDOLoss(a.lambda_x, a.lambda_y)
        criterion.attach(model)
    train_classifier(model, train_loader, test_loader, out, dev, criterion=criterion, epoch_callback=callback,
                     **{**DEFAULTS, "epochs": a.epochs})


if __name__ == "__main__":
    main()
