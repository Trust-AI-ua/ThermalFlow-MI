#!/usr/bin/env python3
"""Stage 1 of ThermalFlow-MI: train the flow matching prior on the attacker's auxiliary data.

Untargeted attack: one unconditional prior per dataset and distribution.
    python scripts/train_prior.py --dataset cftd --distribution in_distribution --attack untargeted
Targeted attack: the target model labels the auxiliary images (top-1 only), then a class-conditional prior is
trained on those labels; one prior per target model.
    python scripts/train_prior.py --dataset cftd --distribution in_distribution --attack targeted --target-model resnet18

Options used by the ablation and defence studies:
    --label-source ground_truth    train on the true identities of the auxiliary images (in-distribution only)
    --label-noise 0.3              replace 30% of the target model's labels by a wrong identity
    --classifier-checkpoint <pt>   the deployed model is a defended one (its labels are used)
    --tag <name>                   save under .../<target model>_<name> so the default prior is kept

Writes checkpoints/<dataset>/priors/<distribution>/<attack>[/<target model>]/model_final.pt, sample grids,
training losses, and for targeted priors auxiliary_labels.csv.
"""
import argparse

import _bootstrap  # noqa: F401
import torch

from thermalflow_mi.attack import FlowMatchingTrainer, RelabeledDataset, build_unet, corrupt_labels, query_labels
from thermalflow_mi.attack.labeling import save_labels
from thermalflow_mi.config import classifier_checkpoint, load_dataset_config, prior_dir
from thermalflow_mi.data import loader, prior_dataset
from thermalflow_mi.target_models import MODELS, load_classifier
from thermalflow_mi.utils import device, set_seed


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--distribution", required=True, choices=["in_distribution", "out_of_distribution"])
    ap.add_argument("--attack", required=True, choices=["untargeted", "targeted"])
    ap.add_argument("--target-model", choices=MODELS)
    ap.add_argument("--classifier-checkpoint", default=None)
    ap.add_argument("--label-source", default="target", choices=["target", "ground_truth"])
    ap.add_argument("--label-noise", type=float, default=0.0)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.attack == "targeted" and not a.target_model:
        ap.error("--attack targeted needs --target-model")
    if a.label_source == "ground_truth" and a.distribution != "in_distribution":
        ap.error("ground-truth labels exist only for in-distribution auxiliary data")

    cfg = load_dataset_config(a.dataset)
    dev = device()
    set_seed(a.seed)
    out = a.out or prior_dir(cfg.name, a.distribution, a.attack, a.target_model, a.tag)
    conditional = a.attack == "targeted"
    model = build_unet(cfg.num_channels, cfg.image_size, cfg.num_classes if conditional else None)
    data = prior_dataset(cfg, a.distribution)

    if not conditional:
        dl = loader(data, a.batch_size, shuffle=True)
    else:
        if a.label_source == "ground_truth":
            labels = [y for _, y in data.samples]
        else:
            ckpt = a.classifier_checkpoint or classifier_checkpoint(cfg.name, a.target_model)
            clf = load_classifier(a.target_model, cfg.num_classes, ckpt, dev)
            labels, conf = query_labels(clf, data.samples, dev)
            save_labels(f"{out}/auxiliary_labels.csv", data.samples, labels, conf, data.has_identity_labels)
            del clf
            torch.cuda.empty_cache()
            if a.label_noise > 0:
                labels = corrupt_labels(labels, a.label_noise, cfg.num_classes)
        data = RelabeledDataset(data, labels)
        gen = torch.Generator().manual_seed(a.seed) if a.label_source == "target" else None
        dl = loader(data, a.batch_size, shuffle=True, generator=gen)

    trainer = FlowMatchingTrainer(model, dev, out, conditional, cfg.num_classes, a.lr, cfg.image_size,
                                  cfg.num_channels)
    print(f"training {'class-conditional' if conditional else 'unconditional'} prior on {len(data)} images "
          f"-> {out}")
    trainer.train(dl, epochs=a.epochs)


if __name__ == "__main__":
    main()
