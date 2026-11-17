#!/usr/bin/env python3
"""Stage 2 of ThermalFlow-MI: rebuild private faces from their leaked degraded copies.

    # untargeted attack on all 1,600 private CFTD images under Gaussian blur
    python scripts/attack.py --dataset cftd --distribution in_distribution --attack untargeted \
        --degradation gaussian_blur
    # targeted attack on ResNet18, 20 images per identity
    python scripts/attack.py --dataset cftd --distribution in_distribution --attack targeted \
        --target-model resnet18 --degradation motion_blur --per-class 20

Each private image x gives the observation y = H x + sigma n (noise seeded per batch of 4), and PnP-Flow
reconstructs x from y with the prior of the attack. The targeted attack conditions the prior on the identity of
the image. Step sizes come from configs/datasets/<dataset>.yaml unless --lr-pnp / --alpha are given.

Options used by the sensitivity, ablation and defence studies:
    --lr-pnp, --alpha                      step-size study
    --attack-sigma-blur / --attack-motion-length / --attack-motion-angle
                                           the attacker assumes a wrong blur kernel (data term only)
    --num-samples M                        denoiser draws per step (ablation; paper M = 5)
    --prior <dir or model_final.pt>        another prior (e.g. a ground-truth-label or defended-label prior)
    --unconditional-prior                  targeted run with the untargeted prior (ablation)
    --delta <delta.pt>                     the released observation is immunized (scripts/immunize.py)

Writes <out>/{original,observed,reconstructed}/*.png, <out>/manifest.csv and <out>/config.json.
Default <out>: results/<dataset>/<distribution>/<attack>[/<target model>]/<degradation>.
"""
import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import torch
from torch.utils.data import Subset

from thermalflow_mi.attack import PnPFlow, build_unet, run_attack
from thermalflow_mi.config import REPO, load_dataset_config, pnp_setting, prior_dir
from thermalflow_mi.data import loader, private_dataset, random_per_class
from thermalflow_mi.degradations import DEGRADATIONS, build_degradation
from thermalflow_mi.target_models import MODELS
from thermalflow_mi.utils import device, set_seed


def parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--distribution", required=True, choices=["in_distribution", "out_of_distribution"])
    ap.add_argument("--attack", required=True, choices=["untargeted", "targeted"])
    ap.add_argument("--target-model", choices=MODELS)
    ap.add_argument("--degradation", required=True, choices=DEGRADATIONS)
    ap.add_argument("--split", default="train", help="private images to attack (default: the training split)")
    ap.add_argument("--per-class", type=int, default=None, help="attack a seeded subset of N images per identity")
    ap.add_argument("--subset-seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--num-samples", type=int, default=5)
    ap.add_argument("--lr-pnp", type=float, default=None)
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--attack-sigma-blur", type=float, default=None)
    ap.add_argument("--attack-motion-length", type=int, default=None)
    ap.add_argument("--attack-motion-angle", type=float, default=None)
    ap.add_argument("--prior", default=None)
    ap.add_argument("--unconditional-prior", action="store_true")
    ap.add_argument("--delta", default=None)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    return ap


def setup(a):
    """Dataset config, solver, private-image loader, operator and noise level for the arguments `a`."""
    if a.attack == "targeted" and not a.target_model:
        raise SystemExit("--attack targeted needs --target-model")
    cfg = load_dataset_config(a.dataset)
    dev = device()
    set_seed(a.seed)
    conditional = a.attack == "targeted" and not a.unconditional_prior
    if a.prior:
        ckpt = Path(a.prior)
        ckpt = ckpt / "model_final.pt" if ckpt.is_dir() else ckpt
    else:
        kind = "targeted" if conditional else "untargeted"
        ckpt = prior_dir(cfg.name, a.distribution, kind, a.target_model) / "model_final.pt"
    model = build_unet(cfg.num_channels, cfg.image_size, cfg.num_classes if conditional else None)
    model.load_state_dict(torch.load(ckpt, map_location=dev))
    lr_pnp, alpha = pnp_setting(cfg, a.degradation)
    lr_pnp = a.lr_pnp if a.lr_pnp is not None else lr_pnp
    alpha = a.alpha if a.alpha is not None else alpha
    op, sigma = build_degradation(a.degradation, cfg.num_channels, cfg.image_size, dev)
    solver = PnPFlow(model, sigma, lr_pnp, alpha, a.steps, a.num_samples, conditional, dev)

    data = private_dataset(cfg, a.split)
    if a.per_class:
        data = Subset(data, random_per_class(data, a.per_class, a.subset_seed))
    dl = loader(data, a.batch_size, shuffle=False)
    info = {"dataset": cfg.name, "distribution": a.distribution, "attack": a.attack, "target_model": a.target_model,
            "degradation": a.degradation, "noise_sigma": sigma, "lr_pnp": lr_pnp, "alpha": alpha,
            "steps": a.steps, "num_samples": a.num_samples, "conditional_prior": conditional,
            "prior": str(ckpt), "num_images": len(data), "per_class": a.per_class, "subset_seed": a.subset_seed}
    return cfg, dev, solver, dl, op, sigma, info


def main():
    a = parser().parse_args()
    cfg, dev, solver, dl, op, sigma, info = setup(a)
    attack_op = None
    if any(v is not None for v in (a.attack_sigma_blur, a.attack_motion_length, a.attack_motion_angle)):
        attack_op, _ = build_degradation(a.degradation, cfg.num_channels, cfg.image_size, dev,
                                         sigma_blur=a.attack_sigma_blur or 1.0,
                                         motion_length=a.attack_motion_length or 15,
                                         motion_angle=a.attack_motion_angle or 0.0)
        info["attack_kernel"] = {"sigma_blur": a.attack_sigma_blur, "motion_length": a.attack_motion_length,
                                 "motion_angle": a.attack_motion_angle}
    delta = torch.load(a.delta) if a.delta else None
    if delta is not None:
        assert len(delta) == len(dl.dataset), "the perturbations were crafted for another image set"
        info["delta"] = str(a.delta)

    out = Path(a.out) if a.out else (REPO / "results" / cfg.name / a.distribution / a.attack
                                     / (a.target_model if a.attack == "targeted" else "") / a.degradation)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(info, indent=2))
    condition = (lambda y: y) if info["conditional_prior"] else None   # condition on the true identity
    print(json.dumps(info, indent=2))
    manifest = run_attack(solver, dl, op, sigma, out, condition, delta, attack_op)
    print(f"done: {manifest}")


if __name__ == "__main__":
    main()
