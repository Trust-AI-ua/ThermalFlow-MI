#!/usr/bin/env python3
"""Image immunization defence: craft one perturbation per released image against the attacker's solver.

    python scripts/immunize.py --dataset cftd --distribution in_distribution --attack targeted \
        --target-model resnet50 --degradation gaussian_blur --per-class 20 --eps 8 --out runs/immun/gaussian_eps8
    python scripts/attack.py   <same arguments> --delta runs/immun/gaussian_eps8/delta.pt --out <dir>

Takes the arguments of scripts/attack.py (same images, prior and observation noise) plus --eps, --iters, --K,
--chunk. Writes <out>/delta.pt and <out>/craft.json.
"""
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import torch

from attack import parser, setup
from thermalflow_mi.attack import observe
from thermalflow_mi.defenses import craft


def main():
    ap = parser()
    ap.add_argument("--eps", type=float, default=8.0, help="L_inf budget in 1/255 of the [0, 1] range")
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--K", type=int, default=20, help="PnP steps of the differentiable surrogate")
    ap.add_argument("--chunk", type=int, default=20)
    a = ap.parse_args()
    if not a.out:
        ap.error("--out is required")
    cfg, dev, solver, dl, op, sigma, info = setup(a)
    xs, ys, ls = [], [], []
    for b, (x, lab) in enumerate(dl):                 # the observations exactly as the attack makes them
        x = x.to(dev)
        xs.append(x), ys.append(observe(x, op.H, sigma, seed=b)), ls.append(lab.to(dev))
    x, y, labels = torch.cat(xs), torch.cat(ys), torch.cat(ls)
    cond = labels if solver.conditional else None
    delta, log = craft(solver, x, y, labels, cond, op, cfg.num_classes, a.eps, a.iters, a.K, a.chunk, a.batch_size)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(delta.cpu(), out / "delta.pt")
    (out / "craft.json").write_text(json.dumps({**info, **log}, indent=1))
    print(f"saved {out / 'delta.pt'}")


if __name__ == "__main__":
    main()
