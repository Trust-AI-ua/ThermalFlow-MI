#!/usr/bin/env python3
"""Score an attack run: attack accuracy under one or more classifiers, and PSNR, SSIM, LPIPS, FID (and KID)
of the reconstructions against the clean private images.

    python scripts/evaluate.py results/cftd/in_distribution/untargeted/gaussian_blur/manifest.csv \
        --dataset cftd --classifiers resnet18 resnet50 vgg16 vit_b_16 --per-class 20

--per-class N   score only the first N images of every identity (the paper's 200-image CFTD tables)
--observed      score the degraded observations instead (the "No attack" rows)
--resize        pil (main tables) or tensor (defence study), see thermalflow_mi/metrics/accuracy.py
--checkpoint    score with another checkpoint than the default one of the (single) classifier
Writes <manifest folder>/metrics[_observed].json unless --out is given.
"""
import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np

from thermalflow_mi.config import classifier_checkpoint, load_dataset_config
from thermalflow_mi.data import first_per_class
from thermalflow_mi.metrics import attack_accuracy, fid, kid, lpips, psnr, ssim
from thermalflow_mi.target_models import MODELS, load_classifier
from thermalflow_mi.utils import device, read_manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("manifest")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--classifiers", nargs="*", default=[], choices=MODELS)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--per-class", type=int, default=None)
    ap.add_argument("--observed", action="store_true")
    ap.add_argument("--resize", default="pil", choices=["pil", "tensor"])
    ap.add_argument("--no-quality", action="store_true", help="accuracy only")
    ap.add_argument("--kid", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    cfg = load_dataset_config(a.dataset)
    dev = device()
    rows = read_manifest(a.manifest)
    if a.per_class:
        rows = first_per_class(rows, a.per_class)
    key = "observed_path" if a.observed else "reconstructed_path"
    refs, cands = [r["original_path"] for r in rows], [r[key] for r in rows]
    labels = [int(r["class_index"]) for r in rows]
    res = {"manifest": str(a.manifest), "scored": key, "n": len(rows)}

    res["accuracy"], res["correct"] = {}, {}
    for m in a.classifiers:
        ck = a.checkpoint if (a.checkpoint and len(a.classifiers) == 1) else classifier_checkpoint(cfg.name, m)
        acc, correct = attack_accuracy(load_classifier(m, cfg.num_classes, ck, dev), cands, labels, dev, a.resize)
        res["accuracy"][m] = 100 * acc
        res["correct"][m] = dict(zip([int(r["global_index"]) for r in rows], correct))
        print(f"accuracy {m}: {100 * acc:.1f}%")
    if not a.no_quality:
        res["psnr"] = float(np.mean([psnr(r, c) for r, c in zip(refs, cands)]))
        res["ssim"] = float(np.mean([ssim(r, c) for r, c in zip(refs, cands)]))
        res["lpips"] = float(np.mean([lpips(r, c, dev) for r, c in zip(refs, cands)]))
        res["fid"] = fid(refs, cands, dev)
        if a.kid:
            res["kid"], res["kid_std"] = kid(refs, cands, dev)
        print({k: round(res[k], 4) for k in ("psnr", "ssim", "lpips", "fid", "kid") if k in res})
    out = Path(a.out) if a.out else Path(a.manifest).parent / f"metrics{'_observed' if a.observed else ''}.json"
    out.write_text(json.dumps(res, indent=1))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
