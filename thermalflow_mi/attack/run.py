"""Run the attack over a set of private images and save original / observed / reconstructed PNGs and a manifest."""
from pathlib import Path

import torch

from ..utils.images import save_png
from ..utils.manifest import write_manifest
from .pnp_flow import observe


def run_attack(solver, loader, op, sigma_noise, out_dir, condition=None, release_delta=None, attack_op=None):
    """Reconstruct every image of `loader` (batches of (x, label), in a fixed order).

    condition      None for the untargeted attack, or f(true labels) -> conditioning labels for the targeted one
                   (the paper conditions on the true identity of the image being reconstructed).
    release_delta  immunization: a tensor with one perturbation per image, added to the released observation.
    attack_op      a different operator used in the data term only (mis-specified blur kernel); the manifest
                   then also gets the residual ||H_k x_hat - y|| of every image.
    """
    out = Path(out_dir)
    dirs = {k: out / k for k in ("original", "observed", "reconstructed")}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    dataset = loader.dataset
    samples, classes = _samples_and_classes(dataset)
    rows, start, dev = [], 0, solver.device
    for b, (x, labels) in enumerate(loader):
        x = x.to(dev, non_blocking=True)
        labels = labels.to(dev, dtype=torch.long)
        delta = release_delta[start:start + len(x)] if release_delta is not None else None
        y = observe(x, op.H, sigma_noise, seed=b, offset=delta)
        cond = condition(labels) if condition is not None else None
        x_hat = solver.reconstruct(y, op.H, op.H_adj, cond,
                                   *((attack_op.H, attack_op.H_adj) if attack_op is not None else ()))
        residual = ((attack_op or op).H(x_hat) - y).flatten(1).norm(dim=1).tolist()
        for i in range(len(x)):
            g = start + i
            src = samples[g][0] if samples else ""
            c = int(labels[i])
            name = f"{g:06d}_id{c + 1:02d}_{Path(src).stem if src else 'sample'}.png"
            for k, img in (("original", x[i]), ("observed", y[i]), ("reconstructed", x_hat[i])):
                save_png(img, dirs[k] / name)
            rows.append({"global_index": g, "class_index": c, "class_name": classes[c] if classes else c + 1,
                         "condition_label": "" if cond is None else int(cond[i]), "source_path": src,
                         "original_path": f"original/{name}", "observed_path": f"observed/{name}",
                         "reconstructed_path": f"reconstructed/{name}", "residual_l2": residual[i]})
        start += len(x)
        print(f"batch {b + 1}/{len(loader)}", flush=True)
    write_manifest(out / "manifest.csv", rows, extra_fields=["residual_l2"])
    return out / "manifest.csv"


def _samples_and_classes(dataset):
    """Source paths in loader order (also through torch Subset) and class names."""
    idx, ds = list(range(len(dataset))), dataset
    while hasattr(ds, "dataset"):
        idx = [ds.indices[i] for i in idx]
        ds = ds.dataset
    samples = [ds.samples[i] for i in idx] if hasattr(ds, "samples") else None
    return samples, getattr(ds, "classes", None)
