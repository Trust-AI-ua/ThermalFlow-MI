"""Image immunization (in the spirit of PhotoGuard, Salman et al., ICML 2023).

The data owner releases y' = y + delta instead of the degraded copy y, with ||delta||_inf <= eps (eps on the [0, 1]
scale, 2 eps on the pipeline's [-1, 1] scale). delta is found by projected gradient descent so that the attack
rebuilds a decoy, the clean image of another identity, instead of the person:
    min_delta || R_K(y + delta) - x_decoy ||^2
R_K is the attacker's own PnP-Flow solver (same prior, conditioning, data term and step sizes) truncated to K steps
with one denoiser draw and gradient checkpointing, so it can be differentiated, with fresh noise at every PGD
iteration. The defender knows the attacker's prior and settings: a best case for the defence.
Paper setting: eps 8/255 and 16/255, 60 PGD iterations, step eps/8, K = 20, chunks of 20 images.
"""
import time

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint


def decoy_indices(labels, num_classes):
    """For each image, the first image of identity (c + 1) mod K present in the set (or the next one present)."""
    first = {int(c): i for i, c in reversed(list(enumerate(labels)))}
    nxt = lambda c: next(first[(c + k) % num_classes] for k in range(1, num_classes)          # noqa: E731
                         if (c + k) % num_classes in first)
    return [nxt(int(c)) for c in labels]


def per_image_operator(op, n, batch_size, device):
    """Operator for a chunk of images that may span several attack batches. Random inpainting draws its mask per
    attack batch (image i of a batch carries mask i), so the masks are rebuilt per image."""
    if hasattr(op, "p") and hasattr(op, "seed"):                    # RandomInpainting
        def make(sl, shape):
            keep = np.random.RandomState(op.seed).binomial(1, 1 - op.p, size=(batch_size, shape[2], shape[3]))
            m = torch.from_numpy(keep).float()[torch.arange(n) % batch_size].unsqueeze(1).to(device)
            return lambda v: m[sl] * v
        return make
    return lambda sl, shape: op.H


def craft(solver, x, y, labels, cond, op, num_classes, eps_255=8.0, iters=60, K=20, chunk=20, batch_size=4,
          log=print):
    """PGD perturbations (one per image, same shape as y) on the [-1, 1] scale."""
    dev = solver.device
    for p in solver.model.parameters():
        p.requires_grad_(False)
    decoy = x[decoy_indices(labels.tolist(), num_classes)]
    make_H = per_image_operator(op, len(x), batch_size, dev)
    sigma = solver.sigma
    eps = 2 * eps_255 / 255
    step = eps / 8

    def surrogate(yy, c, H):
        def one(xx, it):
            t = torch.full((len(xx),), it / K, device=dev)
            z = xx - solver.step_size(t) * H(H(xx) - yy) / sigma ** 2
            tt = t.view(-1, 1, 1, 1)
            return solver.denoise(tt * z + torch.randn_like(z) * (1.0 - tt), t, c)
        xx = H(yy)
        for it in range(K):
            xx = checkpoint(one, xx, it, use_reentrant=False)
        return xx

    delta, t0, history = torch.zeros_like(y), time.time(), []
    for s in range(0, len(x), chunk):
        sl = slice(s, s + chunk)
        H = make_H(sl, y.shape)
        c = cond[sl] if cond is not None else None
        d = torch.zeros_like(y[sl]).uniform_(-eps, eps).requires_grad_(True)
        for _ in range(iters):
            loss = (surrogate(y[sl] + d, c, H) - decoy[sl]).pow(2).mean()
            g, = torch.autograd.grad(loss, d)
            with torch.no_grad():
                d -= step * g.sign()
                d.clamp_(-eps, eps)
        delta[sl] = d.detach()
        history.append({"images": [s, min(s + chunk, len(x))], "decoy_loss": loss.item()})
        log(f"{min(s + chunk, len(x))}/{len(x)}  decoy loss {loss.item():.4f}  {time.time() - t0:.0f}s")
    return delta, {"eps_255": eps_255, "eps_pipeline_scale": eps, "pgd_iters": iters, "pgd_step": step, "K": K,
                   "chunks": history, "seconds": time.time() - t0}
