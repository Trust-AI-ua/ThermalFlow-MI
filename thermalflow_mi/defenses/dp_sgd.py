"""Differentially private training with DP-SGD (Abadi et al., ACM CCS 2016), (epsilon, delta)-DP for the private
training images.

Each step samples every training image with probability q = B / N (Poisson), computes per-example gradients
(torch.func), clips each to L2 norm C, sums them, adds Gaussian noise N(0, (sigma C)^2) and divides by B before
an Adam step. sigma is the smallest noise multiplier that keeps the whole run within (epsilon, delta) under the RDP
accountant of Opacus (used only for the accounting).

As is standard for DP on small datasets (Tramer & Boneh, ICLR 2021), only the classifier head is trained, on the
frozen ImageNet backbone; BatchNorm keeps its ImageNet statistics. Paper setting: epsilon in {1, 8, 50},
delta 1e-5, B 256, C 1, lr 1e-2, 40 epochs, head only.
"""
import time

import torch
import torch.nn.functional as F
from torch.func import functional_call, grad, vmap


def noise_multiplier(epsilon, delta, sample_rate, steps):
    from opacus.accountants.utils import get_noise_multiplier
    return get_noise_multiplier(target_epsilon=epsilon, target_delta=delta, sample_rate=sample_rate, steps=steps,
                                accountant="rdp")


def train_dp(model, train_dataset, device, epsilon=None, delta=1e-5, epochs=40, batch=256, clip=1.0, lr=1e-2,
             micro=16, train="fc", seed=42):
    """DP-SGD training in place. epsilon=None trains without noise (same clipping and sampling). Returns a dict
    with the accounting. `train`: "fc" (head only), "layer4" (layer4 + head) or "all"."""
    n = len(train_dataset)
    q = batch / n
    steps = epochs * n // batch
    sigma = 0.0 if epsilon is None else noise_multiplier(epsilon, delta, q, steps)
    model.to(device).eval()                                  # BatchNorm: ImageNet statistics, frozen
    for mod in model.modules():
        if isinstance(mod, torch.nn.BatchNorm2d):
            for p in mod.parameters():
                p.requires_grad_(False)
    if train != "all":
        keep = ("fc.",) if train == "fc" else ("fc.", "layer4.")
        for k, p in model.named_parameters():
            if not any(t in k for t in keep):
                p.requires_grad_(False)
    params = {k: p for k, p in model.named_parameters() if p.requires_grad}
    frozen = {k: p for k, p in model.named_parameters() if not p.requires_grad}
    buffers = dict(model.named_buffers())
    opt = torch.optim.Adam(params.values(), lr=lr)

    def loss_one(p, x, y):
        logits = functional_call(model, ({**p, **frozen}, buffers), (x.unsqueeze(0),))
        return F.cross_entropy(logits, y.unsqueeze(0))
    per_example = vmap(grad(loss_one), in_dims=(None, 0, 0))

    acct = None
    if sigma:
        from opacus.accountants import RDPAccountant
        acct = RDPAccountant()
    gen = torch.Generator().manual_seed(seed)
    t0 = time.time()
    for step in range(1, steps + 1):
        idx = (torch.rand(n, generator=gen) < q).nonzero().flatten().tolist()
        total = {k: torch.zeros_like(p) for k, p in params.items()}
        for s in range(0, len(idx), micro):
            items = [train_dataset[i] for i in idx[s:s + micro]]
            x = torch.stack([b[0] for b in items]).to(device)
            y = torch.tensor([int(b[1]) for b in items], device=device)
            g = per_example({k: p.detach() for k, p in params.items()}, x, y)
            norms = torch.sqrt(sum(v.flatten(1).pow(2).sum(1) for v in g.values()))
            scale = (clip / (norms + 1e-6)).clamp(max=1.0)
            for k, v in g.items():
                total[k] += torch.einsum("b,b...->...", scale, v)
        for k, p in params.items():
            noise = torch.normal(0.0, sigma * clip, size=p.shape, device=device) if sigma else 0.0
            p.grad = (total[k] + noise) / batch
        opt.step()
        opt.zero_grad(set_to_none=True)
        if acct is not None:
            acct.step(noise_multiplier=sigma, sample_rate=q)
        if step % 25 == 0 or step == steps:
            e = acct.get_epsilon(delta) if acct is not None else float("inf")
            print(f"step {step}/{steps}  epsilon {e:.2f}  {time.time() - t0:.0f}s", flush=True)
    return {"epsilon": acct.get_epsilon(delta) if acct is not None else None, "target_epsilon": epsilon, "delta": delta,
            "noise_multiplier": sigma, "clip": clip, "expected_batch": batch, "sample_rate": q, "steps": steps,
            "epochs": epochs, "lr": lr, "train": train, "num_train": n, "seconds": time.time() - t0}
