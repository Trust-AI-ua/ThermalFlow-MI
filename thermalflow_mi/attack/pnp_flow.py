"""Stage 2: reconstruct the private image from its degraded copy with PnP-Flow (Martin et al., ICLR 2025).

For n = 0 .. N-1 with t_n = n / N:
    z      = x - gamma_n * H^T (H x - y) / sigma^2           gradient step on the data term
    z~     = t_n z + (1 - t_n) eps,  eps ~ N(0, I)           interpolate towards the noise level the prior expects
    x      = mean over M draws of D_t(z~) = z~ + (1 - t_n) v_theta(z~, t_n[, c])
with gamma_n = eta * sigma^2 * (1 - t_n)^alpha. The attack only runs forward passes of the prior: no gradient of
the prior, of an ODE solver or of the target model is needed. The targeted attack conditions the prior on the
identity c of the person being reconstructed.

Paper settings: N = 200 steps, M = 5 denoiser draws; eta and alpha per dataset and degradation in
configs/datasets/*.yaml.
"""
import torch


class PnPFlow:
    def __init__(self, model, sigma_noise, lr_pnp, alpha, steps=200, num_samples=5, conditional=False,
                 device="cuda"):
        self.model = model.to(device).eval()
        self.sigma, self.eta, self.alpha = float(sigma_noise), float(lr_pnp), float(alpha)
        self.steps, self.M, self.conditional, self.device = int(steps), int(num_samples), conditional, device

    def step_size(self, t):
        """gamma(t) = eta * sigma^2 * (1 - t)^alpha, broadcast over the batch."""
        return (self.sigma ** 2) * self.eta * (1.0 - t.view(-1, 1, 1, 1)) ** self.alpha

    def grad_datafit(self, x, y, H, H_adj):
        return H_adj(H(x) - y) / (self.sigma ** 2)

    def denoise(self, x, t, labels=None):
        v = self.model(x, t.reshape(-1), labels) if self.conditional else self.model(x, t.reshape(-1))
        return x + (1.0 - t.view(-1, 1, 1, 1)) * v

    @torch.no_grad()
    def reconstruct(self, y, H, H_adj, labels=None, grad_H=None, grad_H_adj=None):
        """Run the N PnP-Flow steps from x = H^T y. grad_H / grad_H_adj replace the operator in the data term only
        (attacker with a mis-specified blur kernel)."""
        gH, gHa = grad_H or H, grad_H_adj or H_adj
        x = H_adj(y).to(self.device)
        delta = 1.0 / self.steps
        for n in range(self.steps):
            t = torch.ones(len(x), device=self.device) * delta * n
            z = x - self.step_size(t) * self.grad_datafit(x, y, gH, gHa)
            x_new = torch.zeros_like(x)
            tt = t.view(-1, 1, 1, 1)
            for _ in range(self.M):
                z_tilde = tt * z + torch.randn_like(z) * (1.0 - tt)
                x_new += self.denoise(z_tilde, t, labels)
            x = x_new / self.M
        return x


def observe(x, H, sigma_noise, seed, offset=None):
    """The leaked degraded copy y = H x + sigma n. The noise is seeded per batch (seed = batch index), so the same
    images in the same batches always give the same y. The PnP draws continue from this seed. `offset` is added
    to H x before the noise (immunization perturbation)."""
    y = H(x.clone())
    if offset is not None:
        y = y + offset.to(y.device)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return y + torch.randn_like(y) * sigma_noise if sigma_noise > 0 else y
