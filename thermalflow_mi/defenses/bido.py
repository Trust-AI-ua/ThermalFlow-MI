"""BiDO-HSIC (Peng et al., "Bilateral Dependency Optimization: Defending Against Model-inversion Attacks", KDD 2022).

    loss = CE + lambda_x * sum_l HSIC(X, Z_l) - lambda_y * sum_l HSIC(Y, Z_l)
lowers the dependence between the input X and each hidden representation Z_l (less to invert) while keeping the
dependence between Z_l and the label Y (accuracy). HSIC is the biased estimator tr(K_X H K_Z H) / (m - 1)^2 with
Gaussian kernels (median-distance bandwidth) on inputs and features and a linear kernel on the one-hot label. The
hidden representations are the outputs of the four ResNet stages. Paper setting: lambda_x = 5, lambda_y = 50
(chosen so that the accuracy cost equals that of NLS).
"""
import torch
import torch.nn.functional as F


def centred_kernel(x, gaussian=True):
    m = x.shape[0]
    if gaussian:
        xx = (x * x).sum(1, keepdim=True)
        d = (xx + xx.T - 2 * x @ x.T).clamp_min(0)
        med = d[~torch.eye(m, dtype=torch.bool, device=x.device)].median().detach().clamp_min(1e-12)
        k = torch.exp(-d / med)
    else:
        k = x @ x.T
    h = torch.eye(m, dtype=x.dtype, device=x.device) - 1.0 / m
    return k @ h


def hsic(a, b, gaussian_b=True):
    m = a.shape[0]
    return (centred_kernel(a) * centred_kernel(b, gaussian_b).T).sum() / (m - 1) ** 2


class BiDOLoss(torch.nn.Module):
    """Use with attach(model): forward hooks catch the input and the stage outputs during training."""

    def __init__(self, lambda_x=5.0, lambda_y=50.0):
        super().__init__()
        self.lx, self.ly = float(lambda_x), float(lambda_y)
        self.cache = {}

    def attach(self, model, stages=None):
        stages = stages or (model.layer1, model.layer2, model.layer3, model.layer4)

        def pre(_, inp):
            if model.training:
                self.cache.update(training=True, x=inp[0], z=[])
        model.register_forward_pre_hook(pre)
        for s in stages:
            s.register_forward_hook(lambda _, __, out: self.cache["z"].append(out) if model.training else None)
        return model

    def forward(self, logits, labels):
        ce = F.cross_entropy(logits, labels)
        if not self.cache.get("training"):
            return ce
        hx = hy = 0.0
        with torch.autocast(device_type=logits.device.type, enabled=False):
            x = self.cache["x"].flatten(1).double()
            y = F.one_hot(labels, logits.shape[1]).double()
            for z in self.cache["z"]:
                z = z.flatten(1).double()
                hx = hx + hsic(z, x)
                hy = hy + hsic(z, y, gaussian_b=False)
        self.cache.clear()
        return ce + (self.lx * hx - self.ly * hy).float()
