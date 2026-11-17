"""Dataset configuration (configs/datasets/<name>.yaml)."""
from pathlib import Path
from types import SimpleNamespace

import yaml

REPO = Path(__file__).resolve().parents[1]


def load_dataset_config(name_or_path):
    """Load a dataset YAML by name (cftd, charlotte, thermvision) or path. Relative paths in it are resolved
    against the repository root."""
    path = Path(name_or_path)
    if not path.suffix:
        path = REPO / "configs" / "datasets" / f"{name_or_path}.yaml"
    cfg = yaml.safe_load(path.read_text())
    for key in ("data_root", "split_csv", "ood_root"):
        if key in cfg and not Path(cfg[key]).is_absolute():
            cfg[key] = str((REPO / cfg[key]).resolve())
    cfg.setdefault("image_size", 128)
    cfg.setdefault("num_channels", 1)
    return SimpleNamespace(**cfg)


def pnp_setting(cfg, degradation):
    """Step size eta (lr_pnp) and decay alpha of gamma_n = eta * sigma^2 * (1 - t_n)^alpha for one degradation."""
    s = cfg.pnp[degradation]
    return float(s["lr_pnp"]), float(s["alpha"])


# Default locations of trained models and results (all under the repository, all git-ignored).
def classifier_checkpoint(dataset, model, defense="none"):
    sub = "classifiers" if defense in (None, "none") else f"classifiers_{defense}"
    return REPO / "checkpoints" / dataset / sub / model / "final_model.pt"


def prior_dir(dataset, distribution, attack, target_model=None, tag=None):
    """checkpoints/<dataset>/priors/<distribution>/untargeted  or  .../targeted/<target model>[_<tag>]."""
    base = REPO / "checkpoints" / dataset / "priors" / distribution / attack
    if attack == "untargeted":
        return base if tag is None else base.parent / f"untargeted_{tag}"
    return base / (target_model if tag is None else f"{target_model}_{tag}")
