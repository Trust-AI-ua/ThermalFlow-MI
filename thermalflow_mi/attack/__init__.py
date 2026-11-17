"""ThermalFlow-MI: model inversion of thermal faces from a leaked degraded copy.

Stage 1 (offline): label the auxiliary data with the target model (labeling.py, targeted attack only) and train a
flow matching prior on it (flow_matching.py, unet.py).
Stage 2: rebuild each private face from its degraded copy with PnP-Flow, the prior acting as the denoiser
(pnp_flow.py, run.py).
"""
from .flow_matching import FlowMatchingTrainer
from .labeling import RelabeledDataset, corrupt_labels, query_labels, save_labels
from .pnp_flow import PnPFlow, observe
from .run import run_attack
from .unet import ConditionalUNet, UNet, build_unet
