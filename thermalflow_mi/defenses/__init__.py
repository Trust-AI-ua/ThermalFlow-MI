"""Defences evaluated in the paper.

Model-side (change how the target classifier is trained):
    nls.py           negative label smoothing
    bido.py          BiDO-HSIC
    dp_sgd.py        differentially private training (DP-SGD)
Data-side (change the released image):
    immunization.py  adversarial perturbation of the degraded copy against the attacker's own solver
"""
from .bido import BiDOLoss
from .dp_sgd import train_dp
from .immunization import craft
from .nls import NLSLoss
