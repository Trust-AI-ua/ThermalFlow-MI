"""The four degradations of the paper and their measurement noise.

    name                 operator                                   noise sigma (images in [-1, 1])
    gaussian_blur        Gaussian blur, sigma 1, 61 x 61 kernel     0.05
    motion_blur          motion blur, 15 px, 0 deg                  0.05
    box_inpainting       40 x 40 centre box missing                 0.05
    random_inpainting    70% of pixels missing                      0.01
"""
from .base import Degradation, FFTConvolution
from .box_inpainting import BoxInpainting
from .gaussian_blur import GaussianBlur, gaussian_kernel
from .motion_blur import MotionBlur, motion_kernel
from .random_inpainting import RandomInpainting

DEGRADATIONS = ("gaussian_blur", "motion_blur", "box_inpainting", "random_inpainting")
NOISE_SIGMA = {"gaussian_blur": 0.05, "motion_blur": 0.05, "box_inpainting": 0.05, "random_inpainting": 0.01}


def build_degradation(name, num_channels=1, image_size=128, device="cuda", sigma_blur=1.0, motion_length=15,
                      motion_angle=0.0):
    """Returns (operator, noise sigma). sigma_blur / motion_length / motion_angle change the blur kernel
    (the sensitivity study uses them for the attacker's assumed kernel)."""
    if name == "gaussian_blur":
        op = GaussianBlur(sigma_blur, 61, num_channels, image_size, device)
    elif name == "motion_blur":
        op = MotionBlur(motion_length, motion_angle, num_channels, image_size, device)
    elif name == "box_inpainting":
        op = BoxInpainting(20 * image_size // 128)
    elif name == "random_inpainting":
        op = RandomInpainting(0.7)
    else:
        raise ValueError(f"unknown degradation {name!r}; choose from {DEGRADATIONS}")
    return op, NOISE_SIGMA[name]
