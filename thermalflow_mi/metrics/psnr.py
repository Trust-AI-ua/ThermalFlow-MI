"""Peak signal-to-noise ratio (dB) between a reconstruction and the clean image, both 8-bit grayscale in [0, 1]."""
from skimage.metrics import peak_signal_noise_ratio

from ..utils.images import load_gray


def psnr(reference, candidate):
    """reference / candidate: arrays in [0, 1] or image paths."""
    a = load_gray(reference) if isinstance(reference, str) else reference
    b = load_gray(candidate) if isinstance(candidate, str) else candidate
    return float(peak_signal_noise_ratio(a, b, data_range=1.0))
