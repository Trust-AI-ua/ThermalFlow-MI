"""Linear degradation operators y = H x + n on images in [-1, 1] (B x C x H x W)."""
import torch


class Degradation:
    """H is the forward operator and H_adj its adjoint, used in the data term grad = H^T (H x - y) / sigma^2."""

    def H(self, x):
        raise NotImplementedError

    def H_adj(self, x):
        raise NotImplementedError


class FFTConvolution(Degradation):
    """Circular convolution with a k x k kernel, applied in the Fourier domain. H_adj uses the conjugate filter."""

    def __init__(self, kernel, num_channels=1, image_size=128, device="cuda"):
        self.device = torch.device(device)
        self.kernel = kernel.to(self.device)
        k = self.kernel.shape[-1]
        filt = torch.zeros((1, num_channels, image_size, image_size), device=self.device, dtype=self.kernel.dtype)
        filt[..., :k, :k] = self.kernel
        filt = torch.roll(filt, shifts=(-(k - 1) // 2, -(k - 1) // 2), dims=(2, 3))   # kernel centre at the origin
        self.filter = filt
        self.filter_fft = torch.fft.fft2(filt)

    def H(self, x):
        return torch.real(torch.fft.ifft2(torch.fft.fft2(x.to(self.device)) * self.filter_fft))

    def H_adj(self, x):
        return torch.real(torch.fft.ifft2(torch.fft.fft2(x.to(self.device)) * torch.conj(self.filter_fft)))
