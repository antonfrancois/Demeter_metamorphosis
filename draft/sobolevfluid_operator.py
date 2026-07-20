import math

import torch


class SobolevFluidOperator(torch.nn.Module):
    r"""Periodic 2D Sobolev fluid operator on ``[B, 2, H, W]`` pixel fields.

    The operator and its inverse are

    .. math::
        L v = -\alpha \Delta v - \beta \nabla(\nabla \cdot v) + \gamma v,
        \qquad K = L^{-1}.

    Periodic finite differences are diagonalized with a 2D FFT. For unit pixel
    spacing and frequencies :math:`\theta_j = 2\pi k_j/N_j`, define

    .. math::
        q_j = 4\sin^2(\theta_j/2) = 2-2\cos(\theta_j) \approx \xi_j^2,
        \qquad s_j = \sin(\theta_j) \approx \xi_j.

    With :math:`\lambda=\gamma+\alpha(q_x+q_y)`, the exact discrete symbol is

    .. math::
        \widehat L =
        \begin{pmatrix}
        \lambda+\beta q_x & \beta s_xs_y \\
        \beta s_xs_y & \lambda+\beta q_y
        \end{pmatrix}
        \approx (\gamma+\alpha|\xi|^2)I + \beta\xi\xi^T.

    ``apply_operator`` multiplies by this matrix at each frequency;
    ``apply_inverse`` applies its closed-form ``2 x 2`` inverse.
    """

    def __init__(self, alpha=0.5, beta=0.5, gamma=0.001):
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self._symbol_cache = None

        if self.alpha < 0 or self.beta < 0 or self.gamma <= 0:
            raise ValueError("alpha and beta must be non-negative, and gamma must be positive")

    def _symbol(self, field):
        height, width = field.shape[-2:]
        key = (
            height, width, field.device, field.dtype,
            self.alpha, self.beta, self.gamma,
        )
        if self._symbol_cache is not None and self._symbol_cache[0] == key:
            return self._symbol_cache[1]

        theta_y = 2 * math.pi * torch.fft.fftfreq(
            height, device=field.device, dtype=field.dtype
        )[:, None]
        theta_x = 2 * math.pi * torch.fft.rfftfreq(
            width, device=field.device, dtype=field.dtype
        )[None, :]
        laplace_y = 4 * torch.sin(theta_y / 2).square()
        laplace_x = 4 * torch.sin(theta_x / 2).square()
        sin_y = torch.sin(theta_y)
        sin_x = torch.sin(theta_x)
        sin_y[0] = 0
        sin_x[:, 0] = 0
        if height % 2 == 0:
            sin_y[height // 2] = 0
        if width % 2 == 0:
            sin_x[:, -1] = 0

        diagonal = self.gamma + self.alpha * (laplace_x + laplace_y)
        symbol = (
            diagonal + self.beta * laplace_x,
            self.beta * sin_x * sin_y,
            diagonal + self.beta * laplace_y,
        )
        l_xx, l_xy, l_yy = symbol
        determinant = l_xx * l_yy - l_xy.square()
        inverse_symbol = (
            l_yy / determinant,
            -l_xy / determinant,
            l_xx / determinant,
        )
        self._symbol_cache = key, symbol, inverse_symbol
        return symbol

    def _inverse_symbol(self, field):
        self._symbol(field)
        assert self._symbol_cache is not None
        return self._symbol_cache[2]

    @staticmethod
    def _check_field(field):
        if field.ndim != 4 or field.shape[1] != 2:
            raise ValueError(f"field must have shape [B, 2, H, W], got {field.shape}")
        if not torch.is_floating_point(field):
            raise TypeError(f"field must be floating point, got {field.dtype}")

    def apply_operator(self, field):
        """Apply ``L = -alpha Laplacian - beta grad div + gamma Id``."""
        self._check_field(field)
        field_hat = torch.fft.rfft2(field)
        l_xx, l_xy, l_yy = self._symbol(field)
        result_hat = torch.stack(
            (
                l_xx * field_hat[:, 0] + l_xy * field_hat[:, 1],
                l_xy * field_hat[:, 0] + l_yy * field_hat[:, 1],
            ),
            dim=1,
        )
        return torch.fft.irfft2(result_hat, s=field.shape[-2:])

    def apply_inverse(self, field):
        """Apply ``K = L^-1``."""
        self._check_field(field)
        field_hat = torch.fft.rfft2(field)
        k_xx, k_xy, k_yy = self._inverse_symbol(field)
        result_hat = torch.stack(
            (
                k_xx * field_hat[:, 0] + k_xy * field_hat[:, 1],
                k_xy * field_hat[:, 0] + k_yy * field_hat[:, 1],
            ),
            dim=1,
        )
        return torch.fft.irfft2(result_hat, s=field.shape[-2:])

    def forward(self, field):
        return self.apply_inverse(field)

    def extra_repr(self):
        return f"alpha={self.alpha}, beta={self.beta}, gamma={self.gamma}, boundary='periodic'"

    def init_kernel(self, image):
        if image.ndim != 4:
            raise ValueError(f"SobolevFluidOperator supports only 2D images, got {image.shape}")

    def get_all_arguments(self):
        return {
            "name": self.__class__.__name__,
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
        }
