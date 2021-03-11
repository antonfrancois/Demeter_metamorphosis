from typing import Tuple, List
import torch
from torch.nn.functional import pad
import kornia
from kornia.filters.kernels import normalize_kernel2d
from kornia.filters.filter import compute_padding

from fft_conv import fft_conv

def fft_filter2D(input: torch.Tensor, kernel: torch.Tensor,
             border_type: str = 'reflect',
             normalized: bool = False) -> torch.Tensor:
    r"""Function that convolves a tensor with a kernel. This function is almost
    the function filter2D from kornia.

    The function applies a given kernel to a tensor. The kernel is applied
    independently at each depth channel of the tensor. Before applying the
    kernel, the function applies padding according to the specified mode so
    that the output remains in the same shape.

    Args:
        input (torch.Tensor): the input tensor with shape of
          :math:`(B, C, H, W)`.
        kernel (torch.Tensor): the kernel to be convolved with the input
          tensor. The kernel shape must be :math:`(1, kH, kW)`.
        border_type (str): the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``. Default: ``'reflect'``.
        normalized (bool): If True, kernel will be L1 normalized.

    Return:
        torch.Tensor: the convolved tensor of same size and numbers of channels
        as the input.
    """
    if not isinstance(input, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}"
                        .format(type(input)))

    if not isinstance(kernel, torch.Tensor):
        raise TypeError("Input kernel type is not a torch.Tensor. Got {}"
                        .format(type(kernel)))

    if not isinstance(border_type, str):
        raise TypeError("Input border_type is not string. Got {}"
                        .format(type(kernel)))

    if not len(input.shape) == 4:
        raise ValueError("Invalid input shape, we expect BxCxHxW. Got: {}"
                         .format(input.shape))

    if not len(kernel.shape) == 3:
        raise ValueError("Invalid kernel shape, we expect 1xHxW. Got: {}"
                         .format(kernel.shape))

    borders_list: List[str] = ['constant', 'reflect', 'replicate', 'circular']
    if border_type not in borders_list:
        raise ValueError("Invalid border_type, we expect the following: {0}."
                         "Got: {1}".format(borders_list, border_type))

    # prepare kernel
    b, c, h, w = input.shape
    tmp_kernel: torch.Tensor = kernel.unsqueeze(0).to(input.device).to(input.dtype)
    if normalized:
        tmp_kernel = normalize_kernel2d(tmp_kernel)
    # pad the input tensor
    height, width = tmp_kernel.shape[-2:]
    padding_shape: List[int] = compute_padding((height, width))
    input_pad: torch.Tensor = pad(input, padding_shape, mode=border_type)
    #b, c, hp, wp = input_pad.shape

    return fft_conv(input_pad,tmp_kernel.expand(c, -1, -1, -1), groups=c, padding=0, stride=1)


class GaussianRKHS2d(kornia.filters.GaussianBlur2d):

    def __init__(self,
                 sigma: Tuple[float, float],
                 border_type: str = 'reflect') -> None:
        # we deduce the kernel size from sigma,
        # and make sure it is odd
        kernel_size_h = max(6,int(sigma[0]*6))
        kernel_size_h += (1 - kernel_size_h %2)

        kernel_size_w = max(6,int(sigma[1]*6))
        kernel_size_w += (1 - kernel_size_w %2)
        kernel_size = (kernel_size_h,kernel_size_w)

        super().__init__(kernel_size,sigma,border_type)
        # TODO : define better the condition for using the fft filter
        if max(self.kernel_size) > 7:
            self.filter = fft_filter2D
        else:
            self.filter = kornia.filter2D

    def forward(self, x: torch.Tensor):  # type: ignore
        return self.filter(x,self.kernel,self.border_type)