from typing import Tuple, List
import torch
from torch.nn.functional import pad
import kornia
# from kornia.filters.kernels import normalize_kernel2d
import kornia.filters.filter as flt
from math import prod, sqrt, log

from .fft_conv import fft_conv
from .decorators import deprecated

def fft_filter(input: torch.Tensor, kernel: torch.Tensor,
             border_type: str = 'reflect',
             normalized: bool = False) -> torch.Tensor:
    r"""Function that convolves a tensor with a kernel. This function is almost
    the function filter2d from kornia, adapted to work with 2d and 3d tensors.

    The function applies a given kernel to a tensor. The kernel is applied
    independently at each depth channel of the tensor. Before applying the
    kernel, the function applies padding according to the specified mode so
    that the output remains in the same shape.

    Args:
        input (torch.Tensor): the input tensor with shape of
          :math:`(B, C, H, W)` or :math:`(B, C, D, H, W)`
        kernel (torch.Tensor): the kernel to be convolved with the input
          tensor. The kernel shape must be :math:`(1, kH, kW)` or :math:`(1, kD, kH, kW)`.
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

    len_input_shape = len(input.shape)
    len_kernel_shape = len(kernel.shape)
    if not len_input_shape in [4,5] :
        raise ValueError("Invalid input shape, we expect BxCxHxW or BxCxDxHxW. Got: {}"
                         .format(input.shape))

    if not len_kernel_shape in [3,4]:
        raise ValueError("Invalid kernel shape, we expect 1xkHxkW or 1xkDxkHxkW. Got: {}"
                         .format(kernel.shape))

    if  len_input_shape != len_kernel_shape + 1:
        raise ValueError("Size of input and kernel do not match,"
                        +"we expect input.Size() == kernel.Size() + 1, "
                        +"got {} and {}".format(input.shape,kernel.shape) )

    borders_list: List[str] = ['constant', 'reflect', 'replicate', 'circular']
    if border_type not in borders_list:
        raise ValueError("Invalid border_type, we expect the following: {0}."
                         "Got: {1}".format(borders_list, border_type))

    # prepare kernel
    # b, c, h, w = input.shape
    c = input.shape[1]
    tmp_kernel: torch.Tensor = kernel.unsqueeze(0).to(input.device).to(input.dtype)
    if normalized and len_kernel_shape == 3:
        tmp_kernel = kornia.filters.kernels.normalize_kernel2d(tmp_kernel)[None]
    elif normalized and len_kernel_shape == 4:
        norm = tmp_kernel.abs().sum([2,3,4])
        tmp_kernel = tmp_kernel / norm[...,None,None]

    # pad the input tensor
    padding_shape: List[int] = flt._compute_padding(tmp_kernel.shape[2:])
    input_pad: torch.Tensor = pad(input, padding_shape, mode=border_type)
    #b, c, hp, wp = input_pad.shape
    expand_list = len_input_shape * [-1]
    expand_list[0] = c
    return fft_conv(input_pad,tmp_kernel.expand(expand_list),
                    groups=c, padding=0, stride=1)

class GaussianRKHS(torch.nn.Module):
    """ Is equivalent to a gaussian blur. This function support 2d and 3d images in the
    PyTorch convention

    Args :


        border_type (str): the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``. Default: ``'reflect'``.
        normalized (bool): If True, kernel will be L1 normalized.

    Return:
        torch.Tensor: the convolved tensor of same size and numbers of channels
        as the input.

    Test
    # #import matplotlib.pyplot as plt
    # import numpy as np
    # import nibabel as nib
    # import image_3d_visualisation as iv3
    # import reproducing_kernels as rkhs
    # import torch
    #
    # %matplotlib qt
    # irm_type = 'flair'
    # folder_name = 'Brats18_CBICA_APY_1'
    # img = nib.load(ROOT_DIRECTORY+'/../data/brats/'+folder_name+'/'+folder_name+'_'+irm_type+'.nii.gz')
    # # img.affine
    # img_data = torch.Tensor(img.get_fdata())[None,None]
    # sigma = (3,5,5)
    # # img_data = torch.Tensor(img.get_fdata()[125])[None,None]
    # # sigma = (5,5)
    # blured = rkhs.GaussianRKHS(sigma)(img_data)
    # # fig,ax = plt.subplots(1,2)
    # # ax[0].imshow(img_data[0,0])
    # # ax[1].imshow(blured[0,0])
    # iv3.imshow_3d_slider(img_data)
    # iv3.imshow_3d_slider(blured)
    """
    def __init__(self,sigma : Tuple,
                 border_type: str = 'replicate',
                 device = 'cpu'):
        """

        :param sigma: (Tuple[float,float] or [float,float,float])
        :border_type: the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``,
          ``'replicate'`` or ``'circular'``.
          the ``'reflect'`` one is not implemented yet by pytorch
        """
        big_odd = lambda val : max(6,int(val*6)) + (1 - max(6,int(val*6)) %2)
        kernel_size = tuple([big_odd(s) for s in sigma])
        self.sigma = sigma
        super().__init__()
        self._dim = len(sigma)
        if self._dim == 2:
            self.kernel = kornia.filters.get_gaussian_kernel2d(kernel_size,sigma)#[None]
            self.kernel *= prod(sigma)
            self.filter = flt.filter2d
        elif self._dim == 3:
            self.kernel = get_gaussian_kernel3d(kernel_size,sigma)#[None]
            self.kernel *= prod(sigma)
            # self.filter = flt.filter3d
            self.filter = fft_filter
        else:
            raise ValueError("Sigma is expected to be a tuple of size 2 or 3 same as the input dimension,"
                             +"len(sigma) == {}".format(len(sigma)))
        self.border_type = border_type

        # TODO : define better the condition for using the fft filter
        # this filter works in 2d and 3d
        if max(kernel_size) > 7:
            self.filter = fft_filter
        # print(f"filter used : {self.filter}")

    def __repr__(self) -> str:
        # the if is there for compatibilities with older versions
        sig_str= f'sigma :{self.sigma}' if hasattr(self,'sigma') else ''
        return self.__class__.__name__+\
        ','+str(self._dim)+'D '+\
        f'\n\tfilter :{self.filter.__name__}, '+sig_str+\
        f'\n\tkernel_size :{tuple(self.kernel.shape)}'

    def forward(self, input: torch.Tensor):
        """

        :param input: (torch.Tensor): the input tensor with shape of
          :math:`(B, C, H, W)` or :math:`(B, C, D, H, W)`
        :return:
        """
        if (self._dim == 2 and len(input.shape) == 4) or (self._dim == 3 and len(input.shape) == 5):
            return self.filter(input,self.kernel,self.border_type)
        else:
            raise ValueError(f"{self.__class__.__name__} was initialized "
                             f"with a {self._dim}D mask and input shape is : "
                             f"{input.shape}")

def get_gaussian_kernel3d(kernel_size,sigma):
    print(f"kernel_size : {kernel_size}")
    print(f"sigma : {sigma}")
    if not isinstance(kernel_size, tuple) or len(kernel_size) != 3:
        raise TypeError(f"kernel_size must be a tuple of length three. Got {kernel_size}")
    if not isinstance(sigma, tuple) or len(sigma) != 3:
        raise TypeError(f"sigma must be a tuple of length three. Got {sigma}")
    ksize_d, ksize_h, ksize_w= kernel_size
    #print(f"ksize_d : {ksize_d}, ksize_h : {ksize_h}, ksize_w : {ksize_w}")
    #print(f"sigma : {sigma}, {torch.tensor(sigma)}")
    sigma_d, sigma_h, sigma_w = [ float(s) for s in sigma]
    #print(f"sigma_d : {sigma_d}, sigma_h : {sigma_h}, sigma_w : {sigma_w}")
    kernel_d: torch.Tensor = kornia.filters.kernels.get_gaussian_kernel1d(ksize_d, sigma_d)
    kernel_h: torch.Tensor = kornia.filters.kernels.get_gaussian_kernel1d(ksize_h, sigma_h)
    kernel_w: torch.Tensor = kornia.filters.kernels.get_gaussian_kernel1d(ksize_w, sigma_w)
    # kernel_2d: torch.Tensor = kernel_d[:,None] * kernel_h[None]
    # kernel_3d: torch.Tensor = kernel_2d[:,:,None] * kernel_w[None,None]
    kerned_2d = torch.matmul(kernel_d.T,kernel_h)[...,None]
    kernel_3d = torch.matmul(kerned_2d,kernel_w)[None]
    return kernel_3d

@deprecated("Please use GaussianRKHS instead.")
class GaussianRKHS2d(GaussianRKHS):

    def __init__(self,
                 sigma: Tuple[float, float],
                 border_type: str = 'reflect') -> None:

        super(GaussianRKHS2d, self).__init__(sigma,border_type)

class Multi_scale_GaussianRKHS(torch.nn.Module):

    def __init__(self, list_sigmas):
        if isinstance(list_sigmas,tuple):
            raise ValueError("List sigma must be a list of tuple, if you want to use "
                             "a single scale Gaussian RKHS please use the class "
                             "GaussianRKHS instead.")
        super(Multi_scale_GaussianRKHS, self).__init__()
        _ks = []
        for sigma in list_sigmas:
            print("sigma : ",sigma)
            big_odd = lambda val : max(6,int(val*6)) + (1 - max(6,int(val*6)) %2)
            kernel_size = tuple([big_odd(s) for s in sigma])
            _ks.append(kernel_size)
        # Get the max of each dimension.
        self._dim = len(kernel_size)
        kernel_size = tuple([
            max([s[i] for s in _ks ]) for i in range(self._dim)
        ])
        self.list_sigma = list_sigmas

        if self._dim == 2:
            kernel_f = kornia.filters.get_gaussian_kernel2d
            self.filter = fft_filter if max(kernel_size) > 7 else flt.filter2d
        elif self._dim == 3:
            kernel_f = get_gaussian_kernel3d
            self.filter = fft_filter
        else:
            raise ValueError("Sigma is expected to be a tuple of size 2 or 3 same as the input dimension,"
                             +"len(sigma[0]) == {}".format(len(list_sigmas[0])))

        self.kernel = torch.cat(
            [ prod(sigma)*kernel_f(kernel_size,sigma)[None] for sigma in list_sigmas ]
        ).sum(dim=0)#[None]
        self.kernel /= len(list_sigmas)
        self.border_type = 'constant'

    def __repr__(self) -> str:
        # the if is there for compatibilities with older versions
        return self.__class__.__name__+\
                f'(sigma :{self.list_sigma})'
        # f'filter :{self.filter.__name__}, '+sig_str
        # ','+str(self._dim)+'D '+\

    def __call__(self, input : torch.Tensor):
        if (self._dim == 2 and len(input.shape) == 4) or (self._dim == 3 and len(input.shape) == 5):
            return self.filter(input,self.kernel,self.border_type)
        else:
            raise ValueError(f"{self.__class__.__name__} was initialized "
                             f"with a {self._dim}D mask and input shape is : "
                             f"{input.shape}")


class Multi_scale_GaussianRKHS_notAverage(torch.nn.Module):

    def __init__(self, list_sigmas):
        if isinstance(list_sigmas,tuple):
            raise ValueError("List sigma must be a list of tuple, if you want to use "
                             "a single scale Gaussian RKHS please use the class "
                             "GaussianRKHS instead.")
        super(Multi_scale_GaussianRKHS_notAverage, self).__init__()

        self.gauss_list = []
        for sigma in list_sigmas:
            self.gauss_list.append(GaussianRKHS(sigma))



    def __repr__(self) -> str:
        # the if is there for compatibilities with older versions
        return self.__class__.__name__+\
                f'(sigma :{self.list_sigma})'
        # f'filter :{self.filter.__name__}, '+sig_str
        # ','+str(self._dim)+'D '+\

    def __call__(self, input : torch.Tensor):
        output = torch.zeros(input.shape,device=input.device)
        for gauss_rkhs in self.gauss_list:
            output += gauss_rkhs(input)/len(self.gauss_list)
        return output

def _get_sigma_monodim(X,nx,c=.1):
    """
    :param X: int : size of the image
    :param nx: int : number subdivisions of the grid
    :param c: float : value considered as negligible in the gaussian kernel
    :return: float : sigma
    """
    return sqrt(- (X/nx)**2 / 2 * log(c))

def get_sigma_from_img_ratio(img_shape,subdiv,c=.1):
    """The function get_sigma_from_img_ratio calculates the ideal
    (\sigma) values for a Gaussian kernel based on the desired grid
     granularity. Given an image (I) of size (H, W), the goal is to
    divide the image into a grid of (n_h) (in the H direction) and
     (n_w) (in the W direction). Suppose (x) is at the center of a
    square in this (n_h \times n_w) grid. We want to choose
    (\sigma = (\sigma_h, \sigma_w))
     such that the Gaussian centered at (x) is negligible outside
      the grid square.

    In other words, we want to find (\sigma) such that:

    [ e^{\frac{ -\left(\frac{H}{n_h}\right)^2}{2 \sigma^2}} < c; \qquad c \in \mathbb{R} ]

     where (c) is the negligibility constant.


    :param img_shape: torch.Tensor or Tuple[int] : shape of the image
    :param subdiv: int or Tuple[int] or List[Tuple[float]] or List[List[int]] :
        meant to encode the number of subdivisions of the grid, lets details
        the different cases:
         - int : the grid is divided in the same number of subdivisions in each direction
         - Tuple[int] : the grid is divided in the number of subdivisions given in the tuple
                            according to the dimensions of the image
         - List[Tuple[float]] : If a tuple is given, we consider that it contains values of sigmas
         - List[List[int]] : If a list of list is given, we consider that each element of the list
                            is a list of integers that represent the number of subdivisions in each direction
                            we simply apply the 'int case' to each element of the list.
    :param c: float : value considered as negligible in the gaussian kernel
    """
    # TODO : Ajouter un warning en disant que le sigma choisi va produire un kernel plus
    #  grand que la taille de l'image ... Ou voir pourquoi ce n'est pas possible et pourquoi
    # ça le fait quand même.



    def _check_subdiv_tuple_size(subdiv_, dim):
        if len(subdiv_) != dim:
            raise ValueError(f"input subdiv was given as a tuple {subdiv_},"
                             f" must be of length {dim} to match the"
                             f" image shape {img_shape}")
        for i in range(dim):
            # if not isinstance(subdiv_[i], int):
            #     raise ValueError(f"subdivisions must be integers")
            if subdiv_[i] > img_shape[i]:
                raise ValueError(f"subdivisions {subdiv_} must be smaller than the image shape {img_shape}")


    if isinstance(img_shape,torch.Tensor):
        img_shape = img_shape.shape
    if len(img_shape) in [4,5]: # 2D or 3D image
        img_shape = img_shape[2:]
    d = len(img_shape)
    if isinstance(subdiv,int):
        subdiv = [(subdiv,)*d]
    elif isinstance(subdiv,tuple):
        _check_subdiv_tuple_size(subdiv,d)
        subdiv = [subdiv]
    elif isinstance(subdiv,list):
        for sb in subdiv:
            if isinstance(sb,list):

                for j in range(len(sb)):
                    if not isinstance(sb[j],int):
                        raise ValueError(f"subdiv was given as a list of lists {subdiv},"
                                         f" all elements must be integers")
                    sb[j] = get_sigma_from_img_ratio(img_shape,sb[j],c)

            elif isinstance(sb,tuple):
                _check_subdiv_tuple_size(sb,d)
        return subdiv

    if len(subdiv) == 1:
        sigma = tuple([_get_sigma_monodim(u,s,c) for s,u in zip(subdiv[0],img_shape)])
        return sigma
    else:
        sigma = [
            tuple([_get_sigma_monodim(u,s,c) for s,u in zip(sb,img_shape)])
            for sb in subdiv
        ]

        return sigma

