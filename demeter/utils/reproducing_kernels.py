from typing import Tuple, List
import torch
from torch.nn.functional import pad
import kornia
# from kornia.filters.kernels import normalize_kernel2d
import kornia.filters.filter as flt
from math import prod, sqrt, log
import matplotlib.pyplot as plt

from .fft_conv import fft_conv
from .decorators import deprecated

def fft_filter(input: torch.Tensor, kernel: torch.Tensor,
             border_type: str = 'constant',
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


def get_gaussian_kernel1d(sigma,dx = 1, kernel_size = None, device='cpu', dtype=torch.float):
    r"""Function that returns Gaussian filter coefficients.

    Args:
        kernel_size (int): the size of the kernel.
        dx (float): the spacing between the kernel points.
        sigma (float, Tensor): the standard deviation of the kernel.
        device (torch.device): the desired device of the kernel.
        dtype (torch.dtype): the desired data type of the kernel.

    Returns:
        torch.Tensor: 1D tensor with the filter coefficients.

    Shape:
        - Output: :math:`(K,)`
    """
    if kernel_size is None:
        kernel_size = max(6,int(sigma*6/dx)) + (1 - max(6,int(sigma*6/dx)) %2)
    if isinstance(sigma,int):
        sigma = float(sigma)
    if isinstance(sigma, float):
        sigma = torch.tensor([[sigma]], device=device, dtype=dtype)
    batch_size = sigma.shape[0]
    # TODO : Remove the dxs
    # x = torch.linspace(-kernel_size *dx / 2, kernel_size *dx/ 2, kernel_size,
    #                    device=device, dtype=dtype
    #                    ).expand(batch_size, -1)
    # linspace correction to ensure that the kernel is centered on zero exactly
    # x = x - x[0,torch.argmin(x.abs())]
    x = (torch.arange(kernel_size, device=sigma.device, dtype=sigma.dtype)
         - kernel_size // 2).expand(batch_size, -1)

    if kernel_size % 2 == 0:
        x = x + 0.5

    kernel = torch.exp(-x.pow(2.0) / (2 * sigma.pow(2.0)))

    return kernel

def get_gaussian_kernel2d( sigma,dx =(1.,1.), kernel_size=None):
    r"""Function that returns Gaussian filter coefficients.

    Args:
        kernel_size (Tuple[int, int]): the size of the kernel.
        dx (Tuple[float, float]): the spacing between the kernel points.
        sigma (Tuple[float, float]|torch.Tensor): the standard deviation of the kernel.

    Returns:
        torch.Tensor: 2D tensor with the filter coefficients.

    Shape:
        - Output: :math:`(H, W)`
    """
    ksize_h, ksize_w = kernel_size if kernel_size is not None else (None,)*2
    sigma_h, sigma_w = [float(s) for s in sigma]
    dh, dw = [float(d) for d in dx]
    kernel_h: torch.Tensor = get_gaussian_kernel1d(sigma_h, dh, ksize_h)
    kernel_w: torch.Tensor = get_gaussian_kernel1d(sigma_w, dw, ksize_w)
    print(f"kernel_h : {kernel_h.shape}, kernel_w : {kernel_w.shape}")
    # kernel_2d: torch.Tensor = torch.matmul(kernel_h[...,None], kernel_w)
    print(kernel_h.shape)
    kernel_2d = kernel_h[:,:, None] * kernel_w[:,None, :]
    return kernel_2d

def get_gaussian_kernel3d(sigma, dx=(1.,)*3 , kernel_size =None):
    print(f"kernel_size : {kernel_size}")
    print(f"sigma : {sigma}")
    if kernel_size is None:
        kernel_size = (None,)*3
    if not isinstance(kernel_size, tuple) or len(kernel_size) != 3:
        raise TypeError(f"kernel_size must be a tuple of length three. Got {kernel_size}")
    if not ( isinstance(sigma, tuple) or isinstance(sigma,torch.Tensor)) or len(sigma) != 3:
        raise TypeError(f"sigma must be a tuple of length three. Got {sigma}")
    ksize_d, ksize_h, ksize_w= kernel_size
    #print(f"ksize_d : {ksize_d}, ksize_h : {ksize_h}, ksize_w : {ksize_w}")
    #print(f"sigma : {sigma}, {torch.tensor(sigma)}")
    sigma_d, sigma_h, sigma_w = [ float(s) for s in sigma]
    dd, dh, dw = [float(d) for d in dx]
    #print(f"sigma_d : {sigma_d}, sigma_h : {sigma_h}, sigma_w : {sigma_w}")
    kernel_d: torch.Tensor = get_gaussian_kernel1d(sigma_d, dd, ksize_d)
    kernel_h: torch.Tensor = get_gaussian_kernel1d(sigma_h, dh,ksize_h)
    kernel_w: torch.Tensor = get_gaussian_kernel1d(sigma_w, dw, ksize_w)
    # kernel_2d: torch.Tensor = kernel_d[:,None] * kernel_h[None]
    # kernel_3d: torch.Tensor = kernel_2d[:,:,None] * kernel_w[None,None]
    print(f"kernel_d : {kernel_d.shape},\n kernel_h : {kernel_h.shape},\n kernel_w : {kernel_w.shape}")
    # kerned_2d = torch.matmul(kernel_d.T,kernel_h)[...,None]
    # kernel_3d = torch.matmul(kerned_2d,kernel_w)[None]

    kernel_3d = (kernel_d[:,:, None, None]
                 * kernel_h[:,None, :, None]
                 * kernel_w[:,None, None, :])

    return kernel_3d

def plot_gaussian_kernel_1d(kernel: torch.Tensor,sigma,ax =None,rotated = False):
    kernel_size = kernel.shape[1]
    dist_from_center = float(kernel_size)/2
    x = torch.linspace(-dist_from_center,dist_from_center,kernel_size)

    kw = {'color':'gray','linestyle':'--'}
    line_1_x = [-sigma,-sigma]
    line_1_y = [0,kernel[0,x > -sigma][0]]
    line_1_kw = kw.copy()
    line_1_kw['label'] = f"K(sigma) ~= {kernel[0,x > -sigma][0]:.3f}"
    line_2_x = [sigma,sigma]
    line_2_y = [0,kernel[0,x < sigma][-1]]
    line_3_x = [x[0],x[-1]]
    line_3_y = [kernel[0,x > -sigma][0],kernel[0,x < sigma][-1]]


    if ax is None:
        fig, ax = plt.subplots()
    ax.set_title(f'kernel sigma: {sigma}')
    if rotated:
        ax.plot(kernel[0].numpy(),x)
        ax.plot(line_1_y,line_1_x,**line_1_kw)
        ax.plot(line_2_y,line_2_x,**kw)
        ax.plot(line_3_y,line_3_x,**kw)
        ax.set_xlabel('K')
        ax.set_ylabel('x')
    else:
        ax.plot(x,kernel[0].numpy())
        ax.plot(line_1_x,line_1_y,**line_1_kw)
        ax.plot(line_2_x,line_2_y,**kw)
        ax.plot(line_3_x,line_3_y,**kw)
        ax.set_xlabel('x')
        ax.set_ylabel('K')
    ax.legend()
    return ax

def plot_gaussian_kernel_2d(kernel: torch.Tensor, sigma, axes=None):

    if axes is None:
        fig, axes = plt.subplots(2, 2, figsize=(10, 5),constrained_layout=True)
    if len(axes.ravel()) == 4:
        axes[0,1].axis('off')
        axes = axes.ravel()[[True,False,True,True]]
    # plot kernel
    axes[1].imshow(kernel[0], cmap='cividis')
    axes[1].set_title(f'Gaussian Kernel $\sigma$={sigma}')
    axes[1].axis('off')

    # plot kernel profile
    print("rfs")
    plot_gaussian_kernel_1d(kernel[:,:,kernel.shape[2]//2], sigma[0], ax=axes[2],rotated=True)
    plot_gaussian_kernel_1d(kernel[:,kernel.shape[1]//2], sigma[1], ax=axes[0])

    return axes

def plot_gaussian_kernel_3d(kernel: torch.Tensor, sigma, ):
    fig = plt.figure(constrained_layout= True,figsize=(15,5))
    subfigs = fig.subfigures(1, 3)

    ax1 = subfigs[0].subplots(2,2)
    ax2 = subfigs[1].subplots(2,2)
    ax3 = subfigs[2].subplots(2,2)
    subfigs[0].suptitle(f"Axial")
    subfigs[1].suptitle(f"Sagittal")
    subfigs[2].suptitle(f"Coronal")

    plot_gaussian_kernel_2d(kernel[...,kernel.shape[3]//2], sigma[1:], axes=ax1)
    plot_gaussian_kernel_2d(kernel[:,:,kernel.shape[2]//2], sigma[::2], axes=ax2)
    plot_gaussian_kernel_2d(kernel[:,kernel.shape[1]//2], sigma[:2], axes=ax3)

def dx_convention_handler(dx_convention, dim):
    if isinstance(dx_convention, str):
        if dx_convention == 'pixel':
            return (1,)*dim
        else:
            raise ValueError(f"Kernel Operator classes can guess 'pixel' dx_convention"
                             f"only,  got {dx_convention}")
    elif isinstance(dx_convention, int) or isinstance(dx_convention, float):
        return (dx_convention,)*dim
    elif isinstance(dx_convention, tuple):
        if len(dx_convention) == dim:
            return dx_convention
        elif len(dx_convention) == 1:
            return dx_convention*dim
        else:
            raise ValueError(f"dx_convention must be a tuple of length {dim},"
                             f" got {len(dx_convention)}")


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
                 border_type: str = 'constant',
                 device = 'cpu'):
        """

        :param sigma: (Tuple[float,float] or [float,float,float])
        :border_type: the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``,
          ``'replicate'`` or ``'circular'``.
          the ``'reflect'`` one is not implemented yet by pytorch
        """
        # big_odd = lambda val : max(6,int(val*6)) + (1 - max(6,int(val*6)) %2)
        # kernel_size = tuple([big_odd(s) for s in sigma])
        self.sigma = sigma
        super().__init__()
        self._dim = len(sigma)
        if self._dim == 2:
            self.kernel = get_gaussian_kernel2d(sigma)#[None]
            # self.kernel *= prod(sigma)
            self.filter = flt.filter2d
        elif self._dim == 3:
            self.kernel = get_gaussian_kernel3d(sigma)#[None]
            self.kernel *= prod(sigma)
            # self.filter = flt.filter3d
            self.filter = fft_filter
        else:
            raise ValueError("Sigma is expected to be a tuple of size 2 or 3 same as the input dimension,"
                             +"len(sigma) == {}".format(len(sigma)))
        self.border_type = border_type

        # TODO : define better the condition for using the fft filter
        # this filter works in 2d and 3d

        if max(self.kernel.shape) > 7:
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

class VolNormalizedGaussianRKHS(torch.nn.Module):
    """

    Args :


        border_type (str): the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``. Default: ``'reflect'``.
        normalized (bool): If True, kernel will be L1 normalized.

    Return:
        torch.Tensor: the convolved tensor of same size and numbers of channels
        as the input.


    """
    def __init__(self,sigma: Tuple,
                 sigma_convention= 'pixel',
                 dx = (1,),
                 border_type: str = 'constant',
                 device = 'cpu'):
        """

        :param sigma: (Tuple[float,float] or [float,float,float]) : the standard
        deviation of the kernel expressed in pixel unit.
        :param sigma_convention: default 'pixel'. expected modes are: {'pixel','continuous'}
        The unit `sigma` input should be considered
        :param dx: (str or Tuple[float,float] or [float,float,float]) : the length of a pixel in each dimension.
        if the image is isotropic, dx_convention can be a float. If the image is anisotropic, dx_convention must be a tuple
        of length equal to the image dimension. The default value is 1.
        :border_type: the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``,
          ``'replicate'`` or ``'circular'``.
          the ``'reflect'`` one is not implemented yet by pytorch
        """
        # big_odd = lambda val : max(6,int(val*6)) + (1 - max(6,int(val*6)) %2)
        # kernel_size = tuple([big_odd(s) for s in sigma])
        self._dim = len(sigma)
        self.dx = dx_convention_handler(dx,self._dim)
        if sigma_convention == 'pixel':
            self.sigma = torch.tensor(sigma)
            self.sigma_continuous = torch.tensor(
                [s*d for d,s in zip(self.dx, sigma)]
            )
        elif sigma_convention == 'continuous':
            self.sigma_continuous = torch.tensor(sigma)
            self.sigma = torch.tensor(
                [s/d for d,s in zip(self.dx, sigma)]
            )
        else:
            raise ValueError("argument sigma_convention must be 'pixel' or 'continuous'"
                             f"got {sigma_convention}")
        super().__init__()

        # print("dx : ",self.dx)
        # print("sigma : ",self.sigma)
        # print("sigma_continuous : ",self.sigma_continuous)
        if self._dim == 2:
            self.kernel = get_gaussian_kernel2d(self.sigma )#[None]

            self.filter = flt.filter2d
        elif self._dim == 3:
            self.kernel = get_gaussian_kernel3d(self.sigma)#[None]
            self.filter = fft_filter
        else:
            raise ValueError("Sigma is expected to be a tuple of size 2 or 3 same as the input dimension,"
                             +"len(sigma) == {}".format(len(sigma)))

        #  We normalize the kernel to multiply by the product of
        # dx / sigma_continuous, which is equal to 1/sigma.
        # ic(prod(self.sigma_continuous))
        self.kernel /=  prod(self.sigma_continuous)
        self.border_type = border_type

        # TODO : define better the condition for using the fft filter
        # this filter works in 2d and 3d
        kernel_size = self.kernel.shape[2:]
        if max(kernel_size) > 7:
            self.filter = fft_filter
        # print(f"filter used : {self.filter}")

    def init_kernel(self,image):
        if isinstance(self.sigma, tuple) and len(self.sigma) != len(image.shape[2:]) :
            raise ValueError(f"kernelOperator :{self.__class__.__name__}"
                             f"was initialised to be {len(self.sigma)}D"
                             f" with sigma = {self.sigma} and got image "
                             f"source.size() = {image.shape}"
                             )
        self.sigma_continuous = self.sigma_continuous.to(image.device)
        self.sigma = self.sigma.to(image.device)


    def __repr__(self) -> str:
        # the if is there for compatibilities with older versions
        sig_str= f'\n\tsigma_pixel :{self.sigma}' if hasattr(self,'sigma') else ''
        sig_str+= f'\n\tsigma_continous :{self.sigma_continuous}' if hasattr(self, 'sigma_continuous') else ''
        return self.__class__.__name__+\
        ','+str(self._dim)+'D '+\
        f'\n\tfilter :{self.filter.__name__}, '+sig_str+\
        f'\n\tkernel_size :{tuple(self.kernel.shape)}'

    def forward(self, input: torch.Tensor):
        """

        :param input: (torch.Tensor): the input tensor with shape of
          :math:`(B, C, H, W)` or :math:`(B, C, D, H, W)`
        :return: (torch.Tensor): output tensor with same size than the input tensor with shape of
          :math:`(B, C, H, W)` or :math:`(B, C, D, H, W)`
        """
        if (self._dim == 2 and len(input.shape) == 4) or (self._dim == 3 and len(input.shape) == 5):
            view_sig = (1,-1) + (1,)*(len(input.shape)-2)
            # input *= self.sigma_continuous.to(input.device).view(view_sig)**2
            convol = self.filter(input,self.kernel,self.border_type)
            convol *= self.sigma.view(view_sig)**2
            return convol
        else:
            raise ValueError(f"{self.__class__.__name__} was initialized "
                             f"with a {self._dim}D mask and input shape is : "
                             f"{input.shape}")

    def plot(self):
        if self._dim == 2:
            plot_gaussian_kernel_2d(self.kernel, self.sigma_continuous)
        elif self._dim == 3:
            plot_gaussian_kernel_3d(self.kernel, self.sigma_continuous)


@deprecated("Please use GaussianRKHS instead.")
class GaussianRKHS2d(VolNormalizedGaussianRKHS):

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

    def plot(self):
        if self._dim == 2:
            plot_gaussian_kernel_2d(self.kernel,self.list_sigma)
        elif self._dim == 3:
            plot_gaussian_kernel_3d(self.kernel,self.list_sigma)

class Multi_scale_GaussianRKHS_notAverage(torch.nn.Module):

    def __init__(self, list_sigmas):
        if isinstance(list_sigmas,tuple):
            raise ValueError("List sigma must be a list of tuple, if you want to use "
                             "a single scale Gaussian RKHS please use the class "
                             "GaussianRKHS instead.")
        super(Multi_scale_GaussianRKHS_notAverage, self).__init__()

        self.gauss_list = []
        for sigma in list_sigmas:
            self.gauss_list.append(VolNormalizedGaussianRKHS(sigma))



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

