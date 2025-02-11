from collections.abc import Iterable

import torch
import torch.nn.functional as F
from kornia.filters import SpatialGradient, SpatialGradient3d, filter2d, filter3d
from kornia.geometry.transform import resize
from numpy import newaxis
from matplotlib.widgets import Slider

# import decorators
from .toolbox import rgb2gray
from . import bspline as mbs
from . import vector_field_to_flow as vff
from . import decorators as deco
from demeter.constants import *


# from .utils.image_3d_visualisation import image_slice


# ================================================
#        IMAGE BASICS
# ================================================

def reg_open(number, size=None, requires_grad=False, device='cpu'):
    path = ROOT_DIRECTORY
    path += '/examples/im2Dbank/reg_test_' + number + '.png'

    I = rgb2gray(plt.imread(path))
    I = torch.tensor(I[newaxis, newaxis, :],
                     dtype=torch.float,
                     requires_grad=requires_grad,
                     device=device)
    if size is None:
        return I
    else:
        return resize(I, size)


def resize_image(image: torch.Tensor | list[torch.Tensor],
                 scale_factor: float | int | Iterable
                 ):
    """
    Resize an image by a scale factor $s = (s1,s2,s3)$


    :param image: list of tensors [B,C,H,W] or [B,C,D,H,W] torch tensor
    :param scale_factor: float or list or tuple of image dimension size

    : return: tensor of size [B,C,s1*H,s2*W] or [B,C,s1*D, s2*H, s3*W] or list
    containing tensors.
    """
    if isinstance(image, torch.Tensor):
        image = [image]
    device = image[0].device
    Ishape = image[0].shape[2:]
    if isinstance(scale_factor, float | int):
        scale_factor = (scale_factor,) * len(Ishape)
    Ishape_D = tuple([int(s * f) for s, f in zip(Ishape, scale_factor)])
    id_grid = make_regular_grid(Ishape_D, dx_convention='2square').to(device).to(image[0].dtype)
    i_s = []
    for i in image:
        i_s.append(
            torch.nn.functional.grid_sample(i.to(device), id_grid, **DLT_KW_GRIDSAMPLE)
        )
    if len(i_s) == 1:
        return i_s[0]
    return i_s


def image_slice(I, coord, dim):
    """
    Return a slice of the image I at the given coordinate and dimension

    :param I: [H,W,D] numpy array or tensor
    :param coord: int coordinate of the slice, if float it will be casted to int
    :param dim: int in {0,1,2} dimension of the slice
    """
    coord = int(coord)
    if dim == 0:
        return I[coord]
    elif dim == 1:
        return I[:, coord]
    elif dim == 2:
        return I[:, :, coord]


def make_3d_flat(img_3D, slice):
    D, H, W = img_3D.shape

    im0 = image_slice(img_3D, slice[0], 2).T
    im1 = image_slice(img_3D, slice[1], 1).T
    im2 = image_slice(img_3D, slice[2], 0).T

    crop = 20
    # print(D-int(1.7*crop),D+H-int(2.7*crop))
    # print(D+H-int(3.2*crop))
    long_img = np.zeros((D, D + H + H - int(3.5 * crop)))
    long_img[:D, :D - crop] = im0[:, crop // 2:-crop // 2]
    long_img[(D - W) // 2:(D - W) // 2 + W, D - int(1.7 * crop):D + H - int(2.7 * crop)] = im1.numpy()[::-1,
                                                                                           crop // 2:-crop // 2]
    long_img[(D - W) // 2:(D - W) // 2 + W, D + H - int(3 * crop):] = im2.numpy()[::-1, crop // 2:]

    # long_img[long_img== 0] =1
    return long_img


def pad_to_same_size(img_1, img_2):
    """ Pad the two images in order to make images of the same size
    takes

    :param img_1: [T_1,C,D_1,H_1,W_1] or [D_1,H_1,W_1]  torch tensor
    :param img_2: [T_2,C,D_2,H_2,W_2] or [D_2,H_2,W_2]  torch tensor
    :return: will return both images with of shape
    [...,max(D_1,D_2),max(H_1,H_2),max(W_1,W_2)] in a tuple.
    """
    diff = [x - y for x, y in zip(img_1.shape[2:], img_2.shape[2:])]
    padding1, padding2 = (), ()
    for d in reversed(diff):
        is_d_even = d % 2
        if d // 2 < 0:
            padding1 += (-d // 2 + is_d_even, -d // 2)
            padding2 += (0, 0)
        else:
            padding1 += (0, 0)
            padding2 += (d // 2 + is_d_even, d // 2)
    img_1_padded = torch.nn.functional.pad(img_1[0, 0], padding1)[None, None]
    img_2_padded = torch.nn.functional.pad(img_2[0, 0], padding2)[None, None]
    return (img_1_padded, img_2_padded)


def addGrid2im(img, n_line, cst=0.1, method='dots'):
    """ draw a grid to the image

    :param img:
    :param n_line:
    :param cst:
    :param method:
    :return:
    """

    if isinstance(img, tuple):
        img = torch.zeros(img)
    if len(img.shape) == 4:
        _, _, H, W = img.shape
        is_3D = False
    elif len(img.shape) == 5:
        _, _, H, W, D = img.shape
        is_3D = True
    else:
        raise ValueError(f"img should be [B,C,H,W] or [B,C,D,H,W] got {img.shape}")

    try:
        len(n_line)
    except:
        n_line = (n_line,) * (len(img.shape) - 2)

    add_mat = torch.zeros(img.shape)
    row_mat = torch.zeros(img.shape, dtype=torch.bool)
    col_mat = torch.zeros(img.shape, dtype=torch.bool)

    row_centers = (torch.arange(n_line[0]) + 1) * H // n_line[0]
    row_width = int(max(H / 200, 1))
    for lr, hr in zip(row_centers - row_width, row_centers + row_width):
        row_mat[:, :, lr:hr] = True

    col_centers = (torch.arange(n_line[1]) + 1) * W // n_line[1]
    col_width = int(max(W / 200, 1))
    for lc, hc in zip(col_centers - col_width, col_centers + col_width):
        col_mat[:, :, :, lc:hc] = True

    if is_3D:
        depth_mat = torch.zeros(img.shape, dtype=torch.bool)
        depth_centers = (torch.arange(n_line[2]) + 1) * D // n_line[2]
        depth_width = int(max(D / 200, 1))
        for ld, hd in zip(depth_centers - depth_width, depth_centers + depth_width):
            depth_mat[:, :, :, :, ld:hd] = True

    if method == 'lines':
        add_mat[row_mat] += cst
        add_mat[col_mat] += cst
        if is_3D:
            add_mat[depth_mat] += cst
    elif method == 'dots':
        bool = torch.logical_and(row_mat, col_mat)
        if is_3D:
            bool = torch.logical_and(bool, depth_mat)
        add_mat[bool] = cst
    else:
        raise ValueError(f"method must be among `lines`,`dots` got {method}")

    # put the negative grid on the high values image
    add_mat[img > .5] *= -1

    return img + add_mat


def thresholding(image, bounds=(0, 1)):
    return torch.maximum(torch.tensor(bounds[0]),
                         torch.minimum(
                             torch.tensor(bounds[1]),
                             image
                         )
                         )


def spatialGradient(image, dx_convention='pixel'):
    """ Compute the spatial gradient on 2d and 3d images by applying
    a sobel kernel. Perform the normalisation of the gradient according
    to the spatial convention (`dx_convention`) and make it the closer possible
    to the theoretical gradient.

    Parameters
    ----------
    image : Tensor
        [B,C,H,W] or [B,C,D,H,W] tensor.
    dx_convention : str or tensor
        If str, it must be in {'pixel','square','2square'}.
        If tensor, it must be of shape [B,2] or [B,3], where B is the batch size and
        the second dimension is the spatial resolution of the image giving the pixel size.
        Attention : this last values must be in reverse order of the image shape.

    Returns
    -------
    grad_image : Tensor
        [B,C,2,H,W] or [B,C,3,D,H,W] tensor.

    Examples
    --------

    .. code-block:: python

        H,W = (300,400)
        rgi = tb.RandomGaussianImage((H, W), 2, 'square',
                                     a=[-1, 1],
                                      b=[15, 25],
                                     c=[[.3*400 , .3*300], [.7*400, .7*300]])
        image =  rgi.image()
        theoretical_derivative = rgi.derivative()
        print(f"image shape : {image.shape}")
        derivative = tb.spatialGradient(image, dx_convention="square")
        dx = torch.tensor([[1. / (W - 1), 1. / (H - 1)]], dtype=torch.float64)
        derivative_2 = tb.spatialGradient(mage, dx_convention=dx)

    """
    if isinstance(dx_convention, str):
        dx_convention_list = ["pixel", "square", "2square"]

        if not dx_convention in dx_convention_list:
            raise ValueError(f"dx_convention must be one of {dx_convention_list}, got {dx_convention}")
    elif isinstance(dx_convention, tuple):
        dx_convention = torch.tensor(dx_convention)
    elif not isinstance(dx_convention, torch.Tensor):
        raise ValueError(f"dx_convention must be a string or a tensor, got {type(dx_convention)}")
    if len(image.shape) == 4:
        grad_image = spatialGradient_2d(image, dx_convention)
    elif len(image.shape) == 5:
        grad_image = spatialGradient_3d(image, dx_convention)
    else:
        raise ValueError(f"image should be [B,C,H,W] or [B,C,D,H,W] got {image.shape}")

    if isinstance(dx_convention, torch.Tensor):
        B, _, d = grad_image.size()[:3]
        grad_image *= 1. / dx_convention.flip(dims=(0,)).view(B, 1, d, *([1] * d)).to(grad_image.device)
        # grad_image *= 1./dx_convention.view(B, 1, d, *([1] * d)).to(grad_image.device)

        return grad_image
    elif dx_convention == 'square':
        # equivalent to
        # grad_image[0,0,0] *= (W-1)
        # grad_image[0,0,1] *= (H-1)
        # grad_image[0,0,2] *= (D-1)
        # but works for all dim and batches
        B, _, d = grad_image.size()[:3]
        size = torch.tensor(grad_image.size())[3:].flip(dims=(0,)).view(B, 1, d, *([1] * d)).to(grad_image.device)
        grad_image *= size - 1
        return grad_image
    elif dx_convention == '2square':
        # equivalent to
        # grad_image[0,0,0] *= 2/(W-1)
        # grad_image[0,0,1] *= 2/(H-1)
        # grad_image[0,0,2] *= 2/(D-1)
        # but works for all dim and batches

        B, _, d = grad_image.size()[:3]
        size = torch.tensor(grad_image.size())[3:].flip(dims=(0,)).view(B, 1, d, *([1] * d)).to(grad_image.device)
        grad_image *= (size - 1) / 2
        return grad_image
    else:
        return grad_image


def spatialGradient_2d(image, dx_convention='pixel'):
    """
    Compute the spatial gradient on 2d images by applying
    a sobel kernel

    :param image: Tensor [B,C,H,W]
    :param dx_convention:
    :return: [B,C,2,H,W]
    """
    normalized = True  # if dx_convention == "square" else False
    grad_image = SpatialGradient(mode='sobel', normalized=normalized)(image)

    # other normalisation than the pixel one

    # if dx_convention == "square":
    #     _,_,H,W = image.size()
    #     grad_image[:,0,0] *= (W - 1)
    #     grad_image[:,0,1] *= (H - 1)
    # if dx_convention == '2square':
    #     _,_,H,W = image.size()
    #     grad_image[:,0,0] *= (W-1)/2
    #     grad_image[:,0,1] *= (H-1)/2
    return grad_image


def spatialGradient_3d(image, dx_convention='pixel'):
    """
    Compute the spatial gradient on 3d images by applying a sobel kernel

    Parameters
    ---------------

    image : Tensor
        [B,C,D,H,W] tensor
    dx_convention : str or tensor
        If str, it must be in {'pixel','square','2square'}.

    Returns
    ---------------

    Example
    ---------------

    .. code-block:: python

        H,W,D = (50,75,100)
        image = torch.zeros((H,W,D))
        mX,mY,mZ = torch.meshgrid(torch.arange(H),
                              torch.arange(W),
                              torch.arange(D))

        mask_rond = ((mX - H//2)**2 + (mY - W//2)**2).sqrt() < H//4
        mask_carre = (mX > H//4) & (mX < 3*H//4) & (mZ > D//4) & (mZ < 3*D//4)
        mask_diamand = ((mY - W//2).abs() + (mZ - D//2).abs()) < W//4
        mask = mask_rond & mask_carre & mask_diamand
        image[mask] = 1


        grad_image = spacialGradient_3d(image[None,None])
        # grad_image_sum = grad_image.abs().sum(dim=1)
        # iv3d.imshow_3d_slider(grad_image_sum[0])

    """

    B, C, _, _, _ = image.size()
    if C > 1 and B > 1:
        raise ValueError(f"Can't compute gradient on multi channel images with batch size > 1 got {image.size()}")
    if C > 1:
        image = image[0].unsqueeze(1)

    # sobel kernel is not implemented for 3D images yet in kornia
    # grad_image = SpatialGradient3d(mode='sobel')(image)
    kernel = get_sobel_kernel_3d().to(image.device).to(image.dtype)
    # normalise kernel
    kernel = 3 * kernel / kernel.abs().sum()
    spatial_pad = [1, 1, 1, 1, 1, 1]

    image_padded = F.pad(image, spatial_pad, 'replicate').repeat(1, 3, 1, 1, 1)
    grad_image = F.conv3d(image_padded, kernel, padding=0, groups=3, stride=1)
    if C > 1:
        grad_image = grad_image[None]
    else:
        grad_image = grad_image.unsqueeze(1)
    # other normalisation than the pixel one
    # _,_,D,H,W, = image.size()
    # if dx_convention == 'square':
    #     grad_image[0,0,0] *= (W-1)
    #     grad_image[0,0,1] *= (H-1)
    #     grad_image[0,0,2] *= (D-1)
    #     print(f"grad_image min = {grad_image.min()};{grad_image.max()}")
    # if dx_convention == '2square':
    #     # _,_,D,H,W, = image.size()
    #     grad_image[0,0,0] *= (W-1)/2
    #     grad_image[0,0,1] *= (H-1)/2
    #     grad_image[0,0,2] *= (D-1)/2

    return grad_image


def get_sobel_kernel_2d():
    return torch.tensor(
        [
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ])
    # return torch.tensor(
    #     [
    #         [-5,  -4,  0,   4,   5],
    #         [-8, -10,  0,  10,   8],
    #         [-10, -20,  0,  20,  10],
    #         [-8, -10,  0,  10,   8],
    #         [-5, -4,  0,   4,   5]
    #     ]
    # )


def get_sobel_kernel_3d():
    return torch.tensor(
        [
            [[[-1, 0, 1],
              [-2, 0, 2],
              [-1, 0, 1]],

             [[-2, 0, 2],
              [-4, 0, 4],
              [-2, 0, 2]],

             [[-1, 0, 1],
              [-2, 0, 2],
              [-1, 0, 1]]],

            [[[-1, -2, -1],
              [0, 0, 0],
              [1, 2, 1]],

             [[-2, -4, -2],
              [0, 0, 0],
              [2, 4, 2]],

             [[-1, -2, -1],
              [0, 0, 0],
              [1, 2, 1]]],

            [[[-1, -2, -1],
              [-2, -4, -2],
              [-1, -2, -1]],

             [[0, 0, 0],
              [0, 0, 0],
              [0, 0, 0]],

             [[1, 2, 1],
              [2, 4, 2],
              [1, 2, 1]]]
        ]).unsqueeze(1)


# =================================================
#            PLOT
# =================================================
def imCmp(I1, I2, method=None):
    from numpy import concatenate, zeros, ones, maximum, exp
    if len(I1.shape) in [4, 5]:
        shape_to_fill = I1.shape[2:] + (1,)
    else:
        raise ValueError(f"I1 should be [B,C,H,W] or [B,C,K,H,W] got {I1.shape}")
    if not isinstance(I1, np.ndarray):
        I1 = I1.detach().cpu().numpy()
    if not isinstance(I2, np.ndarray):
        I2 = I2.detach().cpu().numpy()
    I1 = I1[0, 0, ...]
    I2 = I2[0, 0, ...]

    if method is None:
        return concatenate((I2[..., None], I1[..., None], zeros(shape_to_fill)), axis=-1)
    elif 'seg' in method:
        u = I2[..., None] * I1[..., None]
        if 'w' in method:
            d = I1[..., None] - I2[..., None]
            # z = np.zeros(d.shape)
            # z[I1[..., None] + I2[...]] = 1
            # print(f'd min = {d.min()},{d.max()}')
            r = maximum(d, 0) / np.abs(d).max()
            g = u + .1 * exp(-d ** 2) * u + maximum(-d, 0) * .2
            b = maximum(-d, 0) / np.abs(d).max()
            # rr,gg,bb = r.copy(),g.copy(),b.copy()
            # rr[r + g + b == 0] =1
            # gg[r + g + b == 0] =1
            # bb[r + g + b == 0] =1
            rgb = concatenate(
                (r, g, b, ones(shape_to_fill)), axis=-1
            )
            rgb = np.clip(rgb, 0, 1)
            # print(f"r {r.min()};{r.max()}")
            return rgb
        if 'h' in method:
            d = I1[..., None] - I2[..., None]
            # z = np.ones(d.shape)
            # z[u == 0] =0
            r = maximum(d, 0) / np.abs(d).max()
            g = u * (1 + exp(-d ** 2))
            b = maximum(-d, 0) / np.abs(d).max()

            rgb = concatenate(
                (r, g, b, ones(shape_to_fill)), axis=-1
            )
            return rgb
        else:
            return concatenate(
                (
                    I1[..., None] - u,
                    u,
                    I2[..., None] - u,
                    ones(shape_to_fill)
                ), axis=-1
            )
    elif 'compose' in method:
        return concatenate(
            (
                I1[..., None],
                (I1[..., None] + I2[..., None]) / 2,
                I2[..., None],
                ones(shape_to_fill)
            ), axis=-1
        )


def temporal_img_cmp(img_1, img2, method="compose"):
    """
    Stack two gray-scales images to compare them. The images must have the same
    height and width and depth (for 3d). Images can be temporal meaning that
    they have a time dimension stored in the first dimension.

    Parameters
    ----------
    img_1 : torch.Tensor
        [T_1,C,H,W] or [T_1,C,D,H,W] tensor C = 1
    img2 : torch.Tensor
        [T_2,C,H,W] or [T_2,C,D,H,W] tensor C = 1
    .. note:

        T_1 = 1 and T_2 > 1 or T_1 > 1 and T_2 = 1 or T_1 = T_2 > 1 works, any other case will raise an error.

    method: str
        method to compare the images, among {'compose','seg','segw','segh'}

    """
    T1, C1, D1, H1, W1 = img_1.shape
    T2, C2, D2, H2, W2 = img2.shape
    if D1 != D2 or H1 != H2 or W1 != W2:
        raise ValueError(
            f"The images must have the same shape, got img_1 : {img_1.shape} and img_2 : {img2.shape}"
        )
    if C1 > 1 or C2 > 1:
        raise ValueError(
            f"The images must have only one channel, got img_1 : {img_1.shape} and img_2 : {img2.shape}"
        )

    if T1 == T2 and T1 == 1:
        return imCmp(img_1, img2, method=method)
    elif T2 > T1 and T1 == 1:
        buffer_img = img2
        img2 = img_1
        img_1 = buffer_img
        T1, T2 = T2, T1

    if T1 > T2 and T2 == 1:
        t_img = np.zeros((T1, D1, H1, W1, 4))
        for t, im1 in enumerate(img_1):
            t_img[t] = imCmp(im1[None], img2, method=method)
        return t_img

    elif T1 == T2 and T1 > 1:
        t_img = np.zeros((T1, D1, H1, W1, 4))
        for t, im1, im2 in enumerate(zip(img_1, img2)):
            t_img[t] = imCmp(im1[None], im2[None], method=method)
        return t_img
    else:
        raise ValueError(
            f"Supports only temporal images with the same number of time of one temporal image and one static image, got img_1 : {img_1.shape} and img_2 : {img2.shape}"
        )


def checkDiffeo(field):
    if len(field.shape) == 4:
        _, H, W, _ = field.shape
        det_jaco = detOfJacobian(field_2d_jacobian(field))[0]
        I = .4 * torch.ones((H, W, 3))
        I[:, :, 0] = (det_jaco <= 0) * 0.83
        I[:, :, 1] = (det_jaco >= 0)
        return I
    elif len(field.shape) == 5:
        field_im = grid2im(field)
        jaco = SpatialGradient3d()(field_im)
        # jaco = spacialGradient_3d(field,dx_convention='pixel')
        det_jaco = detOfJacobian(jaco)
        return det_jaco


@deco.deprecated("Please specify the dimension by using gridDef_plot_2d ot gridDef_plot_3d")
def gridDef_plot(defomation,
                 ax=None,
                 step=2,
                 add_grid=False,
                 check_diffeo=False,
                 title="",
                 color=None,
                 dx_convention='pixel'):
    return gridDef_plot_2d(defomation, ax=ax, step=step, add_grid=add_grid,
                           check_diffeo=check_diffeo, title=title,
                           color=color, dx_convention=dx_convention)


def gridDef_plot_2d(deformation: torch.Tensor,
                    ax=None,
                    step: int | tuple[int] = 2,
                    add_grid: bool = False,
                    check_diffeo: bool = False,
                    dx_convention: str = 'pixel',
                    title: str = "",
                    #  color : str | None = None,
                    # linewidth : int | float = None,
                    # origin : str = 'lower',
                    **kwargs
                    ):
    """ Plot the deformation field as a grid.

    :param deformation: torch.Tensor of shape (1,H,W,2)
    :param ax: matplotlib axis object, if None, the plot makes one new (default None)
    :param step: int | Tuple[int], step of the grid (default 2)
    :param add_grid: (bool), to use if  ̀defomation` is a field (default False).
    If True, add a regular grid to the field.
    :param check_diffeo: (bool), check if the deformation is a diffeomorphism (default False)
    :param dx_convention: (str) convention of the deformation field (default 'pixel')
    :param title: (str) title of the plot
    :param color: (str) color of the grid (default None = black)
    :param linewidth: (int) width of the grid lines (default None = 2)
    :param origin: (str) origin of the plot (default 'lower')

    :return: matplotlib axis object
    """
    # if not torch.is_tensor(deformation):
    #     raise TypeError("showDef has to be tensor object")
    if deformation.size().__len__() != 4 or deformation.size()[0] > 1:
        raise TypeError("deformation has to be a (1,H,W,2) "
                        "tensor object got " + str(deformation.size()))
    deform = deformation.clone().detach()
    if ax is None:
        fig, ax = plt.subplots()

    # Définir les valeurs par défaut pour les paramètres kwargs
    kwargs.setdefault('color', 'black')
    kwargs.setdefault('linewidth', 2)
    origin = kwargs.pop('origin', 'lower')

    if dx_convention == '2square':
        deform = square2_to_pixel_convention(
            deform,
            is_grid=True
        )
    elif dx_convention == 'square':
        deform = square_to_pixel_convention(
            deform,
            is_grid=True
        )

    if add_grid:
        reg_grid = make_regular_grid(deform.size(), dx_convention='pixel')
        deform += reg_grid

    if check_diffeo:
        cD = checkDiffeo(deform)

        title += 'diffeo = ' + str(cD[:, :, 0].sum() <= 0)

        ax.imshow(cD, interpolation='none', origin='lower')
        origin = 'lower'

    if isinstance(step, int | float):
        step_x, step_y = (step, step)
    else:
        step_x, step_y = step

    sign = 1 if origin == 'lower' else -1
    # kw = dict(color=color,linewidth=linewidth)
    l_a = ax.plot(deform[0, :, ::step_y, 0].numpy(),
                  sign * deform[0, :, ::step_y, 1].numpy(), **kwargs)
    l_b = ax.plot(deform[0, ::step_x, :, 0].numpy().T,
                  sign * deform[0, ::step_x, :, 1].numpy().T, **kwargs)

    # add the last lines on the right and bottom edges
    l_c = ax.plot(deform[0, :, -1, 0].numpy(),
                  sign * deform[0, :, -1, 1].numpy(), **kwargs)
    l_d = ax.plot(deform[0, -1, :, 0].numpy().T,
                  sign * deform[0, -1, :, 1].numpy().T, **kwargs)

    lines = l_a + l_b + l_c + l_d

    ax.set_aspect('equal')
    ax.set_title(title)

    return ax, lines


def quiver_plot(field,
                ax=None,
                step=2,
                title="",
                check_diffeo=False,
                color=None,
                dx_convention='pixel',
                real_scale=True,
                remove_grid=False
                ):
    """ Plot the deformation field as a quiver plot.

    Parameters
    ----------
    field : torch.Tensor
        2D tensor of shape (1,H,W,2)
    ax : matplotlib axis object, optional
        If None, the plot makes one new (default None)
    step : int, optional
        Step of the grid (default 2)
    title : str, optional
        Title of the plot
    check_diffeo : bool, optional
        Check if the deformation is a diffeomorphism (default False) by
        displaying the sign of the determinant of the Jacobian matrix at each point.
    color : str, optional
        Color of the grid (default None = black)
    dx_convention : str, optional
        Convention of the deformation field (default 'pixel')
    real_scale : bool, optional
        If True, plot quiver arrow with axis scale (default True)
    remove_grid : bool, optional
        If True, `field` is considered as a deformortion and the regular grid
        is removed  from `field` (default False)

    Returns
    -------
    ax : matplotlib
        Axis object
    """
    if not is_tensor(field):
        raise TypeError("field has to be tensor object")
    if field.size().__len__() != 4 or field.size()[0] > 1:
        raise TypeError("field has to be a (1",
                        "H,W,2) or (1,H,W,D,3) tensor object got ",
                        str(field.size()))

    if ax is None:
        fig, ax = plt.subplots()

    if color is None:
        color = 'black'

    reg_grid = make_regular_grid(field.size(), dx_convention='pixel')
    if dx_convention == '2square':
        field = square2_to_pixel_convention(field, is_grid=remove_grid)
    elif dx_convention == 'square':
        field = square_to_pixel_convention(field, is_grid=remove_grid)

    if remove_grid:
        field -= reg_grid

    if check_diffeo:
        cD = checkDiffeo(reg_grid + field)
        title += 'diffeo = ' + str(cD[:, :, 0].sum() <= 0)

        ax.imshow(cD, interpolation='none', origin='lower')

    # real scale =1 means to plot quiver arrow with axis scale
    (scale_units, scale) = ('xy', 1) if real_scale else (None, None)

    arrows = ax.quiver(reg_grid[0, ::step, ::step, 0], reg_grid[0, ::step, ::step, 1],
                       ((field[0, ::step, ::step, 0]).detach().numpy()),
                       ((field[0, ::step, ::step, 1]).detach().numpy()),
                       color=color,
                       scale_units=scale_units, scale=scale)
    return ax, arrows


def is_tensor(input):
    # print("is_tensor",input.__class__ == type(torch.Tensor),input.__class__,type(torch.Tensor))
    # print("is_tensor", "Tensor" in str(input.__class__), "Tensor" in str(type(torch.Tensor)))
    return "Tensor" in str(input.__class__)


def deformation_show(deformation, step=2,
                     check_diffeo=False, title="", color=None):
    r"""
    Make a plot showing the deformation.

    Parameters
    ----------

    deformation : torch.Tensor
        2D grid tensor of shape (1,H,W,2)
    step, int
        Step of the grid (default 2)
    check_diffeo, bool
        Check if the deformation is a diffeomorphism (default False) by showing the sign of the determinant of the Jacobian matrix at each point.
        green is for positive determinant and red for negative.
    title, str
        Title of the plot
    color, str
        Color of the grid (default None = black)

    Returns
    -------

    None

    Example :
    ----------

    .. code-block:: python

        cms = mbs.getCMS_allcombinaision()

        H,W = 100,150
        # vector defomation generation
        v = mbs.field2D_bspline(cms,(H,W),dim_stack=2).unsqueeze(0)
        v *= 0.5

        deform_diff = vff.FieldIntegrator(method='fast_exp')(v.clone(),forward= True)

        deformation_show(deform_diff,step=4,check_diffeo=True)

    """

    fig, axes = plt.subplots(1, 2)
    fig.suptitle(title)
    regular_grid = make_regular_grid(deformation.size())
    gridDef_plot_2d(deformation, step=step, ax=axes[0],
                    check_diffeo=check_diffeo,
                    color=color)
    quiver_plot(deformation - regular_grid, step=step,
                ax=axes[1], check_diffeo=check_diffeo,
                color=color)
    plt.show()


def vectField_show(field, step=2, check_diffeo=False, title="",
                   dx_convention='pixel'):
    r"""

    :param field: (1,H,W,2) tensor object
    :param step:
    :param check_diffeo: (bool)
    :return:

    Example :
    ----------
    >>>cms = mbs.getCMS_allcombinaision()
    >>>
    >>>H,W = 100,150
    >>># vector defomation generation
    >>>v = mbs.field2D_bspline(cms,(H,W),dim_stack=2).unsqueeze(0)
    >>>v *= 0.5
    >>>
    >>>vectField_show(v,step=4,check_diffeo=True)
    """

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
    fig.suptitle(title)
    regular_grid = make_regular_grid(field.size(), dx_convention=dx_convention)
    gridDef_plot(field + regular_grid, step=step, ax=axes[0],
                 check_diffeo=check_diffeo,
                 dx_convention=dx_convention)
    quiver_plot(field, step=step,
                ax=axes[1], check_diffeo=check_diffeo,
                dx_convention=dx_convention)
    plt.show()


def geodesic_3d_slider(mr):
    """ Display a 3d image

    exemple:
    mr = mt.load_optimize_geodesicShooting('2D_13_10_2021_m0t_m1_001.pk1')
    geodesic_3d_slider(mr)
    """
    image = mr.mp.image_stock.numpy()
    print(image.shape)
    residuals = mr.mp.momentum_stock.numpy()

    fig, ax = plt.subplots(1, 2)

    kw_image_args = dict(cmap='gray',
                         extent=[-1, 1, -1, 1],
                         origin='lower',
                         vmin=0, vmax=1)
    kw_residuals_args = dict(cmap='RdYlBu_r',
                             extent=[-1, 1, -1, 1],
                             origin='lower',
                             vmin=residuals.min(),
                             vmax=residuals.max())

    img_x = ax[0].imshow(image[0, 0], **kw_image_args)
    img_y = ax[1].imshow(residuals[0], **kw_residuals_args)
    # img_z = ax[2].imshow( image_slice(image,init_z_coord,dim=2),origin='lower', **kw_image)
    ax[0].set_xlabel('X')
    ax[1].set_xlabel('Y')
    # ax[2].set_xlabel('Z')

    ax[0].margins(x=0)

    # adjust the main plot to make room for the sliders
    plt.subplots_adjust(left=0.1, right=.9, bottom=0.25)

    # Make sliders.
    axcolor = 'lightgoldenrodyellow'
    # place them [x_bottom,y_bottom,height,width]
    sl_t = plt.axes([0.25, 0.15, 0.5, 0.03], facecolor=axcolor)

    kw_slider_args = dict(
        valmin=0,
        valfmt='%0.0f'
    )
    t_slider = Slider(label='t', ax=sl_t,
                      valmax=image.shape[0] - 1, valinit=0,
                      **kw_slider_args)

    # The function to be called anytime a slider's value changes
    def update(val):
        img_x.set_data(image[int(t_slider.val), 0])
        img_y.set_data(residuals[int(t_slider.val)])
        # img_z.set_data(image_slice(image, z_slider.val, 2))

        fig.canvas.draw_idle()

    # register the update function with each slider
    t_slider.on_changed(update)

    # it is important to store the sliders in order to
    # have them updating
    return t_slider


# =================================================
#            FIELD RELATED FUNCTIONS
# =================================================

def fieldNorm2(field):
    return (field ** 2).sum(dim=-1)


@deco.deprecated("function deprecated. DO NOT USE, see vector_field_to_flow")
def field2diffeo(in_vectField, N=None, save=False, forward=True):
    """function deprecated; see vector_field_to_flow"""
    return vff.FieldIntegrator(method='fast_exp')(in_vectField.clone(), forward=forward)


def imgDeform(img, deform_grid, dx_convention='2square', clamp=False):
    """
    Apply a deformation grid to an image

    Parameters
    ----------
    img : torch.Tensor of shape [B,C,H,W] or [B,C,D,H,W]
        image to deform
    deform_grid : torch.Tensor of shape [B,H,W,2] or [B,D,H,W,3]
        deformation grid
    dx_convention : str, optional
        convention of the deformation grid (default '2square')
    clamp : bool, optional
        if True, clamp the image between 0 and 1 if the max value is less than 1 else between 0 and 255 (default False)

    Returns
    -------
    torch.Tensor
        deformed image of shape [B,C,H,W] or [B,C,D,H,W]
    """

    if img.shape[0] > 1 and deform_grid.shape[0] == 1:
        deform_grid = torch.cat(img.shape[0] * [deform_grid], dim=0)
    if dx_convention == 'pixel':
        deform_grid = pixel_to_2square_convention(deform_grid)
    elif dx_convention == 'square':
        deform_grid = square_to_2square_convention(deform_grid)
    deformed = F.grid_sample(img.to(deform_grid.dtype),
                             deform_grid,
                             **DLT_KW_GRIDSAMPLE
                             )
    # if len(I.shape) == 5:
    #     deformed = deformed.permute(0,1,4,3,2)
    if clamp:
        max_val = 1 if img.max() <= 1 else 255
        # print(f"I am clamping max_val = {max_val}, I.max,min = {I.max(),I.min()},")
        deformed = torch.clamp(deformed, min=0, max=max_val)
    return deformed


def compose_fields(field, grid_on, dx_convention='2square'):
    """ compose a field on a deformed grid

    """
    if field.device != grid_on.device:
        raise RuntimeError("Expexted all tensors to be on same device but got"
                           f"field on {field.device} and grid on {grid_on.device}")
    if dx_convention == 'pixel':
        field = pixel_to_2square_convention(field, is_grid=False)
        grid_on = pixel_to_2square_convention(grid_on, is_grid=True)
    elif dx_convention == 'square':
        field = square_to_2square_convention(field, is_grid=False)
        grid_on = square_to_2square_convention(grid_on, is_grid=True)

    composition = im2grid(
        F.grid_sample(
            grid2im(field), grid_on,
            **DLT_KW_GRIDSAMPLE
        )
    )
    if dx_convention == 'pixel':
        return square2_to_pixel_convention(composition, is_grid=False)
    elif dx_convention == 'square':
        return square2_to_square_convention(composition, is_grid=False)
    else:
        return composition


def vect_spline_diffeo(control_matrix, field_size, N=None, forward=True):
    field = mbs.field2D_bspline(control_matrix, field_size, dim_stack=2)[None]
    return vff.FieldIntegrator(method='fast_exp')(field.clone(), forward=forward)


class RandomGaussianImage:
    r"""
    Generate a random image made from a sum of N gaussians
    and compute the derivative of the image with respect
     to the parameters of the gaussians.

     self.a[i] * torch.exp(- ((self.X - self.c[i])**2).sum(-1) / (2*self.b[i]**2))

     .. math::
         I = \sum_{i=1}^{N} a_i \exp(- \frac{||X - c_i||^2}{2b_i^2})

    Parameters:
    -----------
    size: tuple
        tuple with the image dimensions to create
    n_gaussians: int
        Number of gaussians to sum.
    dx_convention: str
        convention for the grid
    a: list of float
        list of the a parameters of the gaussians controling the amplitude
    b: list of float
        list of the b parameters of the gaussians controling the width
    c: list of float
        list of the c parameters of the gaussians controling the position

    Example:
    --------

    .. code-block:: python
        RGI = RandomGaussianImage((100,100),5,'pixel')
         image = RGI.image()
         derivative = RGI.derivative()

    """

    def __init__(self, size, n_gaussians, dx_convention, a=None, b=None, c=None):
        if a is None:
            self.a = 2 * torch.rand((n_gaussians,)) - 1
        else:
            if len(a) != n_gaussians:
                raise ValueError(f"len(a) = {len(a)} should be equal to n_gaussians = {n_gaussians}")
            self.a = torch.tensor(a)
        if b is None:
            self.b = torch.randint(1, int(min(size) / 5), (n_gaussians,))
        else:
            if len(b) != n_gaussians:
                raise ValueError(f"len(b) = {len(b)} should be equal to n_gaussians = {n_gaussians}")
            self.b = torch.tensor(b)

        self.N = n_gaussians
        self.size = size

        self.X = make_regular_grid(size, dx_convention=dx_convention)
        if dx_convention == 'pixel':
            # self.X = make_meshgrid(
            #     [torch.arange(0,s) for s in size]
            # )
            if c is None:
                self.c = torch.stack(
                    [torch.randint(0, s - 1, (n_gaussians,)) for s in size],
                    dim=1
                )
            else:
                self.c = torch.tensor(c)
            # self.c = torch.randn((n_gaussians, 2))
            # self.c[:,0] = (self.c[:,0] + 1) * size[0] / 2
            # self.c[:,1] = (self.c[:,1] + 1) * size[1] / 2


        elif dx_convention == '2square':
            # self.X = make_meshgrid(
            #     [torch.linspace(-1,1,s) for s in size],
            # )
            if c is None:
                self.c = 2 * torch.rand((n_gaussians, len(size))) - 1
            else:
                self.c = torch.tensor(c)
            if b is None:
                self.b = self.b.to(torch.float)
                bmax = self.b.max()
                self.b *= 1 / bmax
                bmax = self.b.max()
                self.b = bmax * (self.b / bmax) ** 2

        elif dx_convention == 'square':
            # self.X = make_meshgrid(
            #     [torch.linspace(0, 1, s) for s in size],
            # )
            if c is None:
                self.c = torch.rand((n_gaussians, len(size)))
            else:
                self.c = torch.tensor(c)
            if b is None:
                self.b = self.b.to(torch.float)
                bmax = self.b.max()
                self.b *= 1 / (2 * bmax)
                bmax = self.b.max()
                self.b = bmax * (self.b / bmax) ** 2
        else:
            raise NotImplementedError("Et oui")

    def gaussian(self, i):
        """
        return the gaussian with the parameters a,b,c at pos i
        :param i: (int) position of the gaussian
        """
        # print(self.X - self.c[i])
        return self.a[i] * torch.exp(- ((self.X - self.c[i]) ** 2).sum(-1) / (2 * self.b[i] ** 2))

    def image(self):
        """
        return the image made from the sum of the gaussians
        : return: torch.Tensor of shape [1,1,H,W] or [1,1,D,H,W]
        """
        image = torch.zeros((1,) + self.size)
        for i in range(self.N):
            image += self.gaussian(i)

        return image[None]

    def derivative(self):
        """
        Compute the derivative of the image with respect to the position of the gaussians
        : return: torch.Tensor of shape [1,2,H,W] of [1,3,D,H,W]
        """
        derivative = torch.zeros_like(self.X)
        for i in range(self.N):
            derivative += - 1 / self.b[i] ** 2 * (self.X - self.c[i]) * self.gaussian(i)[..., None]
        return grid2im(derivative)


class RandomGaussianField:
    r"""
    Generate a random field made from a sum of N gaussians
    and compute the theoretical divergence of the field. It is usefully
    for testing function on random generated fields with known expressions,
    making it possible to compute theoretical values. It uses the function RandomGaussianImage.

    If $v : \Omega \mapsto \mathbb{R}^d$ for all $n < d$

    .. math::
        v_n = \sum_{i=1}^{N} a_i \exp(- \frac{||X - c_i||^2}{2b_i^2})

    Parameters:
    -----------
    size: tuple
        tuple with the image dimensions to create (H,W) will create a 2d field  of shape (H,W,2),
         ((D,H,W,3) in 3d)
    n_gaussian: int
        Number of gaussians to sum.
    a: list of float
        list of the a parameters of the gaussians controlling the amplitude
    b: list of float
        list of the b parameters of the gaussians controlling the width
    c: list of float
        list of the c parameters of the gaussians controlling the position

    """

    def __init__(self, size, n_gaussian, dx_convention, a=None, b=None, c=None):
        a = [None] * size[-1] if a is None else a
        b = [None] * size[-1] if b is None else b
        c = [None] * size[-1] if c is None else c
        self.rgi_list = [
            RandomGaussianImage(size[:-1], n_gaussian, dx_convention, a=a[i], b=b[i], c=c[i])
            for i in range(size[-1])
        ]

    def field(self):
        """
        return the field made from the sum of the gaussians
        : return: torch.Tensor of shape [1,H,W,2] or [1,D,H,W,3]
        """
        field = torch.stack(
            [rgi.image()[0, 0] for rgi in self.rgi_list],
            dim=-1
        )
        return field[None]

    def divergence(self):
        divergence = torch.zeros(self.rgi_list[0].size)
        for i, rgi in enumerate(self.rgi_list):
            divergence += rgi.derivative()[0, i]
        return divergence[None, None]


def field_2d_jacobian(field):
    r"""
    compute the jacobian of the field

    parameters:
    -----------
    field: field.size (b,h,w,2)

    returns:
    --------
    jacobian of the field.size = (b,2,2,h,w)

    :example:

    .. code-block:: python

        field = torch.zeros((100,100,2))
        field[::2,:,0] = 1
        field[:,::2,1] = 1

        jaco =  field_2d_jacobian(field)


        plt.rc('text',usetex=true)
        fig, axes = plt.subplots(2,2)
        axes[0,0].imshow(jaco[0,0,0,:,:].detach().numpy(),cmap='gray')
        axes[0,0].set_title(r"$\frac{\partial f_1}{\partial x}$")
        axes[0,1].imshow(jaco[0,0,1,:,:].detach().numpy(),cmap='gray')
        axes[0,1].set_title(r"$\frac{\partial f_1}{\partial y}$")
        axes[1,0].imshow(jaco[0,1,0,:,:].detach().numpy(),cmap='gray')
        axes[1,0].set_title(r"$\frac{\partial f_2}{\partial x}$")
        axes[1,1].imshow(jaco[0,1,1,:,:].detach().numpy(),cmap='gray')
        axes[1,1].set_title(r"$\frac{\partial f_2}{\partial y}$")

        plt.show()
    """

    f_d = grid2im(field)
    return SpatialGradient()(f_d)


def field_2d_hessian(field_grad):
    r""" compute the hessian of a field from the jacobian

    :param field_grad: BxnxpxHxW tensor n = p = 2
    :return: Bx8x2xHxW tensor

    :example :

    hess = field_2d_hessian(I_g)
    print('hess.shape = '+str(hess.shape))
    fig, axes = plt.subplots(2,4)
    for x in range(2):
        for d in range(4):
            axes[x][d].imshow(hess[0,d,x,:,:].detach().numpy(),cmap='gray')
            axes[x][d].set_title(str((x,d)))
    plt.show()

    """
    N, _, _, H, W = field_grad.shape
    device = 'cuda' if field_grad.is_cuda else 'cpu'
    hess = torch.zeros((N, 4, 2, H, W), device=device)

    hess[:, :2, :, :, :] = SpatialGradient()(field_grad[:, 0, :, :, :])
    hess[:, 2:, :, :, :] = SpatialGradient()(field_grad[:, 1, :, :, :])

    return hess


# %%
def detOfJacobian(jaco):
    """ compute the determinant of the jacobian from field_2d_jacobian

    :param jaco: B,2,2,H,W tensor
                B,3,3,D,H,W tensor
    :return: B,H,W tensor
    """
    if jaco.shape[1] == 2:
        return jaco[:, 0, 0, :, :] * jaco[:, 1, 1, :, :] - jaco[:, 1, 0, :, :] * jaco[:, 0, 1, :, :]
    elif jaco.shape[1] == 3:
        dxdu = jaco[:, 0, 0]
        dxdv = jaco[:, 0, 1]
        dxdw = jaco[:, 0, 2]
        dydu = - jaco[:, 1, 0]  # The '-' are here to answer
        dydv = - jaco[:, 1, 1]  # to the image convention
        dydw = - jaco[:, 1, 2]  # Be careful using it.
        dzdu = jaco[:, 2, 0]
        dzdv = jaco[:, 2, 1]
        dzdw = jaco[:, 2, 2]
        a = dxdu * (dydv * dzdw - dydw * dzdv)
        b = - dxdv * (dydu * dzdw - dydw * dzdu)
        c = dxdw * (dydu * dzdv - dydv * dzdu)
        return a + b + c


# %%

class Field_divergence(torch.nn.Module):

    def __init__(self, dx_convention='pixel'):
        self.dx_convention = dx_convention

        super(Field_divergence, self).__init__()

    def __repr__(self):
        return (
                self.__class__.__name__
                + '(field dimension ='
                + self.field_dim + 'd, '
                + 'dx_convention ='
                + self.dx_convention
                + ')'
        )

    def forward(self, field):
        """
        Note: we don't use the sobel implementation in SpatialGradient to save computation
        """
        field_as_im = grid2im(field)
        if field.shape[-1] == 2:
            x_sobel = get_sobel_kernel_2d().to(field.device) / 8
            _, H, W, _ = field.shape
            field_x_dx = filter2d(field_as_im[:, 0, :, :].unsqueeze(1),
                                  x_sobel.unsqueeze(0))  # * (2/(H-1)))
            field_y_dy = filter2d(field_as_im[:, 1, :, :].unsqueeze(1),
                                  x_sobel.T.unsqueeze(0))  # * (2/(W-1)))

            field_div = torch.stack([field_x_dx, field_y_dy], dim=0)

        elif field.shape[-1] == 3:
            x_sobel = get_sobel_kernel_3d().to(field.device)
            _, D, H, W, _ = field.shape
            field_x_dx = filter3d(field_as_im[:, 0].unsqueeze(1),
                                  x_sobel[0] / x_sobel[0].abs().sum())
            field_y_dy = filter3d(field_as_im[:, 1].unsqueeze(1),
                                  x_sobel[1] / x_sobel[
                                      1].abs().sum())
            field_z_dz = filter3d(field_as_im[:, 2].unsqueeze(1),
                                  x_sobel[2] / x_sobel[2].abs().sum())
            field_div = torch.stack([field_x_dx, field_y_dy, field_z_dz], dim=0)

        if self.dx_convention == 'square':
            return torch.stack(
                [(s - 1) * field_div[i] for i, s in enumerate(field_as_im.shape[2:][::-1])],
                dim=0).sum(dim=0)

        if self.dx_convention == '2square':
            return torch.stack(
                [(s - 1) / 2 * field_div[i] for i, s in enumerate(field_as_im.shape[2:][::-1])],
                dim=0).sum(dim=0)
        else:
            return field_div.sum(dim=0)


@deco.deprecated
def field_divergence(field, dx_convention='pixel'):
    r"""
    make the divergence of a field, for each pixel $p$ in I
    $$div(I(p)) = \sum_{i=1}^C \frac{\partial I(p)_i}{\partial x_i}$$

    Parameters:
    -----------
    field: torch.Tensor
        of shape (B,H,W,2) or (B,D,H,W,3)

    Returns:
    --------
    div: torch.Tensor
        of shape (B,2,2H,W) or (B,3,3,D,H,W)

    Example:
    ---------

    .. code-block:: python

        cms = torch.tensor([  # control matrices
        [[0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, +1, 0, -1, 0, -1, 0, -1, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, +1, 0, -1, 0, +1, 0, +1, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, +1, 0, +1, 0, -1, 0, +1, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, -1, 0, -1, 0, -1, 0, +1, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0],
         ],
        [[0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, +1, 0, +1, 0, -1, 0, +1, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0],#[0, .2, .75, 1, 0],
         [0, -1, 0, -1, 0, -1, 0, +1, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, +1, 0, -1, 0, -1, 0, -1, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, +1, 0, -1, 0, +1, 0, +1, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0]]
        ],requires_grad=False,dtype=torch.float)

        field_size = (20,20)
        field = mbs.field2D_bspline(cms,field_size,
                                    degree=(3,3),dim_stack=2).unsqueeze(0)

        # field_diff = vect_spline_diffeo(cms,field_size)
        H,W = field_size
        xx, yy = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W))

        div = field_2d_divergence(field)

        # _,d_ax = plt.subplots()
        fig,ax = plt.subplots()

        div_plot = ax.imshow(div[0,0,:,:],origin='lower')
        ax.quiver(field[0,:,:,0],field[0,:,:,1])
        fig.colorbar(div_plot)
        plt.show()
    """
    return Field_divergence(dx_convention)(field)


def pixel_to_2square_convention(field, is_grid=True):
    """
    Convert a field in spatial pixelic convention in one on as
    [-1,1]^2 square as requested by pytorch's gridSample

    Parameters
    ----------
    field : torch.Tensor
        of size [T,H,W,2] or [T,D,H,W,3]
    is_grid : bool
        if True field is considered as a deformation (i.e.: field = (id + v))
        else field is a vector field (i.e.: field = v)
        (default is True)

    Returns
    -------
    field : torch.Tensor
        of size [T,H,W,2] or [T,D,H,W,3]
    """

    field = field.clone()
    if field.shape[-1] == 2:
        _, H, W, _ = field.shape
        mult = torch.tensor((2 / (W - 1), 2 / (H - 1))).to(field.device)
        if not torch.is_tensor(field):
            mult = mult.numpy()
        sub = 1 if is_grid else 0
        return field * mult[None, None, None] - sub
    elif field.shape[-1] == 3 and len(field.shape) == 5:
        _, D, H, W, _ = field.shape
        mult = torch.tensor((2 / (W - 1), 2 / (H - 1), 2 / (D - 1))).to(field.device)
        # if not torch.is_tensor(field): # does not works anymore for a reason
        if not is_tensor(field):
            mult = mult.numpy()
        sub = 1 if is_grid else 0
        return field * mult[None, None, None, None] - sub
    else:
        raise NotImplementedError("Indeed")


def square2_to_pixel_convention(field, is_grid=True):
    """ Convert a field on a square centred and from -1 to 1 convention
    as requested by pytorch's gridSample to one in pixelic convention

    Parameters
    ----------
    field : torch.Tensor
        of size [T,H,W,2] or [T,D,H,W,3]
    is_grid : bool
        if True field is considered as a deformation (i.e.: field = (id + v))
        else field is a vector field (i.e.: field = v)
        (default is True)

    Returns
    -------
    field : torch.Tensor
        of size [T,H,W,2] or [T,D,H,W,3]

    """
    field = field.clone()
    if field.shape[-1] == 2:
        _, H, W, _ = field.shape
        mult = torch.tensor(((W - 1) / 2, (H - 1) / 2)).to(field.device)
        if not is_tensor(field):
            mult = mult.numpy()
        add = 1 if is_grid else 0
        return (field + add) * mult[None, None, None]
    elif field.shape[-1] == 3 and len(field.shape) == 5:
        _, D, H, W, _ = field.shape
        mult = torch.tensor(((W - 1) / 2, (H - 1) / 2, (D - 1) / 2)).to(field.device)
        if not is_tensor(field):
            mult = mult.numpy()
        add = 1 if is_grid else 0
        return (field + add) * mult[None, None, None, None]
    else:
        raise NotImplementedError("Indeed")


def square_to_pixel_convention(field, is_grid=True):
    r""" convert from the square convention to the pixel one,
    meaning: $[-1,1]^d \mapsto [0,W-1]\times[0,H-1]$

    Parameters
    ----------
    field : torch.Tensor
    of size [T,H,W,2] or [T,D,H,W,3]
    is_grid : bool
    useless in this function, kept for consistency with others converters

    Returns
    -------
    field : torch.Tensor
    of size [T,H,W,2] or [T,D,H,W,3]

    """

    for i, s in enumerate(tuple(field.shape[1:-1])[::-1]):
        field[..., i] *= (s - 1)
    return field


def pixel_to_square_convention(field, is_grid=True):
    r""" convert from the pixel convention to the square one,
    meaning: $[0,W-1]\times[0,H-1] \mapsto [-1,1]^d$

    Parameters
    ----------
    field : torch.Tensor
    of size [T,H,W,2] or [T,D,H,W,3]
    is_grid : bool
    useless in this function, kept for consistency with others converters

    Returns
    -------
    field : torch.Tensor
    of size [T,H,W,2] or [T,D,H,W,3]
    """

    for i, s in enumerate(tuple(field.shape[1:-1])[::-1]):
        field[..., i] /= (s - 1)
    return field


def square_to_2square_convention(field, is_grid=True):
    r""" convert from the square convention to the 2square one,
    meaning: $[0,1]^d \mapsto [-1,1]^d$

    Parameters
    ----------
    field : torch.Tensor
        of size [T,H,W,2] or [T,D,H,W,3]
    is_grid : bool
         Must be True if field is a grid+vector field and False if a vector field only

    Returns
    -------
    field : torch.Tensor
    of size [T,H,W,2] or [T,D,H,W,3]
    """

    sub = 1 if is_grid else 0
    return 2 * field - sub


def square2_to_square_convention(field, is_grid=True):
    r""" convert from the 2square convention to the square one,
    meaning: $[-1,1]^d \mapsto [0,1]^d$

    Parameters
    ----------
    field : torch.Tensor
        of size [T,H,W,2] or [T,D,H,W,3]
    is_grid : bool
         Must be True if field is a grid+vector field and False if a vector field only

    Returns
    -------
    field : torch.Tensor
    of size [T,H,W,2] or [T,D,H,W,3]
    """

    add = 1 if is_grid else 0
    return (field + add) / 2


def grid2im(grid):
    """
    Reshape a grid tensor into an image tensor

    -2D  [T,H,W,2] -> [T,2,H,W]
    -3D  [T,D,H,W,3] -> [T,3,D,H,W]

    .. code-block:: python

        # grid to image
        T,D,H,W = (4,5,6,7)

        grid_2D = torch.rand((T,H,W,2))
        grid_3D = torch.rand((T,D,H,W,3))

        image_2D = torch.rand((T,2,H,W))
        image_3D = torch.rand((T,3,D,H,W))

        grid_2D_as_image = grid2im(grid_2D)
        grid_3D_as_image = grid2im(grid_3D)

        # check if the method works
        print('\n  GRID TO IMAGE')
        print(' ==== 2D ====\n')
        print('grid_2D.shape =',grid_2D.shape)
        print('grid_2D_as_image.shape =',grid_2D_as_image.shape)
        print('we have indeed the good shape')
        count = 0
        for i in range(T):
            count += (grid_2D[i,...,0] == grid_2D_as_image[i,0,...]).sum()
            count += (grid_2D[i,...,1] == grid_2D_as_image[i,1,...]).sum()

        print('count is equal to ',count/(T*H*W*2),'and should be equal to 1')

        print(' \n==== 3D ====\n')
        print('grid_3D.shape =',grid_3D.shape)
        print('grid_3D_as_image.shape =',grid_3D_as_image.shape)
        print('we have indeed the good shape')
        count = 0
        for i in range(T):
            count += (grid_3D[i,...,0] == grid_3D_as_image[i,0,...]).sum()
            count += (grid_3D[i,...,1] == grid_3D_as_image[i,1,...]).sum()
            count += (grid_3D[i,...,2] == grid_3D_as_image[i,2,...]).sum()


        print('count is equal to ',count/(T*H*W*D*3),'and should be equal to 1')

    """

    if grid.shape[-1] == 2:  # 2D case
        return grid.transpose(2, 3).transpose(1, 2)

    elif grid.shape[-1] == 3:  # 3D case
        return grid.transpose(3, 4).transpose(2, 3).transpose(1, 2)
    else:
        raise ValueError("input argument expected is [N,H,W,2] or [N,D,H,W,3]",
                         "got " + str(grid.shape) + " instead.")


def im2grid(image):
    """
    Reshape an image tensor into a grid tensor

    - 2D case [T,2,H,W]   ->  [T,H,W,2]
    - 3D case [T,3,D,H,W] ->  [T,D,H,W,3]

    .. code-block:: python

        T,D,H,W = (4,5,6,7)

        grid_2D = torch.rand((T,H,W,2))
        grid_3D = torch.rand((T,D,H,W,3))

        image_2D = torch.rand((T,2,H,W))
        image_3D = torch.rand((T,3,D,H,W))

        # image to grid
        image_2D_as_grid = im2grid(image_2D)
        image_3D_as_grid = im2grid(image_3D)

        print('\n  IMAGE TO GRID')
        print(' ==== 2D ====\n')
        print('image_2D.shape = ',image_2D.shape)
        print('image_2D_as_grid.shape = ',image_2D_as_grid.shape)

        count = 0
        for i in range(T):
            count += (image_2D[i,0,...] == image_2D_as_grid[i,...,0]).sum()
            count += (image_2D[i,1,...] == image_2D_as_grid[i,...,1]).sum()
        print('count is equal to ',count/(T*H*W*2),'and should be equal to 1')

        print(' ==== 3D ====\n')
        print('image_3D.shape = ',image_3D.shape)
        print('image_3D_as_grid.shape = ',image_3D_as_grid.shape)

        count = 0
        for i in range(T):
            count += (image_3D[i,0,...] == image_3D_as_grid[i,...,0]).sum()
            count += (image_3D[i,1,...] == image_3D_as_grid[i,...,1]).sum()
            count += (image_3D[i,2,...] == image_3D_as_grid[i,...,2]).sum()
        print('count is equal to ',count/(T*H*W*D*3.0),'and should be equal to 1')

    """

    # No batch
    if image.shape[1] == 2:
        return image.transpose(1, 2).transpose(2, 3)
    elif image.shape[1] == 3:
        return image.transpose(1, 2).transpose(2, 3).transpose(3, 4)
    else:
        raise ValueError("input argument expected is [B,2,H,W] or [B,3,D,H,W]",
                         "got " + str(image.shape) + " instead.")


def format_sigmas(sigmas, dim):
    if type(sigmas) == int:
        return (sigmas,) * dim
    elif type(sigmas) == tuple:
        return sigmas
    elif type(sigmas) == list:
        return [(s,) * dim for s in sigmas]


def make_regular_grid(deformation_shape,
                      dx_convention='pixel',
                      device=torch.device('cpu'),
                      ):
    """
    API to create_meshgrid, it is the identity deformation

    Parameters
    --------------
    deformation_shape: tuple
        tuple such as (H,W) or (n,H,W,2) for 2D grid
        (D,H,W) or (n,D,H,W,3) for 3D gridim2
    device: torch.device
        device for selecting cpu or cuda usage

    Returns
    ---------
    grid: torch.Tensor
        2D identity deformation with size (1,H,W,2) or
        3D identity deformation with size (1,D,H,W,3)

    """

    def make_meshgrid(tensor_list):
        mesh = tuple(
            list(
                torch.meshgrid(tensor_list, indexing='ij')
            )[::-1]  # reverse the order of the list
        )
        return torch.stack(mesh, dim=-1)[None].to(device)

    if len(deformation_shape) == 4 or len(deformation_shape) == 5:
        deformation_shape = deformation_shape[1:-1]

    if dx_convention == 'pixel':
        return make_meshgrid(
            [torch.arange(0, s, dtype=torch.float) for s in deformation_shape],
        )

    elif dx_convention == '2square':
        return make_meshgrid(
            [torch.linspace(-1, 1, s) for s in deformation_shape],
        )

    elif dx_convention == 'square':
        return make_meshgrid(
            [torch.linspace(0, 1, s) for s in deformation_shape],
        )

    else:
        raise ValueError(f"make_regular_grid : dx_convention must be among"
                         f" ['pixel','2square','square']"
                         f"got {dx_convention}")


# =================================================================
#             LIE ALGEBRA
# =================================================================

def leviCivita_2Dderivative(v, w):
    r"""
    Perform the opetation $\nabla_v w$ in 2D
    """

    d_v = field_2d_jacobian(v)
    d_v_x = w[:, :, 0] * d_v[0, 0, 0, :, :] + w[:, :, 1] * d_v[0, 0, 1, :, :]
    d_v_y = w[:, :, 0] * d_v[0, 1, 0, :, :] + w[:, :, 1] * d_v[0, 1, 1, :, :]

    return torch.stack((d_v_x, d_v_y), dim=2)


def lieBracket(v, w):
    return leviCivita_2Dderivative(v, w) - leviCivita_2Dderivative(w, v)


def BCH(v, w, order=0):
    """ Evaluate the Backer-Campbell-Hausdorff formula"""
    var = v + w
    if order >= 1:
        # print(lieBracket(v,w))
        var += 0.5 * lieBracket(v, w)
    if order == 2:
        var += (lieBracket(v, lieBracket(v, w)) - lieBracket(w, lieBracket(v, w))) / 12
    if order > 2:
        print('non')
    return var


# =================================================================
#             GEOMETRIC HANDELER
# =================================================================

def find_binary_center(bool_img):
    if torch.is_tensor(bool_img):
        indexes = bool_img.nonzero(as_tuple=False)
    else:
        indexes = bool_img.nonzero()
    # Puis pour trouver le centre on cherche le min et max dans chaque dimension.
    # La ligne d'avant ordonne naturellement les indexes.
    min_index_1, max_index_1 = (indexes[0, 0], indexes[-1, 0])
    # print(min_index_1,max_index_1)
    # Ici il y a plus de travail.
    min_index_2, max_index_2 = (torch.argmin(indexes[:, 1]), torch.argmax(indexes[:, 1]))
    min_index_2, max_index_2 = (indexes[min_index_2, 1], indexes[max_index_2, 1])
    if len(bool_img.shape) in [2, 4]:
        centre = (
            torch.div(max_index_2 + min_index_2, 2, rounding_mode='floor'),
            torch.div(max_index_1 + min_index_1, 2, rounding_mode='floor')
        )

    else:
        min_index_3, max_index_3 = (torch.argmin(indexes[:, 2]), torch.argmax(indexes[:, 2]))
        min_index_3, max_index_3 = (indexes[min_index_3, 2], indexes[max_index_3, 2])
        centre = (
            torch.div(max_index_3 + min_index_3, 2, rounding_mode='floor'),
            torch.div(max_index_2 + min_index_2, 2, rounding_mode='floor'),
            torch.div(max_index_1 + min_index_1, 2, rounding_mode='floor'),
        )
    return centre


def make_ball_at_shape_center(img,
                              shape_binarization=None,
                              overlap_threshold=0.1,
                              r_min=None,
                              force_r=None,
                              force_center=None,
                              verbose=False):
    """
    Create a ball centered at the center of the shape in the image. The shape
    is defined by the binarisation of the image for the pixels having the value shape_value.

    Parameters
    ----------
    img : torch.Tensor
        [B,C,H,W] or [B,C,D,H,W] or [H,W] or [D,H,W] The image where
        the ball will be created.
    shape_binarization : torch.Tensor, optional
        a tensor of the same shape as img, being the binarization of the shape
        to create the ball. If None, the shape is defined by the pixels having the
         max value in the image.
    overlap_threshold : float, optional
        The percentage of overlap between the shape and the ball. The default is 0.1.
    r_min : int, optional
        The minimum radius of the ball. The default is None.
    force_r : int, optional
        The radius of the ball. The default is None.
    force_center : tuple, optional
        The center of the ball. The default is None.
    verbose : bool, optional
        Print the center, the radius and the overlap between the shape and the ball.

    Returns
    -------
    ball : torch.Tensor
        The ball as a bool mask of the same shape as img.
    centre : tuple
        The center of the ball and the radius. if 2d (c1,c2,r) if 3d (c1,c2,c3,r)
    """

    if len(img.shape) in [3, 5]:
        is_2D = False
    elif len(img.shape) in [2, 4]:
        is_2D = True
    if len(img.shape) in [4, 5]: img = img[0, 0]

    img = img.cpu()
    if force_center is None:
        if shape_binarization is None:
            max_val = img.max() if shape_binarization is None else shape_binarization
            # find all the indexes having the value shape_value
            # print('indexes :',indexes.shape)
            shape_binarization = (img == max_val)
        centre = find_binary_center(shape_binarization.to(torch.bool))
    else:
        centre = force_center

    if is_2D:
        Y, X = torch.meshgrid(torch.arange(img.shape[0]),
                              torch.arange(img.shape[1]))
    else:
        Z, Y, X = torch.meshgrid(torch.arange(img.shape[0]),
                                 torch.arange(img.shape[1]),
                                 torch.arange(img.shape[2])
                                 )

    def overlap_percentage():
        img_supp = img > 0
        # overlap = torch.logical_and((img_supp).cpu(), bool_ball).sum()
        # seg_sum = (img_supp).sum()
        # return overlap/seg_sum
        prod_seg = img_supp * bool_ball
        sum_seg = img_supp + bool_ball

        return float(2 * prod_seg.sum() / sum_seg.sum())

    if force_r is None:
        r = 3 if r_min is None else r_min
        # sum_threshold = 20
        bool_ball = torch.zeros(img.size(), dtype=torch.bool)
        count = 0
        while overlap_percentage() < overlap_threshold and count < 10:
            r += max(img.shape) // 50

            if is_2D:
                bool_ball = ((X - centre[0]) ** 2 + (Y - centre[1]) ** 2) < r ** 2
            else:
                bool_ball = ((X - centre[0]) ** 2 + (Y - centre[1]) ** 2 + (Z - centre[2]) ** 2) < r ** 2
            ball = bool_ball[None, None].to(img.dtype)
            # i3v.compare_3D_images_vedo(ball,img[None,None])
            count += 1
    else:
        r = force_r
        if is_2D:
            bool_ball = ((X - centre[0]) ** 2 + (Y - centre[1]) ** 2) < r ** 2
        else:
            bool_ball = ((X - centre[0]) ** 2 + (Y - centre[1]) ** 2 + (Z - centre[2]) ** 2) < r ** 2
        ball = bool_ball[None, None].to(img.dtype)

    if verbose:
        print(
            f"centre = {centre}, r = {r} and the seg and ball have {torch.logical_and((img > 0).cpu(), bool_ball).sum()} pixels overlapping")
    return ball, centre + (r,)
