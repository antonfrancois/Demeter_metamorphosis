"""
This module contains the classes used to compute the data attachment term
in the metamorphosis optimization. All data attachment terms must herit from
the abstract class `DataCost`. The module contains the following classes:
`Ssd`, `Ssd_normalized`, `Cfm`, `SimiliSegs`, `Mutlimodal_ssd_cfm`, `Longitudinal_DataCost`.
"""
from mailbox import Error

import torch
from abc import ABC, abstractmethod
from ..utils import torchbox as tb
from ..utils import cost_functions as cf
from math import prod, exp


class DataCost(ABC, torch.nn.Module):
    """
    Abstract class for the data attachment term in the metamorphosis optimization.
    The class `Optimize_geodesicShooting` requires a data attachment term to be provided
    as a subclass of `DataCost`. All subclasses must implement the __init__ and __call__ methods
    that return the data attachment term.

    This method is used to compute the data attachment term in the optimization process.
    It is meant to be given to a child of `Optimize_geodesicShooting`.

    Methods
    -------
    __init__(self, target, **kwargs)
        Initializes the class with the given target.

    __repr__(self)
        Returns a string representation of the DataCost object.

    set_optimizer(self, optimizer)
        Sets the optimizer object. Used during the initialization of the optimizer.

    to_device(self, device)
        Moves the target to the specified device.

    __call__(self, at_step=-1, **kwargs)
        Abstract method that must be implemented by subclasses to return the data attachment term.

    Parameters
    ----------
    target
         target image
    """

    @abstractmethod
    def __init__(self, target, **kwargs):
        self.target = target
        super(DataCost, self).__init__()

    def __repr__(self):
        return f"DataCost  :({self.__class__.__name__})"

    def set_optimizer(self, optimizer):
        """
        DataCost object are meant to be used along a
        method inherited from `Optimize_geodesicShooting`.
        This method is used to set the optimizer object and is usually
        used at the optimizer initialisation.
        """
        self.optimizer = optimizer
        if self.target.shape != self.optimizer.source.shape and not self.target is None:
            raise ValueError(
                "Target and source shape are different."
                f"Got source.shape = {self.optimizer.source.shape}"
                f"and target.shape = {self.target.shape}."
                f"Have you checked your DataCost initialisation ?"
            )

    def to_device(self, device):
        self.target = self.target.to(device)

    @abstractmethod
    def __call__(self, at_step=None, **kwargs):
        if not hasattr(self, "optimizer"):
            raise AttributeError("optimizer has not been initialized, you need to call `set_optimizer(mr)` before calling a DataCost object.")

        """
        :return:
        """
        return 0


class Ssd(DataCost):
    r"""
    This class is used to compute the data attachment term
    as a Sum of Squared Differences (SSD) term. It takes as a parameter
    the target image.
    $$SSD(I,T) = \frac 12 \|I - T\|_2^2 = \frac 12\sum_{x\in \Omega} (I - T)^2$$

    Parameters
    ----------
    target
      torch.Tensor of shape [B,C,H,W] or [B,C,D,H,W]  Target image

    Examples
    --------

    .. code-block:: python

        target = torch.rand(1,1,100,100)
        data_term = dt.Ssd(target)
        mt.lddmm(source,target,geodesic,
            optimizer_method='adadelta',
            data_term = data_term
        )
    """

    def __init__(self, target, **kwargs):
        super(Ssd, self).__init__(target)
        self.ssd = cf.SumSquaredDifference(target)

    def __call__(self, at_step=None):
        """
        Computes the Sum of Squared Differences (SSD) data attachment term.

        Parameters
        ----------
        at_step : int, optional
            The step at which to compute the SSD. If None, computes the SSD
            for the current image. If an integer is provided, computes the SSD
            for the image at the specified step. It is used for longitudinal data terms.

        Returns
        -------
        torch.Tensor
            The computed SSD value.
        """
        super().__call__()
        if at_step is None:
            return self.ssd(self.optimizer.mp.image)
        else:
            return self.ssd(self.optimizer.mp.image_stock[at_step][None])

    def to_device(self, device):
        self.ssd.target = self.ssd.target.to(device)


class Ssd_normalized(DataCost):
    r"""
    This class is used to compute the data attachment term
    as a Sum of Squared Differences (SSD) term but normalized by the number of pixels. It takes as a parameter
    the target image.
    $$SSD(I,T) = \frac 1{2 \#\Omega} \|I - T\|_2^2 = \frac 1{2 \#\Omega}\sum_{x\in \Omega} (I - T)^2$$
    where $\Omega$ is the set of pixels and $\# \Omega$ is the number of pixels.

    Parameters
    ----------
    target
      torch.Tensor of shape [B,C,H,W] or [B,C,D,H,W]  Target image
    """

    def __init__(self, target, **kwargs):
        super(Ssd_normalized, self).__init__(target)
        self.ssd = cf.SumSquaredDifference(target)

    def __call__(self, at_step=None):
        """
        Computes the normalized Sum of Squared Differences (SSD)
        data attachment term.

        Parameters
        ----------
        at_step : int, optional
            The step at which to compute the SSD. If None, computes the SSD
            for the current image. If an integer is provided, computes the SSD
            for the image at the specified step. It is used for longitudinal data terms.

        Returns
        -------
        torch.Tensor
            The computed SSD value.
        """
        super().__call__()
        # print("in ssd normalized img shape",self.optimizer.mp.image.shape)
        if at_step is None:
            return self.ssd(self.optimizer.mp.image) / prod(
                self.optimizer.mp.image.shape[2:]
            )
        else:
            return self.ssd(self.optimizer.mp.image_stock[at_step][None]) / prod(
                self.optimizer.mp.image.shape[2:]
            )

    def to_device(self, device):
        self.ssd.target = self.ssd.target.to(device)


class Cfm(DataCost):
    """This class is used to compute the data attachment term
    as a Cost Function Masking (CFM) term. It takes as a parameter
    the target image and the mask where the sum must be ignored.

    Parameters
    ----------
    target
        torch.Tensor of shape [B,C,H,W] or [B,C,D,H,W], Target image
    mask
        torch.Tensor of the same shape as target

    """

    def __init__(self, target, mask, **kwargs):
        super(Cfm, self).__init__(target)
        self.cfm = cf.SumSquaredDifference(target, cancer_seg=mask)

    def __call__(self, at_step=None):
        super().__call__()
        if at_step is None:
            return self.cfm(self.optimizer.mp.image)
        else:
            return self.cfm(self.optimizer.mp.image_stock[at_step][None])


class SimiliSegs(DataCost):
    """
    Rather than computing the SSD between the source and target images,
    this class computes the SSD between two given masks placed on the
    source and target masks respectively.

    Parameters
    ----------
    mask_source
        torch.Tensor of shape [B,C,H,W] or [B,C,D,H,W], Source mask
    mask_target
        torch.Tensor of the same shape as mask_source, Target mask
    """

    def __init__(self, mask_source, mask_target, **kwargs):
        super(SimiliSegs, self).__init__(None)
        self.mask_source = mask_source
        self.mask_target = mask_target

    def set_optimizer(self, optimizer):
        super(SimiliSegs, self).set_optimizer(optimizer)
        self.optimizer.mp._force_save = True

    def to_device(self, device):
        super(SimiliSegs, self).to_device(device)

    def __call__(self, at_step=None):
        super().__call__()
        if at_step == -1:
            at_step = None
        mask_deform = tb.imgDeform(
            self.mask_source.cpu(),
            self.optimizer.mp.get_deformator(to_t=at_step).to("cpu"),
            dx_convention=self.optimize.dx_convention,
        )
        return (mask_deform - self.mask_target).pow(2).sum() * 0.5


class Mutlimodal_ssd_cfm(DataCost):
    """
    This class is used to compute the data attachment term
    as a combination of the Sum of Squared Differences (SSD) and
    the Cost Function Masking (CFM) terms on multimodal
    (or multichannel) images. It allows to compute the SSD on
    selected channels of the source image and the CFM on the
    remaining channels.

    Parameters
    ----------
    target_ssd
        torch.Tensor of shape [B,C,H,W] or [B,C,D,H,W], Target image for the SSD term
    target_cfm
        torch.Tensor of the same shape as target_ssd, Target image for the CFM term
    source_cfm
        torch.Tensor of the same shape as target_ssd, Source image for the CFM term
    mask_cfm
        torch.Tensor of the same shape as target_ssd, Mask for the CFM term
    """

    def __init__(self, target_ssd, target_cfm, source_cfm, mask_cfm, **kwargs):
        super(Mutlimodal_ssd_cfm, self).__init__(None)
        self.cost = cf.Combine_ssd_CFM(target_ssd, target_cfm, mask_cfm)
        self.source_cfm = source_cfm

    def __call__(self, at_step=None):
        deformator = self.optimizer.mp.get_deformator(to_t=at_step).to(
            self.source_cfm.device
        )
        source_deform = tb.imgDeform(
            self.source_cfm, deformator, dx_convention=self.optimizer.dx_convention
        )
        if at_step is None:
            return self.cost(self.optimizer.mp.image, source_deform)
        else:
            return self.cost(
                self.optimizer.mp.image_stock[at_step][None], source_deform
            )

    def set_optimizer(self, optimizer):
        super(Mutlimodal_ssd_cfm, self).set_optimizer(optimizer)
        self.optimizer.mp._force_save = True

    def to_device(self, device):
        self.source_cfm = self.source_cfm.to(device)

class Mutual_Information(DataCost):
    r"""
    Mutual information measures the amount of information shared between two images. It is effective for multi-modal image registration.

    .. math::

        I(X;Y) = \sum_{x \in X} \sum_{y \in Y} p(x,y) \log \left(\frac{p(x,y)}{p(x)p(y)}\right)

    Where:

    - $X$ and $Y$ are the images being registered.
    - $p(x,y)$ is the joint probability distribution of the intensities.
    - $p(x)$ and $p(y)$ are the marginal probability distributions of the intensities.

    Parameters
    ---------------------
    target : torch.Tensor
        Target image [B,C,H,W] or [B,C,D,H,W]
    bins : int
        Number of bins for the histogram (default : 20)
    min : float
        Minimum value for the histogram (default : 0)
    max : float
        Maximum value for the histogram (default : 1)
    mult : float
        Multiplicative factor for the mutual information (default : 1.0)

    """

    def __init__(self,target,
                 bins = 20,
                 min = 0,
                 max = 1,
                 mult = 1.0,
                 ):
        super(Mutual_Information, self).__init__(target)
        # self.target = target
        self.mult = mult
        self.mi = cf.Mutual_Information(bins, min, max)

    def __call__(self, at_step=-1):
        super().__call__()
        if at_step == -1:
            mi = self.mi(self.optimizer.mp.image,self.target)
            return self.mult/mi
        else:
            return self.mult / self.mi(self.optimizer.mp.image_stock[at_step],self.target)

    def to_device(self, device):
        self.target = self.target.to(device)



class Longitudinal_DataCost(DataCost):
    """This class is used to compute the data
        attachment term for longitudinal data. It takes
         as a parameter an object inherited from `DataCost'
         and apply the sum of the data attachment term over
          the list of target images.

    Parameters
    ----------
    target_dict
        List of dict of target images.  Each dict must contain the key `time` with an integer value corresponding to the time of the data acquisition. The rest of the keys must by the one required by the provided data_cost object. (see example)
    data_cost
        DataCost object (default : Ssd)

    Example
    -------
        >>> from demeter.metamorphosis.data_cost import Cfm,Longitudinal_DataCost
        >>> data_cost = Cfm
        >>> target_dict = [
        >>>         {'time':0,'target':torch.rand(1,1,100,100),'mask':torch.rand(1,1,100,100)},
        >>>         {'time':6,'target':torch.rand(1,1,100,100),'mask':torch.rand(1,1,100,100)},
        >>>         {'time':10,'target':torch.rand(1,1,100,100),'mask':torch.rand(1,1,100,100)}
        >>>     ]
        >>> ldc = Longitudinal_DataCost(target_dict,data_cost)
    """

    def __init__(self, target_dict, data_cost: DataCost = Ssd, **kwargs):

        super(Longitudinal_DataCost, self).__init__(None)
        self.target_dict = target_dict
        self.target_len = len(target_dict)
        self.baseline_dataCost_list = []
        for td in target_dict:
            bdc = data_cost(**td)
            self.baseline_dataCost_list.append(bdc)

    def __call__(self, at_step=None):
        """ """
        super().__call__()
        cost = 0
        for td, bdc in zip(self.target_dict, self.baseline_dataCost_list):
            cost += bdc(at_step=td["time"])
            # image_t  = self.optimizer.mp.image_stock[td['time']]
        return cost

    def set_optimizer(self, optimizer):
        super(Longitudinal_DataCost, self).set_optimizer(optimizer)
        self.optimizer.mp._force_save = True
        self.optimizer.mp._detach_image = False
        for bdc in self.baseline_dataCost_list:
            bdc.set_optimizer(self.optimizer)

    def to_device(self, device):
        for td in self.target_dict:
            for key in td.keys():
                if key == "time":
                    continue
                td[key] = td[key].to(device)


import matplotlib.pyplot as plt
from demeter.utils.image_3d_plotter import get_orthogonal_views_concatenated
class Rotation_Ssd_Cost(DataCost):
    r"""
    Mixture of data costs

    D(I,T) =  gamma *| S \cdot A.T  - T |^2 + (1 - gamma) * | I_1 \cdot A.T - T|^2
    """
    def __init__(self, target,
                 gamma_mode = 'constant',
                 gamma_kwargs : dict = {'gamma': .5},
                 # gamma = None,
                # sigmoid_a = None,
                #  sigmoid_b = None,
                #  sigmoid_c = 4,
                 edges_computes = 1e-3,
                 normalize_ssd = False,
                 verbose = False,
                 plot = False,
                 save_plot=None,
                 save_values= False,
                 **kwargs):

        super(Rotation_Ssd_Cost, self).__init__(target)
        self.ssd = cf.SumSquaredDifference(target)
        self.save_values = save_values
        self._init_compute_gamma_(gamma_mode, gamma_kwargs)

        self.verbose = verbose
        self.plot = plot
        self.save_plot = save_plot
        self.normalize_ssd = normalize_ssd
        self.edges_computes = edges_computes

        if self.save_values:
            self.stock_ssd = torch.empty(200)
            self.stock_ssd_rot = torch.empty(200)
            self.stock_gamma = torch.empty(200)

    def __repr__(self):
        return super().__repr__() + self.gamma_mode # + self.gamma_kwargs

    def _init_compute_gamma_(self, gamma_mode, gamma_kwargs):
        self.gamma_mode = gamma_mode
        if self.gamma_mode == 'constant':
            self.gamma = gamma_kwargs["gamma"]
        elif self.gamma_mode == 'sigmoid':
            self.sigmoid_a = gamma_kwargs["sigmoid_a"]
            self.sigmoid_b = gamma_kwargs["sigmoid_b"]
            self.sigmoid_c = gamma_kwargs["sigmoid_c"]
        elif "variationnal" == self.gamma_mode:
            self.save_values = True
            self.c = gamma_kwargs["c"] # Passe haut pour K
            self.nu = gamma_kwargs["nu"] # Dampening gamma
        else:
            raise ValueError("gamma_mode must be among ['constant', 'sigmoid', 'variationnal']")

    def _compute_ssd_rot_derivative(self):
        _iter = self.optimizer._iter_
        win = 6
        if _iter > win:
            diff = self.stock_ssd_rot[_iter - win +1:_iter] - self.stock_ssd_rot[_iter-win:_iter-1]
        # elif _iter > 1:
        #     diff = self.stock_ssd_rot[1:_iter] - self.stock_ssd_rot[:_iter-1]
        else:
            diff = torch.tensor(-self.c, dtype=torch.float)
        return diff.mean()


    def _compute_gamma_(self, iter):
        if self.gamma_mode == 'constant':
            return self.gamma
        elif "sigmoid" in self.gamma_mode:
            alpha = 2 * self.sigmoid_c /( self.sigmoid_b - self.sigmoid_a)
            beta = - (self.sigmoid_a + self.sigmoid_b) / 2
            g = alpha *( iter + beta)
            gamma = 1/(1 + exp(-g))
            return gamma
        elif "variationnal" in self.gamma_mode:
            # c = 10 # Passe bas pour K
            # nu = .05 # Dampening gamma
            if iter == 0:
                return 1

            old_gamma = self.stock_gamma[iter - 1]
            if old_gamma == 0:
              return 0
            d_r = self._compute_ssd_rot_derivative()
            K = torch.min(torch.tensor(1), - d_r / self.c)
            gamma = old_gamma + (K - old_gamma) * self.nu
            # ic(iter, d_r, K, old_gamma, (K - old_gamma) * self.nu,gamma)
            return  torch.clip(gamma, 0,1)

    def plot_cost_data_term(self):
      fig, ax = plt.subplots(1,2, figsize=(10, 5))

      # Main axis
      ax[0].plot(self.stock_ssd, label="ssd (D(p))")
      ax[0].plot(self.stock_ssd_rot, label="ssd_rot (R(p))")
      ax[0].set_ylabel("SSD terms")

      # Secondary axis
      ax2 = ax[0].twinx()
      ax2.plot(self.stock_gamma, label="gamma", color="green")
      ax2.set_ylabel("Gamma")

      # Combine legends
      lines_1, labels_1 = ax[0].get_legend_handles_labels()
      lines_2, labels_2 = ax2.get_legend_handles_labels()
      ax[0].legend(lines_1 + lines_2, labels_1 + labels_2)

      dict_loss = {
          "ssd (D(p))": self.stock_ssd,
          "ssd_rot (R(p))": self.stock_ssd_rot,
          "Gamma": self.stock_gamma
      }


    def _plot_3d_(self, rotated_image, rotated_source, gamma, ssd, ssd_rot):
        fig, ax = plt.subplots(3,2, figsize=(7,10), constrained_layout=True)
        B,_,D,H,W = rotated_image.shape
        coord = (D//2, H//2, W//2+5)
        cmp_im1 = tb.imCmp(rotated_image, self.target, "compose")[0]
        cmp_im2 = tb.imCmp(rotated_source, self.target, "compose")[0]
        im1 = get_orthogonal_views_concatenated(cmp_im1,coord)
        im2 = get_orthogonal_views_concatenated(cmp_im2,coord)
        rotated_img = get_orthogonal_views_concatenated(rotated_image[0,0], coord)
        rotated_source = get_orthogonal_views_concatenated(rotated_source[0,0], coord)
        target = get_orthogonal_views_concatenated(self.target[0,0], coord)
        source = get_orthogonal_views_concatenated(self.optimizer.source[0,0], coord)
        fig.suptitle(f" iter : {self.optimizer._iter_}: gamma = {gamma:.3f}; ssd = {ssd:.2f}, ssd_rot = {ssd_rot:.2f}")
        ax[0,0].imshow(source, cmap='gray')
        ax[0,0].set_title('source')
        ax[0,1].imshow(target, cmap='gray')
        ax[0,1].set_title('target')

        ax[1,0].imshow(rotated_img, cmap='gray')
        ax[1,0].set_title('rotated Image')
        ax[1,1].imshow(rotated_source, cmap='gray')
        ax[1,1].set_title('rotated Source')
        ax[2,0].imshow(im1)
        # ax[1,0].imshow(torch.abs(rotated_image - self.target)[0,0].detach().cpu().numpy())
        ax[2,0].set_title('rot img vs target')
        ax[2,1].imshow(im2)
        # ax[1,1].imshow(torch.abs(rotated_source - self.target)[0,0].detach().cpu().numpy())
        ax[2,1].set_title('rot source vs target')
        if self.save_plot is not None:
            fig.savefig(str(self.save_plot) + f"_{self.optimizer._iter_:03d}.png")
        plt.show()

    def _plot_2d_(self, rotated_image, rotated_source, gamma, ssd, ssd_rot):
        fig, ax = plt.subplots(2,3, constrained_layout=True)

        fig.suptitle(f" iter : {self.optimizer._iter_}: gamma = {gamma:.3f}; ssd = {ssd:.2f}, ssd_rot = {ssd_rot:.2f}")
        ax[0,0].imshow(rotated_image[0,0].detach().cpu().numpy(), cmap='gray')
        ax[0,0].set_title('rotated Image')
        ax[0,1].imshow(rotated_source[0,0].detach().cpu().numpy(), cmap='gray')
        ax[0,1].set_title('rotated Source')
        im1 = tb.imCmp(rotated_image.detach().cpu(), self.target.detach().cpu(), "compose")
        ax[1,0].imshow(im1[0])
        # ax[1,0].imshow(torch.abs(rotated_image - self.target)[0,0].detach().cpu().numpy())
        ax[1,0].set_title('rot img vs target')
        im2 = tb.imCmp(rotated_source.detach().cpu(), self.target.detach().cpu(), "compose")
        ax[1,1].imshow(im2[0])
        # ax[1,1].imshow(torch.abs(rotated_source - self.target)[0,0].detach().cpu().numpy())
        ax[1,1].set_title('rot source vs target')
        ax[0,2].imshow(self.optimizer.target.detach().cpu().numpy()[0,0], cmap='gray')
        ax[0,2].set_title('target')
        ax[1,2].imshow(self.optimizer.source.detach().cpu().numpy()[0,0], cmap='gray')
        ax[1,2].set_title('source')
        if self.save_plot is not None:
            fig.savefig(str(self.save_plot) + f"_{self.optimizer._iter_:03d}.png")
        plt.show()


    def __call__(self,at_step=None):
        super().__call__()

        gamma = self._compute_gamma_(self.optimizer._iter_)
        grid_rt = self.optimizer.mp.get_affine_deformator()
        # grid_rt = self.optimizer.mp.get_affine_deformation()

        if gamma > 1 - self.edges_computes:
            # ic("Skip compute ssd", gamma)
            rotated_image =  tb.imgDeform(self.optimizer.mp.image.detach(),grid_rt.detach(),dx_convention='2square')
            ssd = torch.tensor(0)
        else:
            rotated_image =  tb.imgDeform(self.optimizer.mp.image,grid_rt,dx_convention='2square')
            ssd = self.ssd(rotated_image)

        if gamma < self.edges_computes:
            rotated_source = tb.imgDeform(self.optimizer.source.detach(),grid_rt.detach(),dx_convention='2square')
            ssd_rot = torch.tensor(0)
            # ic("Skip compute ssd_rot", gamma)
        else:
            rotated_source = tb.imgDeform(self.optimizer.source,grid_rt,dx_convention='2square')
            ssd_rot = self.ssd(rotated_source)

        if self.normalize_ssd:
            ssd /=  prod(rotated_source.shape)
            ssd_rot /= prod(rotated_source.shape)
        if self.verbose:
            print(f"[{self.__repr__()}]")
            print(f"\t gamma = {gamma:.3f} : ssd = {ssd:.3f}, ssd_rot = {ssd_rot:.3f} => Loss = {gamma * ssd_rot + (1-gamma) * ssd:.3f} ")
        # if self.optimizer._iter_  % 5 == 0 and self.plot:
        if self.plot:
            if len(rotated_source.shape) == 4:
                self._plot_2d_(rotated_image, rotated_source, gamma, ssd, ssd_rot)
            elif len(rotated_source.shape) == 5:
                self._plot_3d_(rotated_image, rotated_source, gamma, ssd, ssd_rot)

        if self.save_values:
            self.stock_ssd[self.optimizer._iter_] = ssd.detach()
            self.stock_ssd_rot[self.optimizer._iter_] = ssd_rot.detach()
            self.stock_gamma[self.optimizer._iter_] = gamma
        return gamma * ssd_rot + (1-gamma) * ssd

#
class Rotation_MutualInformation_Cost(DataCost):
    """
    This class combine a DataCost object with rotation.
    """
    def __init__(self, target, alpha : float, **kwargs):

        super(Rotation_MutualInformation_Cost, self).__init__(target)
        self.mutual_info = cf.Mutual_Information()
        self.alpha = alpha



    def __call__(self,at_step=None):
        # if at_step == -1:
        super().__call__()
        # rot_def =   tb.grid_from_rotation(self.optimizer.mp.id_grid, self.optimizer.mp.rot_mat.T)
        rot_def = self.optimizer.mp.get_affine_deformator()
        # if self.optimizer.mp.flag_translation:
            # raise Error("Ca va bugger, fait une expe avant.")
        rot_def += self.optimizer.mp.translation

        rotated_image =  tb.imgDeform(self.optimizer.mp.image,rot_def,dx_convention='2square')
        rotated_source = tb.imgDeform(self.optimizer.source,rot_def,dx_convention='2square')

        cost = self.mutual_info(rotated_image, self.target.to(rotated_image.dtype))
        ssd_rot = self.mutual_info(rotated_source, self.target.to(rotated_source.dtype))

        return self.alpha * ssd_rot + (1-self.alpha) * cost