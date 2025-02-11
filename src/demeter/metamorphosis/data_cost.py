"""
This module contains the classes used to compute the data attachment term
in the metamorphosis optimization. All data attachment terms must herit from
the abstract class `DataCost`. The module contains the following classes:
`Ssd`, `Ssd_normalized`, `Cfm`, `SimiliSegs`, `Mutlimodal_ssd_cfm`, `Longitudinal_DataCost`.
"""

import torch
from abc import ABC, abstractmethod

from ..utils import torchbox as tb
from ..utils import cost_functions as cf
from math import prod


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
