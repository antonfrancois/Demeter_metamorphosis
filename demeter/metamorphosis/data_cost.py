import torch
from abc import ABC, abstractmethod

from ..utils import torchbox as tb
from ..utils import cost_functions as cf
from math import prod

class DataCost(ABC,torch.nn.Module):
    """
    Abstract class for the data attachment term in the metamorphosis optimisation.
    The class `Optimize_geodesicShooting` require
    a data attachment term to be provided as a child class of `DataCost`.
    All child classes must implement the __init__ and__call__ methods
     that returns the data attachment term.
    """

    @abstractmethod
    def __init__(self,target,**kwargs):
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
        if self.target.shape != self.optimizer.source.shape\
                and not self.target is None:
            raise ValueError("Target and source shape are different."
                             f"Got source.shape = {self.optimizer.source.shape}"
                             f"and target.shape = {self.target.shape}."
                             f"Have you checked your DataCost initialisation ?")

    def to_device(self,device):
        self.target = self.target.to(device)

    @abstractmethod
    def __call__(self,at_step = -1,**kwargs):
        """
        :return:
        """
        return 0


class Ssd(DataCost):
    """
    This class is used to compute the data attachment term
    as a Sum of Squared Differences (SSD) term. It takes as a parameter
    the target image.

    Parameters
    ----------

    target : torch.Tensor  Target image
    """

    def __init__(self,target,**kwargs):
        super(Ssd, self).__init__(target)
        self.ssd = cf.SumSquaredDifference(target)

    def __call__(self,at_step = None):
        if at_step is None:
            return self.ssd(self.optimizer.mp.image) #/ prod(self.optimizer.mp.image.shape[2:])
        else:
            return self.ssd(self.optimizer.mp.image_stock[at_step][None])  #/ prod(self.optimizer.mp.image.shape[2:])

    def to_device(self,device):
        self.ssd.target = self.ssd.target.to(device)

class Ssd_normalized(DataCost):

    def __init__(self,target,**kwargs):
        super(Ssd_normalized,self).__init__(target)
        self.ssd = cf.SumSquaredDifference(target)

    def __call__(self, at_step=None):
        # print("in ssd normalized img shape",self.optimizer.mp.image.shape)
        if at_step is None:
            return self.ssd(self.optimizer.mp.image) / prod(self.optimizer.mp.image.shape[2:])
        else:
            return self.ssd(self.optimizer.mp.image_stock[at_step][None])  / prod(self.optimizer.mp.image.shape[2:])

    def to_device(self,device):
        self.ssd.target = self.ssd.target.to(device)

class Cfm(DataCost):
    def __init__(self,target,mask,**kwargs):
        """  This class is used to compute the data attachment term
        as a Cost Function Masking (CFM) term. It takes as a parameter
        the target image and the mask where the sum must be ignored.

        Parameters:
        ----------
        target : torch.Tensor
          Target image
        mask : torch.Tensor
            of the same shape as target

        """
        super(Cfm, self).__init__(target)
        self.cfm = cf.SumSquaredDifference(target,cancer_seg=mask)

    def __call__(self,at_step = None):
        if at_step is None:
            return self.cfm(self.optimizer.mp.image)
        else:
            return self.cfm(self.optimizer.mp.image_stock[at_step][None])

class SimiliSegs(DataCost):
    """ Make the deformation register segmentations."""

    def __init__(self,mask_source,mask_target,**kwargs):
        super(SimiliSegs, self).__init__(None)
        self.mask_source = mask_source
        self.mask_target = mask_target

    def set_optimizer(self, optimizer):
        super(SimiliSegs, self).set_optimizer(optimizer)
        self.optimizer.mp._force_save = True

    def to_device(self,device):
        super(SimiliSegs, self).to_device(device)

    def __call__(self,at_step = None):
        if at_step == -1: at_step = None
        mask_deform = tb.imgDeform(self.mask_source.cpu(),
                                   self.optimizer.mp.get_deformator(to_t=at_step).to('cpu'),
                                   dx_convention=self.optimize.dx_convention)
        return  (mask_deform - self.mask_target).pow(2).sum()*.5

class Mutlimodal_ssd_cfm(DataCost):

    def __init__(self,target_ssd,target_cfm,source_cfm,mask_cfm,**kwargs):
        super(Mutlimodal_ssd_cfm, self).__init__(None)
        self.cost = cf.Combine_ssd_CFM(target_ssd,target_cfm,mask_cfm)
        self.source_cfm = source_cfm

    def __call__(self,at_step = None):
        deformator = self.optimizer.mp.get_deformator(to_t=at_step).to(self.source_cfm.device)
        source_deform = tb.imgDeform(self.source_cfm,deformator,dx_convention=self.optimizer.dx_convention)
        if at_step is None:
            return self.cost(self.optimizer.mp.image,source_deform)
        else:
            return self.cost(self.optimizer.mp.image_stock[at_step][None],source_deform)

    def set_optimizer(self, optimizer):
        super(Mutlimodal_ssd_cfm, self).set_optimizer(optimizer)
        self.optimizer.mp._force_save = True

    def to_device(self,device):
        self.source_cfm = self.source_cfm.to(device)


class Longitudinal_DataCost(DataCost):

    def __init__(self, target_dict,data_cost = Ssd, **kwargs):
        """ This class is used to compute the data
        attachment term for longitudinal data. It takes
         as a parameter an object inherited from `DataCost'
         and apply the sum of the data attachment term over
          the list of target images.

         target_dict : List of dict of target images.
            Each dict must contain the key `time` with an
             integer value corresponding to the time of the
             data acquisition. The rest of the keys must by the one
             required by the provided data_cost object. (see example)
         data_cost : DataCost object (default : Ssd)

         Example:
            ```python
            >>> from metamorphosis.data_cost import Cfm,Longitudinal_DataCost
            >>> data_cost = Cfm
            >>> target_dict = [
            >>>         {'time':0,'target':torch.rand(1,1,100,100),'mask':torch.rand(1,1,100,100)},
            >>>         {'time':6,'target':torch.rand(1,1,100,100),'mask':torch.rand(1,1,100,100)},
            >>>         {'time':10,'target':torch.rand(1,1,100,100),'mask':torch.rand(1,1,100,100)}
            >>>     ]
            >>> ldc = Longitudinal_DataCost(target_dict,data_cost)
            ```
        """
        super(Longitudinal_DataCost, self).__init__(None)
        self.target_dict = target_dict
        self.target_len = len(target_dict)
        self.baseline_dataCost_list = []
        for td in target_dict:
            bdc = data_cost(**td)
            self.baseline_dataCost_list.append(bdc)

    def __call__(self,at_step = None):
        """

        """
        cost = 0
        for td, bdc in zip(self.target_dict,self.baseline_dataCost_list):
            cost += bdc(at_step=td['time'])
            # image_t  = self.optimizer.mp.image_stock[td['time']]
        return cost

    def set_optimizer(self, optimizer):
        super(Longitudinal_DataCost, self).set_optimizer(optimizer)
        self.optimizer.mp._force_save = True
        self.optimizer.mp._detach_image = False
        for bdc in self.baseline_dataCost_list:
            bdc.set_optimizer(self.optimizer)

    def to_device(self,device):
        for td in self.target_dict:
            for key in td.keys():
                if key == 'time': continue
                td[key] = td[key].to(device)