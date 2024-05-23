import torch
from abc import ABC, abstractmethod

import utils.torchbox as tb
import utils.cost_functions as cf


class DataCost(ABC,torch.nn.Module):

    @abstractmethod
    def __init__(self):
        super(DataCost, self).__init__()

    def __repr__(self):
        return f"DataCost  :({self.__class__.__name__})"

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def to_device(self,device):
        pass

    @abstractmethod
    def __call__(self):
        """
        :return:
        """
        return 0


class Ssd(DataCost):

    def __init__(self,target):
        super(Ssd, self).__init__()
        self.ssd = cf.SumSquaredDifference(target)

    def __call__(self):
        return self.ssd(self.optimizer.mp.image)

    def to_device(self,device):
        self.ssd.target = self.ssd.target.to(device)

class Cfm(DataCost):
    def __init__(self,target,mask):
        super(Cfm, self).__init__()
        self.cfm = cf.SumSquaredDifference(target,cancer_seg=mask)

    def __call__(self):
        return self.cfm(self.optimizer.mp.image)

class SimiliSegs(DataCost):
    """ Make the deformation register segmentations."""

    def __init__(self,mask_source,mask_target):
        super(SimiliSegs, self).__init__()
        self.mask_source = mask_source
        self.mask_target = mask_target

    def set_optimizer(self, optimizer):
        super(SimiliSegs, self).set_optimizer(optimizer)
        self.optimizer.mp._force_save = True

    def to_device(self,device):
        super(SimiliSegs, self).to_device(device)

    def __call__(self):
        mask_deform = tb.imgDeform(self.mask_source.cpu(),
                                   self.optimizer.mp.get_deformator(),
                                   dx_convention='pixel')
        return  (mask_deform - self.mask_target).pow(2).sum()*.5

class Mutlimodal_ssd_cfm(DataCost):

    def __init__(self,target_ssd,target_cfm,source_cfm,mask_cfm):
        super(Mutlimodal_ssd_cfm, self).__init__()
        self.cost = cf.Combine_ssd_CFM(target_ssd,target_cfm,mask_cfm)
        self.source_cfm = source_cfm

    def __call__(self):
        deformator = self.optimizer.mp.get_deformator().to(self.source_cfm.device)
        source_deform = tb.imgDeform(self.source_cfm,deformator,dx_convention='pixel')
        return self.cost(self.optimizer.mp.image,source_deform)

    def set_optimizer(self, optimizer):
        super(Mutlimodal_ssd_cfm, self).set_optimizer(optimizer)
        self.optimizer.mp._force_save = True

    def to_device(self,device):
        self.source_cfm = self.source_cfm.to(device)