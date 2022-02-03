import matplotlib.pyplot as plt
import torch

import my_torchbox as tb


class sumSquaredDifference:
    """ Compute the sum squared difference of two images """

    def __init__(self, target, cancer_seg=None,cancer_on_target=True):
        # self.source =source
        self.target = target
        # print('posterior init : source '+ str(source.shape))
        self.field_size = target.shape[-2:]
        # self.field_size = (source.shape[3],source.shape[2])
        self.cancer_on_target = cancer_on_target
        self.cancer_seg = cancer_seg  #mask # TODO : verifier que masks size = [N,0,H,W]

    def __call__(self,source_deform=None):
        return self.function(source_deform)

    def function(self,source_deform):
        """

        :param field_diff: tensor (H,W,2)
        :return:
        """
        self.source_deform = source_deform
        # print('poterior function : source_deform '+str(self.source_deform.shape))
        if self.cancer_seg is None:
            return .5*((self.target - self.source_deform)**2).sum()
        elif self.cancer_on_target:
            return .5*(self.cancer_seg* (self.target - self.source_deform)**2).sum()



    def imgShow(self,source,axes = None):
        if axes is None:
            fig,axes = plt.subplots(1,2)
        axes[0].imshow(tb.imCmp(source,self.target))
        axes[0].set_title('before')

        axes[1].imshow(tb.imCmp(self.source_deform,self.target))
        axes[1].set_title('after registration')

        # plt.show()
        return axes

