import matplotlib.pyplot as plt
import torch

import utils.torchbox as tb


class SumSquaredDifference:
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
        else:
            print("WARNING : la formule est fausse, corrige ça Anton du futur")
            return .5*((self.target - self.cancer_seg * self.source_deform)**2).sum()


    def imgShow(self,axes = None):
        if axes is None:
            fig,axes = plt.subplots(1,2)
        axes[0].imshow(tb.imCmp(self.source,self.target))
        axes[0].set_title('before')

        axes[1].imshow(tb.imCmp(self.source_deform,self.target))
        axes[1].set_title('after registration')

        # plt.show()
        return axes

class Combine_ssd_CFM:

    def __init__(self,ssd_target,cfm_target,seg,cancer_on_target = True):
        self.B_ssd,self.B_cfm = ssd_target.shape[0],cfm_target.shape[0]
        self.ssd = SumSquaredDifference(ssd_target)
        self.cfm = SumSquaredDifference(cfm_target,seg,cancer_on_target=cancer_on_target)

    def __call__(self,img_ssd,img_cfm):
        if img_ssd.shape[0] != self.B_ssd:
            raise ValueError(f"The ssd comparison was initialized with batch size {self.B_ssd} "
                             f"and img_ssd is shape : {img_ssd}")
        if img_cfm.shape[0] != self.B_cfm:
            raise ValueError(f"The cfm comparison was initialized with batch size {self.B_cfm} "
                             f"and img_cfm is shape : {img_cfm}")

        return self.ssd(img_ssd)/self.B_ssd + self.cfm(img_cfm)/self.B_cfm