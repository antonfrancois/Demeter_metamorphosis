"""
This module contains the class for the joined mask metamorphosis.
but it is broken for the moment. Please come back later.
"""


import torch
import matplotlib.pyplot as plt
from math import prod,sqrt

from demeter.constants import *
from demeter.metamorphosis import Geodesic_integrator,Optimize_geodesicShooting

import demeter.utils.torchbox as tb
import demeter.utils.cost_functions as cf


# class Mask_intensity(ABC):
#     """ Class for customizing the intensity addition area at the image update state.
#
#
#
#     """
#     def __init__(self,precomputed_mask):
#         self.precomputed_mask = precomputed_mask
#
#     @abstractmethod
#     def __call__(self,integrated_mask,i):
#         pass
#
#     def to_device(self,device):
#         self.precomputed_mask = self.precomputed_mask.to(device)
#
#     @abstractmethod
#     def derivative(self,*args):
#         pass
#
#     def get_mask(self):
#         return 0
#
# class mask_default(Mask_intensity):
#
#     def __init__(self):
#         pass
#
#     def __repr__(self):
#         return self.__class__.__name__ + " - no precomputed mask"
#
#     def __call__(self,integrated_mask,i):
#         return integrated_mask
#
#     def to_device(self,device):
#         pass
#
#     def derivative(self,*args):
#         return 1
#
#
# class mask_sum(Mask_intensity):
#     def __init__(self,precomputed_mask):
#         super().__init__(precomputed_mask)
#
#     def __repr__(self):
#         return self.__class__.__name__ + " - precomputed mask: f(N_t,P_t) = N_t + P_t"
#
#     def __call__(self,integrated_mask,i):
#         # self.integrated_mask = integrated_mask
#         return integrated_mask + self.precomputed_mask[i][None]
#
#     def derivative(self,*args):
#         return 1
#
#     def get_mask(self):
#         return self.precomputed_mask.detach().cpu()

class Weighted_joinedMask_Metamorphosis_integrator(Geodesic_integrator):
    """

    Dans cette version, on considère que l'image intégrée
    est une image à deux canaux, le premier étant l'image
    le second le masque. (i.e.: image.shape[1] == 2)
    avec image[:,0] l'image et image[:,1] le masque.

    """
    def __init__(self,rho,
                 **kwargs
                 ):
        super(Weighted_joinedMask_Metamorphosis_integrator, self).__init__(**kwargs)
        self.rho = rho

        # if mask_function is None:
        #     self.mask_function = mask_default()
        # elif isinstance(mask_function,Mask_intensity):
        #     self.mask_function = mask_function
        # else:
        #     raise ValueError("mask_function must be a Mask_intensity object")


        # if rho_M == 0:
        #     self.channel_weight = torch.tensor([self.rho_I, 1])
        # else:
        self.channel_weight = torch.tensor([ 1, self.rho])[None,:]

    def __repr__(self):
        return (self.__class__.__name__ +
                f"(\n\trho (mask) = {self.rho},\n" +
                f"\t\t{self.kernelOperator.__repr__()} \n\t)")


    def _field_I_cst_mult_(self):
        if self.mu_I == 0 and self.rho_I == 0:
            return 1
        else:
            return self.rho_I/self.mu_I

    def _field_M_cst_mult_(self):
        if self.mu_M == 0 and self.rho_M == 0:
            return 1
        else:
            return self.rho_M/self.mu_M

    def step(self):
        ## update field
        # Note that self.image is a concatenation of the image and the mask
        grad_image_mask = tb.spatialGradient(self.image,dx_convention=self.dx_convention)

        masks = torch.stack(
            [
            self.image[:,1].clone(),
            self.rho * torch.ones_like(self.image[:,1])
            ], dim=1
        ).detach()
        field_momentum = ((torch.sqrt(masks) * self.momentum) * grad_image_mask).sum(dim=1)
        # print("field_momentum",field_momentum.shape)
        # print("masks",masks.shape)
        self.field = tb.im2grid(self.kernelOperator(-(field_momentum)))

        try:
            volDelta = prod(self.kernelOperator.dx)
        except AttributeError:
            volDelta = 1

        ## prepare mask for group multiplication
        ## prepare deformation
        # sq_msk = torch.sqrt(self.image[0,1])
        deform_I = (self.id_grid
                    - masks[0,0][...,None] * self.field / self.n_step)
        deform_M = (self.id_grid
                    - self.rho * self.field / self.n_step)
        deform = torch.cat([deform_I,deform_M],dim=0)
        assert not torch.isnan(deform).any(), "NaN detected in deform"
        assert not torch.isinf(deform).any(), "Inf detected in deform"
        # ic(self._i,self.field.min().item(),self.field.max().item(),
        #    self.id_grid.min().item(),self.id_grid.max().item(),
        #    torch.sqrt(masks).mean().item(),sqrt(self.rho)
        #    )

        # # update image and mask
        # apply deformation to both the image and mask
        image_defI = tb.imgDeform(self.image[:,0][None],deform_I,dx_convention=self.dx_convention)
        image_defM = tb.imgDeform(self.image[:,1][None],deform_M,dx_convention=self.dx_convention)
        image_def = torch.cat([image_defI,image_defM],dim=1)
        ic(image_def.shape)

        # image_def = tb.imgDeform(self.image.transpose(0,1),deform,dx_convention=self.dx_convention).transpose(0,1)
        # image_def *= torch.sqrt(masks)

        # add the residual times the mask to the image
        # resi_I = (1 - self.image[:,1]) * self.momentum[:,0]
        # resi_M = (1 - self.rho) * self.momentum[:,1]
        # self.residuals = torch.stack([resi_I,resi_M],dim=1)
        self.residuals = (1 - masks) * self.momentum / volDelta

        if self.debug:
            print("i : ",self._i)
            fig,ax = plt.subplots(1,2)
            a=  ax[0].imshow(self.momentum[0,0].detach().cpu(),cmap='gray')
            plt.colorbar(a,ax=ax[0])
            b = ax[1].imshow(self.momentum[0,1].detach().cpu(),cmap='gray')
            plt.colorbar(b,ax=ax[1])
            plt.show()
        self.image = image_def + self.residuals / self.n_step
        assert not torch.isnan(self.momentum).any(), "NaN detected in self.momentum"
        assert not torch.isinf(self.momentum).any(), "Inf detected in self.momentum"

        ic(self.momentum.transpose(0,1).shape, deform.shape)
        # # update residual
        momentum_I = self._compute_div_momentum_semiLagrangian_(
            deform_I,
            self.momentum[:,0][None],
            torch.sqrt(masks[0,0])[None]
        )
        momentum_M = self._compute_div_momentum_semiLagrangian_(
            deform_M,
            self.momentum[:,1][None],
            torch.sqrt(masks[0,1])[None]
        )

        self.momentum = torch.cat([momentum_I,momentum_M],dim=1)
        # self.momentum = self._compute_div_momentum_semiLagrangian_(
        #     deform,
        #     self.momentum.transpose(0,1),
        #     torch.sqrt(masks).transpose(0,1)
        # ).transpose(0,1)
        assert not torch.isnan(self.momentum).any(), "NaN detected in self.momentum"
        assert not torch.isinf(self.momentum).any(), "Inf detected in self.momentum"

        return (self.image, self.field, self.residuals)

    def step_sharp(self):
        ## update field
        # ic("update field")
        # ic(self.image.shape)
        grad_image_mask = tb.spatialGradient(self.image,dx_convention='pixel')
        # ic(grad_image_mask.max())
#         ic(grad_image_mask.shape)
#         ic(self.residuals.shape)
        cst_I, cst_M = self._field_I_cst_mult_(), self._field_M_cst_mult_()
        # ic(grad_image_mask.shape)
        # ic(self.residuals.shape)
        pre_field_I = cst_I * self.residuals[:,0] * grad_image_mask[:,0]
        # ic(pre_field_I.shape)
#         ic(pre_field_I.shape)
#         ic((self.residuals[:,0] * grad_image_mask[:,0]).shape)
#         ic((self.residuals[:,1] * grad_image_mask[:,1]).shape)

        pre_field = (pre_field_I
                     + cst_M * self.residuals[:,1] * grad_image_mask[:,1])#.sum(dim=1,keepdim=True)
        # ic(pre_field.shape)
        self.field = tb.im2grid(self.kernelOperator(-(pre_field)))
#         ic(self.field.shape)
#         ic(self.field.max())

        ## prepare deformation
#         ic("prepare deformation")
        deform = self.id_grid - self.field / self.n_step
        # resi_to_add = self.residuals

        # # update image and mask
        #         ic("update image and mask")
        # apply deformation to both the image and mask
        self.image = tb.imgDeform(self.image,deform,dx_convention='pixel')
        # ic(self.image.shape)
        # add the residual times the mask to the image
        resi_I = self.residuals[:,0].clone()
        resi_M = self.residuals[:,1].clone()

        if self.mu_I > 0:
            self.image[:,0] = (self.image[:,0]
                               + self.mu_I * resi_I * self.mask_function(self.image[:,1].clone(),self._i) / self.n_step)
        # if self.mu_I > 0:
        #     self.image[:,0] = self.image[:,0] + self.mu_I * resi_I * torch.rand(self.image[:,0].shape).to(device) / self.n_step

        # ic(self.image.shape)
        if self.mu_M > 0:
            self.image[:,1] = self.image[:,1] + self.mu_M * resi_M/self.n_step

        # # update residual
#         ic(self.residuals.shape)
        z_I = self.residuals[:,0]
        self._update_momentum_semiLagrangian_(deform)
        # add term into the residual of the mask
        self.residuals[:,1] = (self.residuals[:,1]
                + self.mask_function.derivative(self.image[:,1].clone(),self._i) * z_I**2)
        # ic(self.residuals.shape)

        return (self.image, self.field, self.residuals)

    def forward(self,image_mask,
                # mask,
                momentum_ini =0,
                **kwargs):
        # concatenate the image and the mask at dim = 2
        # stack_image = torch.cat([image,mask],dim=1)
        if type(momentum_ini) == int:
            momentum_ini = momentum_ini * torch.ones_like(image_mask).to(image_mask.device)
        self.to_device(image_mask.device)
        super(Weighted_joinedMask_Metamorphosis_integrator, self).forward(image_mask,momentum_ini,**kwargs)


    def to_device(self,device):
        # self.channel_weight = self.channel_weight.to(device)
        super(Weighted_joinedMask_Metamorphosis_integrator, self).to_device(device)
        # self.mask_function.to_device(device)

    def plot(self,n_figs=5):
        if n_figs == -1:
            n_figs = self.n_step
        plot_id = torch.quantile(torch.arange(self.image_stock.shape[0],dtype=torch.float),
                                 torch.linspace(0,1,n_figs)).round().int()
        v_abs_max = torch.quantile(self.residuals.abs(),.99)

        fig,ax = plt.subplots(n_figs,5,figsize=(20,15))

        for i,t in enumerate(plot_id):
            ax[i,0].imshow(self.image_stock[t,0].cpu(),**DLT_KW_IMAGE)
            ax[i,0].set_title('image')
            ax[i,1].imshow(self.image_stock[t,1].cpu(),**DLT_KW_IMAGE)
            ax[i,1].set_title('mask')
            ax[i,2].imshow(self.momentum_stock[t,0].cpu(),**DLT_KW_RESIDUALS)
            ax[i,2].set_title('residuals_image')
            ax[i,3].imshow(self.momentum_stock[t,1].cpu(),**DLT_KW_RESIDUALS)
            ax[i,3].set_title('residuals_image')
            # ax[i,4].imshow(self.residuals_stock[t].cpu(),vmin=-v_abs_max,vmax=v_abs_max,**tb.DLT_KW_IMAGE)
            # ax[i,4].set_title('residual')
            tb.gridDef_plot_2d(self.get_deformation(t),
                            add_grid=True,
                            ax=ax[i,-1],
                            step=int(min(self.field_stock.shape[2:-1])/30),
                            check_diffeo=True)



class Weighted_joinedMask_Metamorphosis_Shooting(Optimize_geodesicShooting):
    """
    Perform the optimization of the weighted joined mask metamorphosis


    Warning: self.source and self.target are a concatenation of the image and the mask
    ```
    source = torch.cat([source,mask_source],dim=1)
    ```
    """
    def __init__(self,
                source,target,mask_source,mask_target,
                 geodesic: Weighted_joinedMask_Metamorphosis_integrator,
                 cost_cst : float,
                 data_term = None,
                 optimizer_method='LBFGS_torch',
                 **kwargs
                 ):

        source = torch.cat([source,mask_source],dim=1)
        target = torch.cat([target,mask_target],dim=1)
        super().__init__(source, target, geodesic, cost_cst,data_term, optimizer_method)

        self._cost_saving_ = self._joined_cost_saving_

        # Après réflexion, je pense qu'on as pas besoin de ça.
        # # Re-Initialise data_term, it was already initialised in parent class,
        # # but we need to override it for this method.
        # self.data_term = mt.Ssd(self.stack_target)
        # self.data_term.set_optimizer(self)

        self._plot_forward_ = self._plot_forward_joined_

    def get_all_arguments(self):
        # TODO:  use super for kernelOp, n_step ....
        return {
            'lambda':self.cost_cst,
            'rho':self.rho,
            'kernelOperator':self.kernelOperator.__repr__(),
            'n_step':self.mp.n_step,
            'sharp':self.mp.flag_sharp,
        }

    def get_ssd_def(self,only_image = False):
        image_def = tb.imgDeform(self.source,self.mp.get_deformator(),dx_convention=self.dx_convention)
        if only_image:
            return float(cf.SumSquaredDifference(self.target[:,0][None])(image_def[:,0][None]))
        else:
            return float(cf.SumSquaredDifference(self.target)(image_def))

    def cost(self, momentum_ini: torch.Tensor) -> torch.Tensor:
        lamb = self.cost_cst
        ic(momentum_ini.min().item(),momentum_ini.max().item())
        self.mp.forward(self.source, momentum_ini, save=False, plot=0)

         # Compute the data_term, a specific data term has been written for this class
        self.data_loss = self.data_term()

        # Norm V
        self.norm_v_2 = self._compute_V_norm_(momentum_ini, self.source)

        # Norm L2 on z_I
        mask = self.source[0,1]
        mI =  momentum_ini[0,0]
        self.norm_zI_2 =  ((1 - mask) * mI * mI).sum() /prod(self.source.shape[2:])
        # Norm L2 on z_M

        # if self.mp.rho_M != 0:
            # Norm L2 on z_M
        mM = momentum_ini[0,1]
        self.norm_zM_2 = (1 - self.mp.rho) * (mM  * mM).sum()/prod(self.source.shape[2:])
        self.total_cost = (self.data_loss
                           + lamb * (self.norm_v_2
                                     + self.norm_zI_2
                                     + self.norm_zM_2)
                           )

        return self.total_cost

    def _plot_forward_joined_(self):
        fig,ax = plt.subplots(1,4,figsize=(20,5))
        ax[0].imshow(self.mp.image[0,0].detach().cpu(),**DLT_KW_IMAGE)
        ax[1].imshow(self.mp.image[0,1].detach().cpu(),**DLT_KW_IMAGE)
        a = ax[2].imshow(self.mp.momentum[0,0].detach().cpu(),cmap='RdYlBu_r',origin='lower')
        fig.colorbar(a,ax=ax[2],fraction=0.046, pad=0.04)
        b = ax[3].imshow(self.mp.residuals[0,0].detach().cpu(),cmap='RdYlBu_r',origin='lower')
        fig.colorbar(b,ax=ax[3],fraction=0.046, pad=0.04)
        plt.show()

    def _joined_cost_saving_(self,i,loss_stock):
        """A variation of Optimize_geodesicShooting._default_cost_saving_

        :param i: index for saving the according values
                !!! if `loss_stock` is None, `loss_stock` will be initialized, and
                `i` must have the value of the number of iterations.
        :param loss_stock:
        :return: updated `loss_stock`"""
        # Initialise loss_stock
        if loss_stock is None:
            d = 4
            return torch.zeros((i, d))

        loss_stock[i,0] = self.data_loss.cpu().detach()
        loss_stock[i,1] = self.norm_v_2.cpu().detach()
        loss_stock[i,2] = self.norm_zI_2.cpu().detach()
        loss_stock[i,3] = self.norm_zM_2.cpu().detach()

        return loss_stock

    def plot_cost(self):
        fig1, ax1 = plt.subplots(1, 2,figsize=(10,10))

        cost_stock = self.to_analyse[1].detach().numpy()


        ssd_plot = cost_stock[:, 0]
        ax1[0].plot(ssd_plot, "--", color='blue', label='ssd')
        ax1[1].plot(ssd_plot, "--", color='blue', label='ssd')

        if self.cost_cst != 0:
            normv_plot = cost_stock[:, 1] * self.cost_cst
            ax1[0].plot(normv_plot, "--", color='green', label='normv')
            ax1[1].plot(cost_stock[:, 1], "--", color='green', label='normv')
            total_cost = ssd_plot + normv_plot

            norm_l2_on_zI = cost_stock[:, 2] * self.cost_cst
            total_cost += norm_l2_on_zI
            ax1[0].plot(norm_l2_on_zI, "--", color='orange', label='norm L2 on zI')
            ax1[1].plot(cost_stock[:, 2], "--", color='orange', label='norm L2 on zI')

            norm_l2_on_zM = cost_stock[:, 3] * self.cost_cst
            total_cost += norm_l2_on_zM
            ax1[0].plot(norm_l2_on_zM, "--", color="purple", label='norm L2 on zM')
            ax1[1].plot(cost_stock[:, 3], "--", color='purple', label='norm L2 on zM')

        ax1[0].plot(total_cost, color='black', label=r'\Sigma')
        ax1[0].legend()
        ax1[1].legend()

        ax1[0].set_title(f"Lambda = {str(self.cost_cst)},"+
                         f" rho = {str(self.mp.rho)},"
                         )

    def plot_imgCmp(self):
        r""" Display and compare the deformed image $I_1$ with the target$
        """
        fig,ax = plt.subplots(4,2,figsize = (10,20),constrained_layout=True)
        image_kw = dict(cmap='gray',origin='lower',vmin=0,vmax=1)
        set_ticks_off(ax)
        def plot(ax,i):
            ftsz = 11
            ax[0,0].imshow(self.source[0,i,:,:].detach().cpu().numpy(),
                           **image_kw)
            ax[0,0].set_title("source",fontsize = ftsz)
            ax[0,1].imshow(self.target[0,i,:,:].detach().cpu().numpy(),
                           **image_kw)
            ax[0,1].set_title("target",fontsize = ftsz)

            ax[1,1].imshow(
                tb.imCmp(
                    self.target[0,i][None,None],
                    self.mp.image.detach().cpu()[0,i][None,None],
                    method='compose'
                ),
                **image_kw)
            ax[1,1].set_title("comparaison deformed image with target",fontsize = ftsz)
            ax[1,0].imshow(self.mp.image[0,i].detach().cpu().numpy(),**image_kw)
            ax[1,0].set_title("Integrated source image",fontsize = ftsz)
            tb.quiver_plot(self.mp.get_deformation()- self.id_grid,
                           ax=ax[1,1],step=15,color=GRIDDEF_YELLOW,
                           )

            text_param = (f"Lambda = {str(self.cost_cst)},"+
                         f" rho = {str(self.mp.rho)},")
            try:
                text_param += f" gamma = {self.mp._get_gamma_()}"
            except AttributeError:
                pass
            ax[1,1].text(10,self.source.shape[2] - 10,text_param ,c="white",size=ftsz)

            text_score = ""
            if type(self.get_DICE()) is float:
                text_score += f"dice : {self.get_DICE():.2f},"

            if type(self.get_landmark_dist()) is float:
                localSource_ldmk_kw = source_ldmk_kw.copy()
                localSource_ldmk_kw['markersize'] = 10
                localTarget_ldmk_kw = target_ldmk_kw.copy()
                localTarget_ldmk_kw['markersize'] = 10
                localDeform_ldmk_kw = deform_ldmk_kw.copy()
                localDeform_ldmk_kw['markersize'] = 10
                ax[1,1].plot(self.source_landmark[:,0],self.source_landmark[:,1],**localSource_ldmk_kw)
                ax[1,1].plot(self.target_landmark[:,0],self.target_landmark[:,1],**localTarget_ldmk_kw)
                ax[1,1].plot(self.deform_landmark[:,0],self.deform_landmark[:,1],**localDeform_ldmk_kw)
                ax[1,1].quiver(self.source_landmark[:,0],self.source_landmark[:,1],
                               self.deform_landmark[:,0]-self.source_landmark[:,0],
                               self.deform_landmark[:,1]-self.source_landmark[:,1],
                                 color= "#2E8DFA")
                ax[1,1].legend()
                text_score += f"landmark : {self.get_landmark_dist():.2f},"
            ax[1,1].text(10,10,text_score,c='white',size=ftsz)

        plot(ax[0:2,0:2],0)
        plot(ax[2:,0:2],1)

        return fig,ax