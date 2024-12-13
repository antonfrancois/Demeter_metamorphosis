import torch
import warnings
import matplotlib.pyplot as plt
from math import prod

from abc import ABC, abstractmethod
from demeter.metamorphosis import Geodesic_integrator,Optimize_geodesicShooting

from demeter.utils.constants import *
import demeter.utils.torchbox as tb
import demeter.utils.cost_functions as cf

class Mask_intensity(ABC):
    """ Class for customizing the intensity addition area at the image update state.



    """
    def __init__(self,precomputed_mask):
        self.precomputed_mask = precomputed_mask

    @abstractmethod
    def __call__(self,integrated_mask,i):
        pass

    def to_device(self,device):
        self.precomputed_mask = self.precomputed_mask.to(device)

    @abstractmethod
    def derivative(self,*args):
        pass

    def get_mask(self):
        return 0

class mask_default(Mask_intensity):

    def __init__(self):
        pass

    def __repr__(self):
        return self.__class__.__name__ + " - no precomputed mask"

    def __call__(self,integrated_mask,i):
        return integrated_mask

    def to_device(self,device):
        pass

    def derivative(self,*args):
        return 1


class mask_sum(Mask_intensity):
    def __init__(self,precomputed_mask):
        super().__init__(precomputed_mask)

    def __repr__(self):
        return self.__class__.__name__ + " - precomputed mask: f(N_t,P_t) = N_t + P_t"

    def __call__(self,integrated_mask,i):
        # self.integrated_mask = integrated_mask
        return integrated_mask + self.precomputed_mask[i][None]

    def derivative(self,*args):
        return 1

    def get_mask(self):
        return self.precomputed_mask.detach().cpu()

class Weighted_joinedMask_Metamorphosis_integrator(Geodesic_integrator):
    """

    Dans cette version, on considère que l'image intégrée
    est une image à deux canaux, le premier étant l'image
    le second le masque. (i.e.: image.shape[1] == 2)
    avec image[:,0] l'image et image[:,1] le masque.

    """
    def __init__(self,rho_I,
                 mu_I=1,
                 rho_M=0, # It is best to keep rho_M = mu_M = 0
                 mu_M=0,  # but id mu_M = 0, rho_M must be 0
                 sigma_v=(1, 1),
                 mask_function : Mask_intensity = None, # used for adding a precomputed mask to the image intensity addintions.
                 n_step=None,
                 border_effect=True,
                 sharp = False,
                 **kwargs
                 ):
        super(Weighted_joinedMask_Metamorphosis_integrator, self).__init__(sigma_v)
        if mu_I < 0: mu_I = 0
        if mu_M is None or rho_M is None: mu_M, rho_M = 0, 0
        if mu_M < 0: mu_M = 0
        if mu_M == 0 and rho_M>0:
            warnings.warn("mu_M <= 0 but rho_M > 0, rho_M will be set to 0")
            rho_M = mu_M = 0
        self.mu_I, self.mu_M = mu_I, mu_M
        self.rho_I, self.rho_M = rho_I, rho_M
        self.n_step = n_step

        if mask_function is None:
            self.mask_function = mask_default()
        elif isinstance(mask_function,Mask_intensity):
            self.mask_function = mask_function
        else:
            raise ValueError("mask_function must be a Mask_intensity object")


        if rho_M == 0:
            self.channel_weight = torch.tensor([self.rho_I, 1])
        else:
            self.channel_weight = torch.tensor([ self.rho_I, self.rho_M])[None,:]

    def __repr__(self):
        mask_identity = self.mask_function.__repr__()
        return self.__class__.__name__ + f"(\n\t\tmask intensity additons : {mask_identity}),\n" \
                                         f"\t\t{self.kernelOperator.__repr__()} \n\t)"

    def _get_mu_(self):
        return self.mu_I,self.mu_M


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
        # ic("update field")
        # ic(self.image.shape)
        grad_image_mask = tb.spacialGradient(self.image,dx_convention='pixel')
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
        self._update_residuals_semiLagrangian_(deform)
        # add term into the residual of the mask
        self.residuals[:,1] = (self.residuals[:,1]
                + self.mask_function.derivative(self.image[:,1].clone(),self._i) * z_I**2)
        # ic(self.residuals.shape)

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
        self._update_residuals_semiLagrangian_(deform)
        # add term into the residual of the mask
        self.residuals[:,1] = (self.residuals[:,1]
                + self.mask_function.derivative(self.image[:,1].clone(),self._i) * z_I**2)
        # ic(self.residuals.shape)

        return (self.image, self.field, self.residuals)

    def forward(self,image_mask,
                # mask,
                residual =0,
                **kwargs):
        # concatenate the image and the mask at dim = 2
        # stack_image = torch.cat([image,mask],dim=1)
        if type(residual) == int:
            residual = residual * torch.ones_like(image_mask).to(image_mask.device)
        self.to_device(image_mask.device)
        super(Weighted_joinedMask_Metamorphosis_integrator, self).forward(image_mask,residual,**kwargs)


    def to_device(self,device):
        # self.channel_weight = self.channel_weight.to(device)
        # super(Weighted_joinedMask_Metamorphosis_integrator, self).to_device(device)
        self.mask_function.to_device(device)

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
            ax[i,2].imshow(self.residuals_stock[t,0].cpu(),**DLT_KW_RESIDUALS)
            ax[i,2].set_title('residuals_image')
            ax[i,3].imshow(self.residuals_stock[t,1].cpu(),**DLT_KW_RESIDUALS)
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

    # def __repr__(self):


    def _get_mu_I(self):
        return self.mp.mu_I

    def _get_mu_M(self):
        return self.mp.mu_M

    def _get_mu_(self):
        # TODO : ce n'est pas kacher
        return (self._get_mu_I(), self._get_mu_M())
    def _get_rho_I(self):
        return self.mp.rho_I

    def _get_rho_M(self):
        return self.mp.rho_M

    def _get_rho_(self):
        # TODO : ce n'est pas kacher
        return (self._get_rho_I(), self._get_rho_M())

    def get_ssd_def(self,only_image = False):
        image_def = tb.imgDeform(self.source,self.mp.get_deformator(),dx_convention='pixel')
        if only_image:
            return float(cf.SumSquaredDifference(self.target[:,0][None])(image_def[:,0][None]))
        else:
            return float(cf.SumSquaredDifference(self.target)(image_def))

    def cost(self,residuals_ini: torch.Tensor) -> torch.Tensor:
        lamb = self.cost_cst
        rho = self._get_rho_()

        self.mp.forward(self.source, residuals_ini, save=False, plot=0)

         # Compute the data_term, a specific data term has been written for this class
        self.data_loss = self.data_term()

        # Norm V
        self.norm_v_2 = self._compute_V_norm_(residuals_ini,self.source)

        # Norm L2 on z_I
        zI = residuals_ini[0,0]
        mask = self.source[0,1]
        self.norm_zI_2 = self.mp.rho_I * (zI * mask * zI).sum() /prod(self.source.shape[2:])
        # Norm L2 on z_M
        self.total_cost = self.data_loss + lamb * (self.norm_v_2 + self.norm_zI_2)

        if self.mp.rho_M != 0:
            # Norm L2 on z_M
            zM = residuals_ini[0,1]
            self.norm_zM_2 = self.mp.rho_M * (zM  * zM).sum()/prod(self.source.shape[2:])
            self.total_cost += lamb * self.norm_zM_2

        return self.total_cost

    def _plot_forward_joined_(self):
        fig,ax = plt.subplots(1,2,figsize=(10,5))
        ax[0].imshow(self.mp.image[0,0].detach().cpu(),**DLT_KW_IMAGE)
        ax[1].imshow(self.mp.image[0,1].detach().cpu(),**DLT_KW_IMAGE)

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
        if self.mp.rho_M != 0:
            loss_stock[i,3] = self.norm_zM_2.cpu().detach()

        return loss_stock

    def plot_cost(self):
        fig1, ax1 = plt.subplots(1, 2,figsize=(10,10))

        cost_stock = self.to_analyse[1].detach().numpy()

        ssd_plot = cost_stock[:, 0]
        ax1[0].plot(ssd_plot, "--", color='blue', label='ssd')
        ax1[1].plot(ssd_plot, "--", color='blue', label='ssd')

        normv_plot = cost_stock[:, 1] / self.cost_cst
        ax1[0].plot(normv_plot, "--", color='green', label='normv')
        ax1[1].plot(cost_stock[:, 1], "--", color='green', label='normv')
        total_cost = ssd_plot + normv_plot

        norm_l2_on_zI = cost_stock[:, 2] / self.cost_cst * self.mp.rho_I
        total_cost += norm_l2_on_zI
        ax1[0].plot(norm_l2_on_zI, "--", color='orange', label='norm L2 on zI')
        ax1[1].plot(cost_stock[:, 2], "--", color='orange', label='norm L2 on zI')

        if self.mp.rho_M != 0:
            norm_l2_on_zM = cost_stock[:, 3] / self.cost_cst * self.mp.rho_M
            total_cost += norm_l2_on_zM
            ax1[0].plot(norm_l2_on_zM, "--", color="purple", label='norm L2 on zM')
            ax1[1].plot(cost_stock[:, 3], "--", color='purple', label='norm L2 on zM')

        ax1[0].plot(total_cost, color='black', label=r'\Sigma')
        ax1[0].legend()
        ax1[1].legend()
        str_cst = "" if self.mp.mu_M == 0 else (f"mu_M = {str(self.mp.mu_M)},"
                                                 f" rho_M = {str(self.mp.rho_M)},")
        ax1[0].set_title(f"Lambda = {str(self.cost_cst)},"+
                         f" mu_I = {str(self.mp.mu_I)}," +
                         f" rho_I = {str(self.mp.rho_I)},"+
                            str_cst
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
            str_cst = "" if self.mp.mu_M == 0 else (f"mu_M = {str(self.mp.mu_M)},"
                                                 f" rho_M = {str(self.mp.rho_M)},")

            text_param = (f"Lambda = {str(self.cost_cst)},"+
                         f" mu_I = {str(self.mp.mu_I)}," +
                         f" rho_I = {str(self.mp.rho_I)},"+ str_cst)
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