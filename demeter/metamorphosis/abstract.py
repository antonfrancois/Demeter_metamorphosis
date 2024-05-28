import torch
import matplotlib.pyplot as plt
import warnings
from math import prod
import pickle
import os, sys, csv#, time
from icecream import ic

sys.path.append(os.path.abspath('../'))
from datetime import datetime
from abc import ABC, abstractmethod

from utils.optim import GradientDescent
from utils.constants import *
import utils.reproducing_kernels as rk
import utils.torchbox as tb
import utils.vector_field_to_flow as vff
from utils.toolbox import update_progress,format_time, get_size, fig_to_image,save_gif_with_plt
from utils.decorators import time_it
import utils.cost_functions as cf
import utils.fill_saves_overview as fill_saves_overview

import metamorphosis.data_cost as dt

# =========================================================================
#
#            Abstract classes
#
# =========================================================================
# See them as a toolkit

class Geodesic_integrator(torch.nn.Module,ABC):
    """ Abstract class for defining the way of integrating over geodesics

    """
    @abstractmethod
    def __init__(self,sigma_v,multiScale_average= False):
        super().__init__()
        self._force_save = False

        if isinstance(sigma_v,tuple):
            self.sigma_v = sigma_v# is used if the optmisation is later saved
            self.kernelOperator = rk.GaussianRKHS(sigma_v,
                                                     border_type='constant')
        elif isinstance(sigma_v,list):
            self.sigma_v = sigma_v
            if multiScale_average:
                self.kernelOperator = rk.Multi_scale_GaussianRKHS_notAverage(sigma_v)
            else:
                self.kernelOperator = rk.Multi_scale_GaussianRKHS(sigma_v)
        else:
            ValueError("Something went wrong with sigma_v")
        # self.border_effect = border_effect
        # self._init_sharp_()

    def _init_sharp_(self,sharp):
        # print(f'sharp = {sharp}')
        if sharp is None:
            try:
                sharp = self.flag_sharp
            except AttributeError:
                sharp = False
        if not sharp:
            self.flag_sharp = False
            return 0
        if self.__class__.__name__ == 'Metamorphosis_path':
            self.step = self._step_sharp_semiLagrangian
        self.flag_sharp = True
        self.save = True
        self._force_save = True
        self._phis = [[None]*i for i in range(1,self.n_step+1)]
        self._resi_deform = []

    @abstractmethod
    def step(self):
        pass

    @abstractmethod
    def _get_mu_(self):
        pass

    def forward(self, image, momentum_ini,
                field_ini=None,
                save=True,
                plot =0,
                t_max = 1,
                verbose=False,
                sharp=None,
                debug= False):
        r""" This method is doing the temporal loop using the good method `_step_`

        :param image: (tensor array) of shape [1,1,H,W]. Source image ($I_0$)
        :param field_ini: to be deprecated, field_ini is id_grid
        :param momentum_ini: (tensor array) of shape [H,W]. note that for images
        the momentum is equal to the residual ($p = z$)
        :param save: (bool) option to save the integration intermediary steps.
        :param plot: (int) positive int lower than `self.n_step` to plot the indicated
                         number of intemediary steps. (if plot>0, save is set to True)

        """
        if len(momentum_ini.shape) not in [4, 5]:
            raise ValueError(f"residual_ini must be of shape [B,C,H,W] or [B,C,D,H,W] got {momentum_ini.shape}")
        device = momentum_ini.device
        # print(f'sharp = {sharp} flag_sharp : {self.flag_sharp},{self._phis}')
        self._init_sharp_(sharp)
        self.source = image.detach()
        self.image = image.clone().to(device)
        self.momentum = momentum_ini
        self.debug = debug
        try:
            self.save = True if self._force_save else save
        except AttributeError:
            self.save = save

        self.id_grid = tb.make_regular_grid(momentum_ini.shape[2:], device=device)
        assert self.id_grid != None

        if field_ini is None:
            self.field = self.id_grid.clone()
        else:
            self.field = field_ini #/self.n_step

        if plot > 0:
            self.save = True

        if self.save:
            self.image_stock = torch.zeros((t_max*self.n_step,)+image.shape[1:])
            self.field_stock = torch.zeros((t_max*self.n_step,)+self.field.shape[1:])
            self.momentum_stock = torch.zeros((t_max * self.n_step,) + momentum_ini.shape[1:])

        for i,t in enumerate(torch.linspace(0,t_max,t_max*self.n_step)):
            self._i = i

            _,field_to_stock,residuals_dt = self.step()

            if self.image.isnan().any() or self.momentum.isnan().any():
                raise OverflowError("Some nan where produced ! the integration diverged",
                                    "changing the parameters is needed (increasing n_step can help) ")

            if self.save:
                self.image_stock[i] = self.image[0].detach().to('cpu')
                self.field_stock[i] = field_to_stock[0].detach().to('cpu')
                self.momentum_stock[i] = residuals_dt.detach().to('cpu')

            if verbose:
                update_progress(i/(t_max*self.n_step))

        # try:
        #     _d_ = device if self._force_save else 'cpu'
        #     self.field_stock = self.field_stock.to(device)
        # except AttributeError: pass

        if plot>0:
            self.plot(n_figs=plot)

    def _image_Eulerian_integrator_(self,image,vector_field,t_max,n_step):
        """ image integrator using an Eulerian scheme

        :param image: (tensor array) of shape [T,1,H,W]
        :param vector_field: (tensor array) of shape [T,H,W,2]
        :param t_max: (float) the integration will be made on [0,t_max]
        :param n_step: (int) number of time steps in between [0,t_max]

        :return: (tensor array) of shape [T,1,H,W] integrated with vector_field
        """
        dt = t_max/n_step
        for t in torch.linspace(0,t_max,n_step):
            grad_I = tb.spacialGradient(image,dx_convention='pixel')
            grad_I_scalar_v = (grad_I[0]*tb.grid2im(vector_field)).sum(dim=1)
            image = image - grad_I_scalar_v * dt
        return image

    def _compute_vectorField_(self, momentum, grad_image):
        r""" operate the equation $K \star (z_t \cdot \nabla I_t)$

        :param momentum: (tensor array) of shape [H,W] or [D,H,W]
        :param grad_image: (tensor array) of shape [B,C,2,H,W] or [B,C,3,D,H,W]
        :return: (tensor array) of shape [B,H,W,2]
        """
        # C = residuals.shape[1]
        return tb.im2grid(self.kernelOperator((-(momentum.unsqueeze(2) * grad_image).sum(dim=1))))

    def _compute_vectorField_multimodal_(self, momentum, grad_image):
        r""" operate the equation $K \star (z_t \cdot \nabla I_t)$

        :param momentum: (tensor array) of shape [B,C,H,W] or [B,C,D,H,W]
        :param grad_image: (tensor array) of shape [B,C,2,H,W] or [B,C,3,D,H,W]
        :return: (tensor array) of shape [B,H,W,2]
        """
        wheigths = self.channel_weight.to(momentum.device)
        W = wheigths.sum()
        # ic(residuals.shape,self.channel_weight.shape)
        return tb.im2grid(self.kernelOperator((
                -((wheigths * momentum).unsqueeze(2) * grad_image
                  ).sum(dim=1)
                # / W
        ))) # PAS OUF SI BATCH

    def _update_field_multimodal_(self):
        grad_image = tb.spacialGradient(self.image,dx_convention='pixel')
        self.field = self._compute_vectorField_multimodal_(self.momentum, grad_image)
        self.field *= self._field_cst_mult()

    def _field_cst_mult(self):
        if self._get_mu_() == 0:
            return 1
        else:
            return self._get_rho_()/self._get_mu_()

    def _update_field_(self):
        grad_image = tb.spacialGradient(self.image,dx_convention='pixel')
        self.field = self._compute_vectorField_(self.momentum, grad_image)
        self.field *= self._field_cst_mult()

    def _update_residuals_Eulerian_(self):
        residuals_dt = - tb.Field_divergence(dx_convention='pixel')(
            self.momentum[0,0][None, :, :, None] * self.field,
                            )

        self.momentum = self.momentum + residuals_dt / self.n_step

    def _update_residuals_semiLagrangian_(self,deformation):
        div_v_times_z = self.momentum * tb.Field_divergence(dx_convention='pixel')(self.field)[0,0]

        self.momentum = tb.imgDeform(self.momentum,
                                     deformation,
                                     dx_convention='pixel',
                                     clamp=False) \
                        - div_v_times_z / self.n_step

    def _compute_sharp_intermediary_residuals_(self):
        device = self.momentum.device
        resi_cumul = torch.zeros(self.momentum.shape, device=device)
        # for k,phi in enumerate(self._phis[self._i][:]):
        for k,phi in enumerate(self._phis[self._i][1:]):
            resi_cumul += tb.imgDeform(self.momentum_stock[k][None].to(device),
                                       phi,
                                       dx_convention='pixel',
                                       clamp=False)
        resi_cumul = resi_cumul + self.momentum
        return resi_cumul
        # Non sharp but working residual
        # if self._i >0:
        #     for k,z in enumerate(self._resi_deform):
        #         self._resi_deform[k] = tb.imgDeform(z[None,None].to(device),
        #                                             self._phis[self._i][self._i],
        #                                             'pixel')[0,0]
        #     self._phis[self._i - 1] = None
        # self._resi_deform.append(self.residuals.clone())


    def _update_image_Eulerian_(self):
        self.image =  self._image_Eulerian_integrator_(self.image,self.field,1/self.n_step,1)
        self.image = self.image + (self.momentum * self.mu) / self.n_step

    def _update_image_semiLagrangian_(self,deformation,residuals = None,sharp=False):
        if residuals is None: residuals = self.momentum
        image = self.source if sharp else self.image
        self.image = tb.imgDeform(image,deformation,dx_convention='pixel')
        if self._get_mu_() != 0: self.image += (residuals *self.mu)/self.n_step

    def _update_sharp_intermediary_field_(self):
        # print('update phi ',self._i,self._phis[self._i])
        self._phis[self._i][self._i] = self.id_grid - self.field/self.n_step
        if self._i > 0:
            for k,phi in enumerate(self._phis[self._i - 1]):
                self._phis[self._i][k] = phi + tb.compose_fields(
                    -self.field/self.n_step,phi,'pixel'
                ).to(self.field.device)
                # self._phis[self._i][k] = tb.compose_fields(
                #     phi,
                #     self._phis[self._i][self._i],
                #     # self.field/self.n_step,
                #     'pixel'
                # ).to(self.field.device)

    def _update_momentum_weighted_semiLagrangian_(self, deformation):
        f_t = self.rf.f(self._i)
        fz_times_div_v = f_t * self.momentum * tb.Field_divergence(dx_convention='pixel')(self.field)[0, 0]
        div_fzv = -tb.imgDeform(f_t * self.momentum,
                                deformation,
                                dx_convention='pixel',
                                clamp=False)[0, 0] \
                  + fz_times_div_v / self.n_step
        z_time_dtF = self.momentum * self.rf.dt_F(self._i)
        self.momentum = - (div_fzv + z_time_dtF) / f_t

    def _update_image_weighted_semiLagrangian_(self, deformation,residuals = None,sharp=False):
        if residuals is None: residuals = self.momentum
        image = self.source if sharp else self.image
        self.image = tb.imgDeform(image, deformation, dx_convention='pixel') + \
                     (self.rf.F_div_f(self._i) *  residuals) / self.n_step

                     # (self.rf.mu * self.rf.mask[self._i] * residuals) / self.n_step

        # ga = (self.rf.mask[self._i] * self.residuals) / self.n_step
        # plt.figure()
        # p = plt.imshow(ga[0])
        # plt.colorbar(p)
        # plt.show()

    def _update_field_oriented_weighted_(self):
        grad_image = tb.spacialGradient(self.image, dx_convention='pixel')
        if self.flag_W:
            free_field = tb.im2grid(
                # self.kernelOperator((- self.rf.f(self._i) * self.residuals * grad_image[0]))
                (- self.rf.f(self._i) * self.momentum * grad_image[0])
            )/self._get_mu_()
        else:
            free_field = self._compute_vectorField_(self.momentum, grad_image)
            free_field *= self._field_cst_mult()
        oriented_field = 0
        if self.flag_O:
            mask_i = self.orienting_mask[self._i][..., None].clone()
            free_field *= 1 / (1 + (self.gamma * mask_i))

            oriented_field = self.orienting_field[self._i][None].clone()
            oriented_field *= (self.gamma * mask_i) / (1 + (self.gamma * mask_i))


        # self.field = free_field + oriented_field
        self.field = tb.im2grid(self.kernelOperator(tb.grid2im(free_field + oriented_field)))


    def get_deformation(self,n_step=None,save=False):
        r"""Returns the deformation use it for showing results
        $\Phi = \int_0^1 v_t dt$

        :return: deformation [1,H,W,2] or [2,H,W,D,3]
        """
        if n_step == 0:
            return self.id_grid.detach().cpu() + self.field_stock[0][None].detach().cpu()/self.n_step
        temporal_integrator = vff.FieldIntegrator(method='temporal',save=save)
        if n_step is None:
            return temporal_integrator(self.field_stock/self.n_step,forward=True)
        else:
            return temporal_integrator(self.field_stock[:n_step]/self.n_step,forward=True)

    def get_deformator(self,n_step=None,save = False):
        r"""Returns the inverse deformation use it for deforming images
        $\Phi^{-1}$

        :return: deformation [T,H,W,2] or [T,H,W,D,3]
        """
        if n_step == 0:
            return self.id_grid.detach().cpu()-self.field_stock[0][None].detach().cpu()/self.n_step
        temporal_integrator = vff.FieldIntegrator(method='temporal',save=save)
        if n_step is None:
            return temporal_integrator(self.field_stock/self.n_step,forward=False)
        else:
            return temporal_integrator(self.field_stock[:n_step]/self.n_step,forward=False)

    # ==================================================================
    #                       PLOTS
    # ==================================================================

    def plot(self,n_figs=5):
        if n_figs ==-1:
            n_figs = self.n_step
        plot_id = torch.quantile(torch.arange(self.image_stock.shape[0],dtype=torch.float),
                                 torch.linspace(0,1,n_figs)).round().int()


        kw_image_args = dict(cmap='gray',
                      extent=[-1,1,-1,1],
                      origin='lower',
                      vmin=0,vmax=1)
        # v_abs_max = (self.residuals_stock.abs().max()).max()
        v_abs_max = torch.quantile(self.momentum.abs(), .99)
        kw_residuals_args = dict(cmap='RdYlBu_r',
                      extent=[-1,1,-1,1],
                      origin='lower',
                      vmin=-v_abs_max,
                      vmax=v_abs_max)
        size_fig = 5
        C = self.momentum_stock.shape[1]
        plt.rcParams['figure.figsize'] = [size_fig*3,n_figs*size_fig]
        fig,ax = plt.subplots(n_figs,2 + C)

        for i,t in enumerate(plot_id):
            i_s =ax[i,0].imshow(self.image_stock[t,:,:,:].detach().permute(1,2,0).numpy(),
                                **kw_image_args)
            ax[i,0].set_title("t = "+str((t/(self.n_step-1)).item())[:3])
            ax[i,0].axis('off')
            fig.colorbar(i_s,ax=ax[i,0],fraction=0.046, pad=0.04)

            for j in range(C):
                r_s =ax[i,j+1].imshow(self.momentum_stock[t,j].detach().numpy(),
                                      **kw_residuals_args)
                ax[i,j+1].axis('off')

            fig.colorbar(r_s,ax=ax[i,-2],fraction=0.046, pad=0.04)

            tb.gridDef_plot_2d(self.get_deformation(t),
                            add_grid=True,
                            ax=ax[i,-1],
                            step=int(min(self.field_stock.shape[2:-1])/30),
                            check_diffeo=True)

        return fig,ax

    def plot_deform(self,target,temporal_nfig = 0):

        if self.save == False:
            raise TypeError("metamophosis_path.forward attribute 'save' has to be True to use self.plot_deform")

        temporal = (temporal_nfig>0)
        # temporal integration over v_t
        temporal_integrator = vff.FieldIntegrator(method='temporal',
                                                  save=temporal,
                                                  dx_convention='pixel')


        # field_stock_toplot = tb.pixel2square_convention(self.field_stock)
        # tb.gridDef_plot(field_stock_toplot[-1][None],dx_convention='2square')
        if temporal:
            full_deformation_t = temporal_integrator(self.field_stock/self.n_step,
                                                     forward=True)
            full_deformator_t = temporal_integrator(self.field_stock/self.n_step,
                                                    forward=False)
            full_deformation = full_deformation_t[-1].unsqueeze(0)
            full_deformator = full_deformator_t[-1].unsqueeze(0)
        else:
            full_deformation = temporal_integrator(self.field_stock/self.n_step,
                                                   forward=True)
            full_deformator = temporal_integrator(self.field_stock/self.n_step,
                                                  forward=False)

        fig , axes = plt.subplots(3,2, constrained_layout=True,figsize=(20,30))
        # show resulting deformation

        tb.gridDef_plot_2d(full_deformation,step=int(max(self.image.shape)/30),ax = axes[0,0],
                         check_diffeo=True,dx_convention='2square')
        tb.quiver_plot(full_deformation -self.id_grid.cpu() ,step=int(max(self.image.shape)/30),
                        ax = axes[0,1],check_diffeo=False)

        # show S deformed by full_deformation
        S_deformed = tb.imgDeform(self.source.cpu(),full_deformator,
                                  dx_convention='pixel')
        # axes[1,0].imshow(self.source[0,0,:,:].cpu().permute(1,2,0),cmap='gray',origin='lower',vmin=0,vmax=1)
        # axes[1,1].imshow(target[0].cpu().permute(1,2,0),cmap='gray',origin='lower',vmin=0,vmax=1)
        # axes[2,0].imshow(S_deformed[0,0,:,:].permute(1,2,0),cmap='gray',origin='lower',vmin=0,vmax=1)
        # axes[2,1].imshow(tb.imCmp(target,S_deformed),origin='lower',vmin=0,vmax=1)

        axes[1,0].imshow(self.source[0,0,:,:].cpu(),cmap='gray',origin='lower',vmin=0,vmax=1)
        axes[1,1].imshow(target[0,0].cpu(),cmap='gray',origin='lower',vmin=0,vmax=1)
        axes[2,0].imshow(S_deformed[0,0,:,:],cmap='gray',origin='lower',vmin=0,vmax=1)
        axes[2,1].imshow(
            tb.imCmp(
                target[:,0][None],
                S_deformed[:,0][None],
                method='compose'
            ),origin='lower',vmin=0,vmax=1)

        set_ticks_off(axes)
        if temporal:
            t_max = full_deformator_t.shape[0]
            plot_id = torch.quantile(torch.arange(t_max,dtype=torch.float),
                                 torch.linspace(0,1,temporal_nfig)).round().int()
            size_fig = 5
            plt.rcParams['figure.figsize'] = [size_fig,temporal_nfig*size_fig]
            fig,ax = plt.subplots(temporal_nfig)

            for i,t in enumerate(plot_id):
                tb.quiver_plot(full_deformation_t[i].unsqueeze(0) - self.id_grid ,
                               step=10,ax=ax[i])
                tb.gridDef_plot(full_deformation_t[i].unsqueeze(0),
                                add_grid=False,step=10,ax=ax[i],color='green')

                tb.quiver_plot(self.field_stock[i].unsqueeze(0),
                               step=10,ax=ax[i],color='red')


    def save_to_gif(self,object,file_name,folder=None,delay=40,
                    clean= True):

        # prepare list of object
        if 'image' in object and 'deformation' in object:
            image_list_for_gif = []
            image_kw = dict()
            for n in range(self.n_step):
                deformation = self.get_deformation(n).cpu()
                img = self.image_stock[n,0].cpu().numpy()
                fig,ax = plt.subplots()
                ax.imshow(img,**DLT_KW_IMAGE)
                tb.gridDef_plot_2d(deformation,ax=ax,
                           step=10,
                           # color='#FFC759',
                           color='#E5BB5F',
                           linewidth=3
                           )
                image_list_for_gif.append(fig_to_image(fig,ax))
            plt.close(fig)

        elif ('image' in object or 'I'in object) and 'quiver' in object:
            image_list_for_gif = []
            for n in range(self.n_step):
                deformation = self.get_deformation(n).cpu()
                if n != 0:
                    deformation -= self.id_grid.cpu()
                img = self.image_stock[n,0].cpu().numpy()
                fig,ax = plt.subplots()
                ax.imshow(img,**DLT_KW_IMAGE)
                tb.quiver_plot(deformation,ax=ax,
                           step=10,
                           color='#E5BB5F',
                           )
                image_list_for_gif.append(fig_to_image(fig,ax))
            image_kw = dict()
            plt.close(fig)
        elif 'image' in object or 'I' in object:
            image_list_for_gif = [I[0].numpy() for I in self.image_stock]
            image_kw = DLT_KW_IMAGE
        elif 'residual' in object or "z" in object:
            image_list_for_gif = [z[0].numpy() for z in self.momentum_stock]
            #image_kw = DLT_KW_RESIDUALS
            image_kw = dict(cmap='RdYlBu_r', origin='lower',
                            vmin=self.momentum_stock.min(), vmax=self.momentum_stock.max())
        elif 'deformation' in object:
            image_list_for_gif = []
            for n in range(self.n_step):
                deformation = self.get_deformation(n).cpu()
                if n == 0:
                    deformation += self.id_grid.cpu()
                fig,ax = plt.subplots()
                tb.gridDef_plot_2d(deformation,ax=ax,
                           step=10,
                           color='black',
                           # color='#E5BB5F',
                           linewidth=5
                           )
                image_list_for_gif.append(fig_to_image(fig,ax))
            image_kw = dict()
            plt.close(fig)
        elif 'quiver' in object:
            image_list_for_gif = []
            for n in range(self.n_step):
                deformation = self.get_deformation(n).cpu()
                if n != 0:
                    deformation -= self.id_grid.cpu()
                fig,ax = plt.subplots()
                tb.quiver_plot(deformation,ax=ax,
                           step=10,
                           color='black',
                           )
                image_list_for_gif.append(fig_to_image(fig,ax))
            image_kw = dict()
            plt.close(fig)
        else:
            raise ValueError("object must be a string containing at least"
                             "one of the following : `image`,`residual`,`deformation`.")

        path,im = save_gif_with_plt(image_list_for_gif,file_name,folder,
                                    duplicate=True,image_args=image_kw,verbose=True,
                                    delay=delay,
                                    clean=clean)
        return path,im

class Optimize_geodesicShooting(torch.nn.Module,ABC):
    """ Abstract method for geodesic shooting optimisation. It needs to be provided with an object
    inheriting from Geodesic_integrator
    """
    @abstractmethod
    def __init__(self,source : torch.Tensor,
                     target : torch.Tensor,
                     geodesic : Geodesic_integrator,
                     cost_cst,
                     data_term = None,
                     optimizer_method : str = 'grad_descent'
                 ):
        """

        Important note to potential forks : all childs of this method
        must have the same __init__ method for proper loading.
        :param source:
        :param target:
        :param geodesic:
        :param cost_cst:
        :param optimizer_method:
        """
        super().__init__()
        self.mp = geodesic
        self.source = source
        self.target = target
        if isinstance(self.mp.sigma_v, tuple) and len(self.mp.sigma_v) != len(source.shape[2:]) :
            raise ValueError(f"Geodesic integrator :{self.mp.__class__.__name__}"
                             f"was initialised to be {len(self.mp.sigma_v)}D"
                             f" with sigma_v = {self.mp.sigma_v} and got image "
                             f"source.size() = {source.shape}"
                             )

        self.cost_cst = cost_cst
        # optimize on the cost as defined in the 2021 paper.
        self._cost_saving_ = self._default_cost_saving_

        self.optimizer_method_name = optimizer_method #for __repr__
        # forward function choice among developed optimizers
        if optimizer_method == 'grad_descent':
            self._initialize_optimizer_ = self._initialize_grad_descent_
            self._step_optimizer_ = self._step_grad_descent_
        elif optimizer_method == 'LBFGS_torch':
            self._initialize_optimizer_ = self._initialize_LBFGS_
            self._step_optimizer_ = self._step_LBFGS_
        elif optimizer_method == 'adadelta':
            self._initialize_optimizer_ = self._initialize_adadelta_
            self._step_optimizer_ = self._step_adadelta_
        else:
            raise ValueError(
                "\noptimizer_method is " + optimizer_method +
                "You have to specify the optimizer_method used among"
                "{'grad_descent', 'LBFGS_torch','adadelta'}"
                             )

        self.data_term = dt.Ssd(self.target) if data_term is None else data_term
        self.data_term.set_optimizer(self)
        # print("data_term : ",self.data_term)
        # self.temporal_integrator = vff.FieldIntegrator(method='temporal',save=False)
        self.is_DICE_cmp = False # Is dice alredy computed ?
        self._plot_forward_ = self._plot_forward_dlt_

        # # Default parameters to save (write to file)
        # self.field_to_save = FIELD_TO_SAVE
    # @abstractmethod
    # def _compute_V_norm_(self,*args):
    #     pass
    def _compute_V_norm_(self,*args):
        """

        usage 1: _compute_V_norm_(field)
            :field: torch Tensor of shape [1,H,W,2] or [1,D,H,W,3]
        usage 2: _compute_V_norm_(residual, image)
            :residual: torch Tensor of shape [1,C,H,W] or [1,C,D,H,W]
            :image: torch Tensor of shape [1,C,H,W] or [1,C,D,H,W]
        :return: float
        """

        # Computes only
        if len(args) == 2 and not args[0].shape[-1] in [2,3] :
            residual, image = args[0],args[1]
            C = residual.shape[1]
            grad_source = tb.spacialGradient(image)
            grad_source_resi = (grad_source * residual.unsqueeze(2)).sum(dim=1) #/ C
            K_grad_source_resi = self.mp.kernelOperator(grad_source_resi)

            return (grad_source_resi * K_grad_source_resi).sum()
        elif len(args) == 1 and args[0].shape[-1] in [2,3]:
            print("\n Warning !!!!  make sure that kernelOrerator is self adjoint.")
            raise ValueError("This method to compute the V norm is wrong"
                             " with gaussian Kernels")
            field = args[0]
            k_field = self.mp.kernelOperator(field)
            return (k_field * field).sum()
        else:
            raise ValueError(f"Bad arguments, see usage in Doc got args = {args}")


    @abstractmethod
    def cost(self,residual_ini):
        pass


    @abstractmethod
    def _get_mu_(self):
        pass

    @abstractmethod
    def _get_rho_(self):
        pass

    def get_geodesic_distance(self,only_zero = False):
        if only_zero:
            return float(self._compute_V_norm_(
                self.to_analyse[0],
                self.source
            ))
        else:
            dist = float(self._compute_V_norm_(
                    self.mp.momentum_stock[0][None],
                    self.mp.source
                ))
            for t in range(self.mp.momentum_stock.shape[0] - 1):
                dist += float(self._compute_V_norm_(
                    self.mp.momentum_stock[t + 1][None],
                    self.mp.image_stock[t][None]
                ))
            return dist

    def __repr__(self) -> str:
        return self.__class__.__name__ +\
            '(cost_parameters : {\n\t\tmu ='+ str(self._get_mu_())     +\
            ', \n\t\trho ='+str(self._get_rho_()) +\
            ', \n\t\tlambda =' + str(self.cost_cst)+'\n\t},'+\
            f'\n\tgeodesic integrator : '+ self.mp.__repr__()+\
            f'\n\tintegration method : '+      self.mp.step.__name__ +\
            f'\n\toptimisation method : '+ self.optimizer_method_name+\
            f'\n\t# geodesic steps =' +      str(self.mp.n_step) + '\n)'

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #   Implemented OPTIMIZERS
    # GRADIENT DESCENT
    def _initialize_grad_descent_(self,dt_step,max_iter=20):
        self.optimizer = GradientDescent(self.cost,self.parameter,lr=dt_step)

    def _step_grad_descent_(self):
        self.optimizer.step(verbose=False)



    # LBFGS
    def _initialize_LBFGS_(self,dt_step,max_iter = 20):
        self.optimizer = torch.optim.LBFGS([self.parameter],
                                           max_eval=15,
                                           max_iter=max_iter,
                                           lr=dt_step)

        def closure():
            self.optimizer.zero_grad()
            L = self.cost(self.parameter)
            #save best cms
            # if(self._it_count >1 and L < self._loss_stock[:self._it_count].min()):
            #     cms_tosave.data = self.cms_ini.detach().data
            L.backward()
            return L
        self.closure = closure

    def _step_LBFGS_(self):
        self.optimizer.step(self.closure)

    def _initialize_adadelta_(self,dt_step,max_iter=None):
        self.optimizer = torch.optim.Adadelta([self.parameter],
                                              lr=dt_step,
                                              rho=0.9,
                                              weight_decay=0)

        def closure():
            self.optimizer.zero_grad()
            L = self.cost(self.parameter)
            #save best cms
            # if(self._it_count >1 and L < self._loss_stock[:self._it_count].min()):
            #     cms_tosave.data = self.cms_ini.detach().data
            L.backward()
            return L
        self.closure = closure

    def _step_adadelta_(self):
        self.optimizer.step(self.closure)
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def _default_cost_saving_(self,i,loss_stock):
        """

        :param i: index for saving the according values
                !!! if `loss_stock` is None, `loss_stock` will be initialized, and
                `i` must have the value of the number of iterations.
        :param loss_stock:
        :return: updated `loss_stock`
        """
        #initialise loss_stock
        if loss_stock is None:
            d = 3 if self._get_mu_() != 0 else 2
            return torch.zeros((i,d))
        if self._get_mu_() != 0: # metamophosis
            loss_stock[i,0] = self.data_loss.detach()
            loss_stock[i,1] = self.norm_v_2.detach()
            loss_stock[i,2] = self.norm_l2_on_z.detach()
        else: #LDDMM
            loss_stock[i,0] = self.data_loss.detach()
            loss_stock[i,1] = self.norm_v_2.detach()
        return loss_stock

    def _plot_forward_dlt_(self):
        plt.figure()
        plt.imshow(self.mp.image[0,0].detach().cpu(),**DLT_KW_IMAGE)
        plt.show()

    @time_it
    def forward(self,
                z_0,
                n_iter = 10,
                grad_coef = 1e-3,
                verbose= True,
                plot=False,
                sharp=None
                ):
        r""" The function is and perform the optimisation with the desired method.
        The result is stored in the tuple self.to_analyse with two elements. First element is the optimized
        initial residual ($z_O$ in the article) used for the shooting.
        The second is a tensor with the values of the loss norms over time. The function
        plot_cost() is designed to show them automatically.

        :param z_0: initial residual. It is the variable on which we optimize.
        `require_grad` must be set to True.
        :param n_iter: (int) number of optimizer iterations
        :param verbose: (bool) display advancement

        """
        self.source = self.source.to(z_0.device)
        self.target = self.target.to(z_0.device)
        # self.mp.kernelOperator.kernel = self.mp.kernelOperator.kernel.to(z_0.device)
        self.data_term.to_device(z_0.device)

        self.parameter = z_0 # optimized variable
        self._initialize_optimizer_(grad_coef,max_iter=n_iter)

        self.id_grid = tb.make_regular_grid(z_0.shape[2:],z_0.device)
        if self.id_grid is None:
            raise ValueError(f"The initial momentum provided might have the wrong shape, got :{z_0.shape}")

        self.cost(self.parameter)

        loss_stock = self._cost_saving_(n_iter,None) # initialisation
        loss_stock = self._cost_saving_(0,loss_stock)

        for i in range(1,n_iter):
            # print("\n",i,"==========================")
            self._step_optimizer_()
            loss_stock = self._cost_saving_(i,loss_stock)

            if verbose:
                update_progress((i+1)/n_iter,message=('ssd : ',loss_stock[i,0]))
            if plot and i in [n_iter//4,n_iter//2,3*n_iter//4]:
                self._plot_forward_()


        # for future plots compute shooting with save = True
        self.mp.forward(self.source.clone(),
                        self.parameter.detach().clone(),
                        save=True,
                        plot=0)

        self.to_analyse = (self.parameter.detach(),loss_stock)
        self.to_device('cpu')

    def to_device(self,device):
        # self.mp.kernelOperator.kernel = self.mp.kernelOperator.kernel.to(device)
        self.source = self.source.to(device)
        self.target = self.target.to(device)
        self.parameter = self.parameter.to(device)
        self.id_grid = self.id_grid.to(device)
        self.data_term.to_device(device)
        self.to_analyse = (self.to_analyse[0].to(device),
                           self.to_analyse[1].to(device))

    def forward_safe_mode(self,
                          z_0,
                          n_iter = 10,
                          grad_coef = 1e-3,
                          verbose= True,
                          mode=None
                          ):
        """ Same as Optimize_geodesicShooting.forward(...) but
        does not stop the program when the integration diverges.
        If mode is not None, it tries to change the parameter
        until convergence as described in ```mode```

        :param z_0: initial residual. It is the variable on which we optimize.
        `require_grad` must be set to True.
        :param n_iter: (int) number of optimizer iterations
        :param verbose: (bool) display advancement
        :param mode:
            `'grad_coef'` this mode will decrease the grad_coef by
            dividing it by 10.
        :return:
        """
        try:
            self.forward(z_0,n_iter,grad_coef,verbose=verbose)
        except OverflowError:
            if mode is None:
                print("Integration diverged : Stop.\n\n")
                self.to_analyse= 'Integration diverged'
            elif mode == "grad_coef":
                print(f"Integration diverged :"
                      f" set grad_coef to {grad_coef*0.1}")
                self.forward_safe_mode(z_0,n_iter,grad_coef*0.1,verbose,mode=mode)

    def compute_landmark_dist(self,source_landmark,target_landmark=None,forward=True,verbose=True):
        # from scipy.interpolate import interpn
        # import numpy as np
        # compute deformed landmarks
        if forward:
            deformation = self.mp.get_deformation()
        else: deformation = self.mp.get_deformator()
        deform_landmark = []
        for l in source_landmark:
            idx =  (0,) + tuple([int(j) for j in l.flip(0)])
            deform_landmark.append(deformation[idx].tolist())

        def land_dist(land_1,land_2):
            print(f"land type : {land_1.dtype}, {land_2.dtype}")
            try: # caused by applying .round() on an int tensor
                return (
                    (land_1 - land_2.round()).abs()
                ).sum()/source_landmark.shape[0]
            except RuntimeError:
                return (
                    (land_1 - land_2).abs()
                ).sum()/source_landmark.shape[0]


        self.source_landmark = source_landmark
        self.target_landmark = target_landmark
        self.deform_landmark = torch.Tensor(deform_landmark)
        if target_landmark is None:
            return self.deform_landmark
        self.landmark_dist = land_dist(target_landmark,self.deform_landmark)
        dist_source_target = land_dist(target_landmark,source_landmark)
        print(f"Landmarks:\n\tBefore : {dist_source_target}\n\tAfter : {self.landmark_dist}")
        return self.deform_landmark, self.landmark_dist,dist_source_target

    def get_landmark_dist(self):
        try:
            return float(self.landmark_dist)
        except AttributeError:
            return 'not computed'

    def compute_DICE(self,source_segmentation,target_segmentation,plot=False,forward=True):
        """ Compute the DICE score of a regristration. Given the segmentations of
        a structure  (ex: ventricules) that should be present in both source and target image.
        it gives a score close to one if the segmentations are well matching after transformation.


        :param source_segmentation: Tensor of source size?
        :param target_segmentation:
        :return: (float) DICE score.
        """
        # TODO : make the computation
        self.is_DICE_cmp = True
        deformator = self.mp.get_deformator() if forward else self.mp.get_deformation()
        device = source_segmentation.device
        if len(source_segmentation.shape) == 2 or (len(source_segmentation.shape))==3:
            source_segmentation = source_segmentation[None,None]
        source_deformed = tb.imgDeform(source_segmentation,deformator.to(device),
                                       dx_convention='pixel')
        # source_deformed[source_deformed>1e-2] =1
        # prod_seg = source_deformed * target_segmentation
        # sum_seg = source_deformed + target_segmentation
        #
        # self.dice = 2*prod_seg.sum() / sum_seg.sum()
        self.dice = tb.dice(source_deformed,target_segmentation)
        if plot:
            fig,ax = plt.subplots()
            ax.imshow(tb.imCmp(target_segmentation,source_deformed))
            plt.show()
        return self.dice, source_deformed

    def get_DICE(self):
        # if self.is_DICE_cmp :
        #     return self.DICE
        # else:
        try:
            return self.dice
        except AttributeError:
            return 'not computed'


    def get_ssd_def(self):
        image_def = tb.imgDeform(self.source,self.mp.get_deformator(),dx_convention='pixel')
        return float(cf.SumSquaredDifference(self.target)(image_def))

    def save(self,source_name,target_name,message=None,destination=None,file=None):
        """ Save an optimisation to be later loaded and write all sort of info
        in a csv file

        :param source_name: (str) will appear in the file name
        :param target_name: (str) will appear in the file name
        :param message: (str) will appear in the csv storing all data
        :param destination: path of the folder to store the csvfile overview
        :param file:
        :return:
        """

        if self.to_analyse == 'Integration diverged':
            print("Can't save optimisation that didn't converged")
            return 0
        self.to_device('cpu')
        path =ROOT_DIRECTORY+'/my_metamorphosis/saved_optim/'
        date_time = datetime.now()
        if type(self.mp.sigma_v) is list: n_dim = '2D' if len(self.mp.sigma_v[0]) ==2 else '3D'
        else: n_dim = '2D' if len(self.mp.sigma_v) ==2 else '3D'
        id_num = 0

        # flag for discriminating against different kinds of Optimizers
        try:
            isinstance(self.mp.rf,Residual_norm_function)
            modifier_str = self.mp.rf.__repr__()
        except AttributeError:
            modifier_str = 'None'

        if 'afrancois' in ROOT_DIRECTORY:loc = '_atipnp'
        elif 'anfrancois' in ROOT_DIRECTORY: loc = '_attelecom'
        elif 'fanton' in ROOT_DIRECTORY: loc = '_atbar'
        else: loc = ''
        # build file name
        def file_name_maker_(id_num):
            return n_dim+date_time.strftime("_%d_%m_%Y")+'_'+\
                    source_name+'_to_'+target_name+loc+'_{:03d}'.format(id_num)+\
                    '.pk1'
        file_name = file_name_maker_(id_num)
        while file_name in os.listdir(path):
            id_num+=1
            file_name = file_name_maker_(id_num)


        state_dict = fill_saves_overview._optim_to_state_dict_(self,file_name,
                dict(
                    time = date_time.strftime("%d/%m/%Y %H:%M:%S"),
                    saved_file_name='', # Petit hack pour me simplifier la vie.
                    source = source_name,
                    target = target_name,
                    n_dim = n_dim
                ),
                message=message
        )
        fill_saves_overview._write_dict_to_csv(state_dict,path=destination,csv_file=file)


        #=================
        # save the data
        # print(self.data_term)
        # copy and clean dictonary containing all values
        dict_copy = {}
        # print("\n SAVING :")
        # print(self.__dict__.keys())
        for k in FIELD_TO_SAVE:
            # print(k,' >> ',self.__dict__.get(k))
            dict_copy[k] = self.__dict__.get(k)
            # ic(dict_copy[k])
            if torch.is_tensor(dict_copy[k]):
                dict_copy[k] = dict_copy[k].cpu().detach()
        dict_copy['mp'] = self.mp # For some reason 'mp' wasn't showing in __dict__

        # dict_copy['data_term'] = self.data_term
        if type(self.data_term) != Ssd:
            print('\nBUG WARNING : An other data term than Ssd was detected'
                "For now our method can't save it, it is ok to visualise"
                "the optimisation, but be careful loading the optimisation.\n")
        # save landmarks if they exist
        try:
            dict_copy['landmarks'] = (self.source_landmark,self.target_landmark,self.deform_landmark)
        except AttributeError:
            print('No landmark detected')
            pass

        # print(dict_copy.keys())
        with open(path+file_name,'wb') as f:
            pickle.dump(dict_copy,f,pickle.HIGHEST_PROTOCOL)
        print('Optimisation saved in '+path+file_name+'\n')
        if 'afrancois' in ROOT_DIRECTORY:
            print(f"To get file : sh shell/pull_from_server.sh -vl ipnp -f {file_name}")
        elif loc == '_attelecom':
            print(f"To get file : sh shell/pull_from_server.sh -vl gpu3 -f {file_name}")

        return file_name,path



    # ==================================================================
    #                 PLOTS
    # ==================================================================

    def get_total_cost(self):
        total_cost = self.to_analyse[1][:,0] + \
                    self.cost_cst * self.to_analyse[1][:,1]
        if self._get_mu_() != 0 :
            ic(type(self._get_rho_()))
            if type(self._get_rho_()) == float:
                total_cost += self.cost_cst*(self._get_rho_())* self.to_analyse[1][:,2]
            elif type(self._get_rho_()) == tuple:
                total_cost += self.cost_cst*(self._get_rho_()[0])* self.to_analyse[1][:,2]
        return total_cost

    def plot_cost(self):
        """ To display the evolution of cost during the optimisation.


        """
        fig1,ax1 = plt.subplots(1,2,figsize=(10,5))

        ssd_plot = self.to_analyse[1][:,0].numpy()
        ax1[0].plot(ssd_plot,"--",color = 'blue',label='ssd')
        ax1[1].plot(ssd_plot,"--",color = 'blue',label='ssd')

        normv_plot = self.cost_cst*self.to_analyse[1][:,1].detach().numpy()
        ax1[0].plot(normv_plot,"--",color = 'green',label='normv')
        ax1[1].plot(self.to_analyse[1][:,1].detach().numpy(),"--",color = 'green',label='normv')
        total_cost = ssd_plot +normv_plot
        if self._get_mu_() != 0:
            norm_l2_on_z = self.cost_cst*(self._get_rho_())* self.to_analyse[1][:,2].numpy()
            total_cost += norm_l2_on_z
            ax1[0].plot(norm_l2_on_z,"--",color = 'orange',label='norm_l2_on_z')
            ax1[1].plot(self.to_analyse[1][:,2].numpy(),"--",color = 'orange',label='norm_l2_on_z')

        ax1[0].plot(total_cost, color='black',label=r'$\Sigma$')
        ax1[0].legend()
        ax1[1].legend()
        ax1[0].set_title("Lambda = "+str(self.cost_cst)+
                    " mu = "+str(self._get_mu_()) +
                    " rho = "+str(self._get_rho_()))

    def plot_imgCmp(self):
        r""" Display and compare the deformed image $I_1$ with the target$
        """
        fig,ax = plt.subplots(2,2,figsize = (20,20),constrained_layout=True)
        image_kw = dict(cmap='gray',origin='lower',vmin=0,vmax=1)
        set_ticks_off(ax)
        ax[0,0].imshow(self.source[0,0,:,:].detach().cpu().numpy(),
                       **image_kw)
        ax[0,0].set_title("source",fontsize = 25)
        ax[0,1].imshow(self.target[0,0,:,:].detach().cpu().numpy(),
                       **image_kw)
        ax[0,1].set_title("target",fontsize = 25)

        ax[1,1].imshow(tb.imCmp(self.target,self.mp.image.detach().cpu(),method='compose'),**image_kw)
        ax[1,1].set_title("comparaison deformed image with target",fontsize = 25)
        ax[1,0].imshow(self.mp.image[0,0].detach().cpu().numpy(),**image_kw)
        ax[1,0].set_title("Integrated source image",fontsize = 25)
        tb.quiver_plot(self.mp.get_deformation()- self.id_grid,
                       ax=ax[1,1],step=15,color=GRIDDEF_YELLOW,
                       )

        text_param = f"mu = {self._get_mu_()}, rho = {self.mp._get_rho_()},"
        try:
            text_param += f" gamma = {self.mp._get_gamma_()}"
        except AttributeError:
            pass
        ax[1,1].text(10,self.source.shape[2] - 10,text_param ,c="white",size=25)

        text_score = ""
        if type(self.get_DICE()) is float:
            text_score += f"dice : {self.get_DICE():.2f},"

        if type(self.get_landmark_dist()) is float:
            ax[1,1].plot(self.source_landmark[:,0],self.source_landmark[:,1],**source_ldmk_kw)
            ax[1,1].plot(self.target_landmark[:,0],self.target_landmark[:,1],**target_ldmk_kw)
            ax[1,1].plot(self.deform_landmark[:,0],self.deform_landmark[:,1],**deform_ldmk_kw)
            ax[1,1].quiver(self.source_landmark[:,0],self.source_landmark[:,1],
                           self.deform_landmark[:,0]-self.source_landmark[:,0],
                           self.deform_landmark[:,1]-self.source_landmark[:,1],
                             color= "#2E8DFA")
            ax[1,1].legend()
            text_score += f"landmark : {self.get_landmark_dist():.2f},"
        ax[1,1].text(10,10,text_score,c='white',size=25)


        return fig,ax

    def plot_deform(self,temporal_nfigs = 0):
        residuals = self.to_analyse[0]
        #print(residuals.device,self.source.device)
        self.mp.forward(self.source.clone(),residuals,save=True,plot=0)
        self.mp.plot_deform(self.target,temporal_nfigs)

    def plot(self):
        self.plot_cost()
        self.plot_imgCmp()
        # self.plot_deform()







def load_optimize_geodesicShooting(file_name,path=None,verbose=True):
    """ load previously saved optimisation in order to plot it later."""

    # import pickle
    import io

    class CPU_Unpickler(pickle.Unpickler):
        """usage :
            #contents = pickle.load(f) becomes...
            contents = CPU_Unpickler(f).load()
        """
        def find_class(self, module, name):
            # print(f"Unpickler DEBUG : module:{module}, name:{name}")
            if module == 'torch.storage' and name == '_load_from_bytes':
                return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
            else:
                if module == 'metamorphosis': module = 'my_metamorphosis.metamorphosis'
                if name == 'metamorphosis_path': name = 'Metamorphosis_path'
                if name == 'multi_scale_GaussianRKHS': name = 'Multi_scale_GaussianRKHS'
                # print('module :',module,' name : ', name)
                return super().find_class(module, name)

    if path is None:
        path =ROOT_DIRECTORY+OPTIM_SAVE_DIR
    if not file_name in os.listdir(path):
        raise FileNotFoundError("File "+file_name+" does not exist in "+path)
    with open(path+file_name,'rb') as f:
        # opti_dict = pickle.load(f)
        opti_dict = CPU_Unpickler(f).load()


    # ic(opti_dict.keys())
    # ic(opti_dict['mp']._get_mu_())
    # ic(opti_dict['optimizer_method_name'])
    # SELECT THE RIGHT METAMORPHOSIS OPTIMISER
    # TODO : Modifier ça optimize_geo est une classe abstraite maintenant.
    # TODO : prevoir de pouvoir ouvrir multimodal optim
    # prevoir de decider de quelle class on parle en fonction de "F".
    # optimizer = Multimodal_Optim
    # print('Multimodal Metamorphosis loaded :' )
    flag_JM = False
    try:   # CONSTRAINTED METAMORPHOSIS
        isinstance(opti_dict['mp'].rf,Residual_norm_function)
        optimizer = Constrained_Optim
        if verbose: print('Constrained > Weighted')

    except AttributeError:
        # ic("the try failed")
    #     ic(opti_dict.keys())
    #     pass
        if type(opti_dict['mp']._get_mu_()) == tuple: # WEIGHTED JOINED MASK METAMORPHOSIS
            optimizer = Optimize_weighted_joinedMask    # Utiliser le masque sauvé pour detecter si c'est un joined mask
            flag_JM = True
            if verbose: print('Weighted joined mask Metamorphosis loaded :' )

        else: # CLASSIC METAMORPHOSIS
            optimizer = Optimize_metamorphosis
            if verbose: print('Classic Metamorphosis loaded :' )

    # ic(flag_JM)
    # ic(opti_dict['source'].shape)
    if verbose: print('DT:',opti_dict['data_term'])
    if flag_JM:
        new_optim = optimizer(opti_dict['source'][:,0][None],
                              opti_dict['target'][:,0][None],
                              opti_dict['source'][:,1][None],
                              opti_dict['target'][:,1][None],
                              opti_dict['mp'],
                              cost_cst=opti_dict['cost_cst'],
                              data_term=opti_dict['data_term'],
                              optimizer_method=opti_dict['optimizer_method_name'])

    else:
        new_optim = optimizer(opti_dict['source'],
                            opti_dict['target'],
                            opti_dict['mp'],
                            cost_cst=opti_dict['cost_cst'],
                            optimizer_method=opti_dict['optimizer_method_name'])
    for k in FIELD_TO_SAVE[5:]:
        try:
            new_optim.__dict__[k] = opti_dict[k]
        except KeyError:
            print("old fashioned Metamorphosis : No data_term, default is Ssd")
            pass
    # print('\n OPTI DICT',opti_dict.keys())
    if 'landmarks' in opti_dict.keys():
        new_optim.compute_landmark_dist(opti_dict['landmarks'][0],opti_dict['landmarks'][1])
        # new_optim.source_landmark =
        # new_optim.target_landmark =
        # new_optim.deform_landmark = opti_dict['landmarks'][2]

    ic(new_optim.__class__.__name__)
    new_optim.loaded_from_file = file_name
    if verbose: print(f'New optimiser loaded ({file_name}) :\n',new_optim.__repr__())
    return new_optim
