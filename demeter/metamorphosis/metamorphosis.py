import torch
import matplotlib.pyplot as plt
import warnings
from math import prod
import pickle
import os, sys, csv#, time
# sys.path.append(os.path.abspath('../'))
from datetime import datetime
from abc import ABC, abstractmethod

import my_torchbox as tb
import vector_field_to_flow as vff
from my_toolbox import update_progress,format_time, get_size, fig_to_image,save_gif_with_plt
from my_optim import GradientDescent
import reproducing_kernels as rk
import cost_functions as cf
from constants import *
from decorators import deprecated,time_it
import fill_saves_overview as fill_saves_overview

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
            # print(f"\nTUPLE {sigma_v}")
            self.sigma_v = sigma_v# is used if the optmisation is later saved
            self.kernelOperator = rk.GaussianRKHS(sigma_v,
                                                     border_type='constant')
        elif isinstance(sigma_v,list):
            # print("LIST")
            self.sigma_v = sigma_v
            if multiScale_average:
                self.kernelOperator = rk.multi_scale_GaussianRKHS_notAverage(sigma_v)
            else:
                self.kernelOperator = rk.multi_scale_GaussianRKHS(sigma_v)
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

    def forward(self,image,residual_ini,
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
        :param residual_ini: (tensor array) of shape [H,W]. initial residual ($z_0$)
        :param save: (bool) option to save the integration intermediary steps.
        :param plot: (int) positive int lower than `self.n_step` to plot the indicated
                         number of intemediary steps. (if plot>0, save is set to True)

        """
        if len(residual_ini.shape) not in [4,5]:
            raise ValueError(f"residual_ini must be of shape [B,C,H,W] or [B,C,D,H,W] got {residual_ini.shape}")
        device = residual_ini.device
        # print(f'sharp = {sharp} flag_sharp : {self.flag_sharp},{self._phis}')
        self._init_sharp_(sharp)
        self.source = image.detach()
        self.image = image.clone().to(device)
        self.residuals = residual_ini
        self.debug = debug
        try:
            self.save = True if self._force_save else save
        except AttributeError:
            self.save = save

        self.id_grid = tb.make_regular_grid(residual_ini.shape[2:],device=device)
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
            self.residuals_stock = torch.zeros((t_max*self.n_step,)+residual_ini.shape[1:])


        for i,t in enumerate(torch.linspace(0,t_max,t_max*self.n_step)):
            self._i = i

            _,field_to_stock,residuals_dt = self.step()

            if self.image.isnan().any() or self.residuals.isnan().any():
                raise OverflowError("Some nan where produced ! the integration diverged",
                                    "changing the parameters is needed (increasing n_step can help) ")

            if self.save:
                self.image_stock[i] = self.image[0].detach().to('cpu')
                self.field_stock[i] = field_to_stock[0].detach().to('cpu')
                self.residuals_stock[i] = residuals_dt.detach().to('cpu')

            if verbose:
                update_progress(i/(t_max*self.n_step))

        # try:
        #     _d_ = device if self._force_save else 'cpu'
        #     self.field_stock = self.field_stock.to(device)
        # except AttributeError: pass

        if plot>0:
            self.plot(n_figs=plot)

    def _image_Eulerian_integrator_(self,image,vector_field,t_max,n_step=None):
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

    def _compute_vectorField_(self,residuals,grad_image):
        r""" operate the equation $K \star (z_t \cdot \nabla I_t)$

        :param residuals: (tensor array) of shape [H,W] or [D,H,W]
        :param grad_image: (tensor array) of shape [B,C,2,H,W] or [B,C,3,D,H,W]
        :return: (tensor array) of shape [B,H,W,2]
        """
        C = residuals.shape[1]
        return tb.im2grid(self.kernelOperator((-(residuals.unsqueeze(2) * grad_image).sum(dim=1))))

    def _field_cst_mult(self):
        if self._get_mu_() == 0:
            return 1
        else:
            return self._get_rho_()/self._get_mu_()

    def _update_field_(self):
        grad_image = tb.spacialGradient(self.image,dx_convention='pixel')
        self.field = self._compute_vectorField_(self.residuals,grad_image)
        self.field *= self._field_cst_mult()

    def _update_residuals_Eulerian_(self):
        residuals_dt = - tb.Field_divergence(dx_convention='pixel')(
                            self.residuals[0,0][None,:,:,None]* self.field,
                            )

        self.residuals = self.residuals + residuals_dt /self.n_step

    def _update_residuals_semiLagrangian_(self,deformation):
        div_v_times_z = self.residuals* tb.Field_divergence(dx_convention='pixel')(self.field)[0,0]

        self.residuals = tb.imgDeform(self.residuals,
                                      deformation,
                                      dx_convention='pixel',
                                      clamp=False)\
                         - div_v_times_z/self.n_step

    def _compute_sharp_intermediary_residuals_(self):
        device = self.residuals.device
        resi_cumul = torch.zeros(self.residuals.shape,device=device)
        for k,phi in enumerate(self._phis[self._i][1:]):
            resi_cumul += tb.imgDeform(self.residuals_stock[k][None].to(device),
                                       phi,dx_convention='pixel')
        resi_cumul = resi_cumul + self.residuals
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
        self.image = self.image + (self.residuals *self.mu)/ self.n_step

    def _update_image_semiLagrangian_(self,deformation,residuals = None,sharp=False):
        if residuals is None: residuals = self.residuals
        image = self.source if sharp else self.image
        self.image = tb.imgDeform(image,deformation,dx_convention='pixel')
        if self._get_mu_() != 0: self.image += (residuals *self.mu)/self.n_step

    def _update_sharp_intermediary_field_(self):
        # print('update phi ',self._i,self._phis[self._i])
        self._phis[self._i][self._i] = self.id_grid - self.field/self.n_step
        if self._i > 0:
            for k,phi in enumerate(self._phis[self._i - 1]):
                # self._phis[self._i][k] = phi - tb.compose_fields(self.field/self.n_step,phi,'pixel')
                self._phis[self._i][k] = tb.compose_fields(phi,self._phis[self._i][self._i],
                                                           'pixel').to(self.field.device)


    def get_deformation(self,n_step=-1,save=False):
        r"""Returns the deformation use it for showing results
        $\Phi = \int_0^1 v_t dt$

        :return: deformation [1,H,W,2] or [2,H,W,D,3]
        """
        if n_step == 0:
            return self.field_stock[0][None]/self.n_step
        temporal_integrator = vff.FieldIntegrator(method='temporal',save=save)
        return temporal_integrator(self.field_stock[:n_step]/self.n_step,forward=True)

    def get_deformator(self,n_step=-1,save = False):
        r"""Returns the inverse deformation use it for deforming images
        $\Phi^{-1}$

        :return: deformation [T,H,W,2] or [T,H,W,D,3]
        """
        if n_step == 0:
            return self.id_grid-self.field_stock[0][None]/self.n_step
        temporal_integrator = vff.FieldIntegrator(method='temporal',save=save)
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
        v_abs_max = torch.quantile(self.residuals.abs(),.99)
        kw_residuals_args = dict(cmap='RdYlBu_r',
                      extent=[-1,1,-1,1],
                      origin='lower',
                      vmin=-v_abs_max,
                      vmax=v_abs_max)
        size_fig = 5
        C = self.residuals_stock.shape[1]
        plt.rcParams['figure.figsize'] = [size_fig*3,n_figs*size_fig]
        fig,ax = plt.subplots(n_figs,2 + C)

        for i,t in enumerate(plot_id):
            i_s =ax[i,0].imshow(self.image_stock[t,:,:,:].detach().permute(1,2,0).numpy(),
                                **kw_image_args)
            ax[i,0].set_title("t = "+str((t/(self.n_step-1)).item())[:3])
            ax[i,0].axis('off')
            fig.colorbar(i_s,ax=ax[i,0],fraction=0.046, pad=0.04)

            for j in range(C):
                r_s =ax[i,j+1].imshow(self.residuals_stock[t,j].detach().numpy(),
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

        plt.rcParams['figure.figsize'] = [20,30]
        fig , axes = plt.subplots(3,2)
        # show resulting deformation

        tb.gridDef_plot_2d(full_deformation,step=int(max(self.image.shape)/30),ax = axes[0,0],
                         check_diffeo=True,dx_convention='2square')
        tb.quiver_plot(full_deformation -self.id_grid.cpu() ,step=int(max(self.image.shape)/30),
                        ax = axes[0,1],check_diffeo=False)

        # show S deformed by full_deformation
        S_deformed = tb.imgDeform(self.source.cpu(),full_deformator,
                                  dx_convention='pixel')
        axes[1,0].imshow(self.source[0,:,:,:].cpu().permute(1,2,0),cmap='gray',origin='lower',vmin=0,vmax=1)
        axes[1,1].imshow(target[0].cpu().permute(1,2,0),cmap='gray',origin='lower',vmin=0,vmax=1)
        axes[2,0].imshow(S_deformed[0,:,:,:].permute(1,2,0),cmap='gray',origin='lower',vmin=0,vmax=1)
        axes[2,1].imshow(tb.imCmp(target,S_deformed),origin='lower',vmin=0,vmax=1)

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
                           linewidth=5
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
            image_list_for_gif = [z.numpy() for z in self.residuals_stock]
            #image_kw = DLT_KW_RESIDUALS
            image_kw = dict(cmap='RdYlBu_r',origin='lower',
                            vmin=self.residuals_stock.min(),vmax=self.residuals_stock.max())
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
        self.data_term = Ssd(self.target) if data_term is None else data_term
        self.data_term.set_optimizer(self)
        # print("data_term : ",self.data_term)

        # self.temporal_integrator = vff.FieldIntegrator(method='temporal',save=False)
        self.is_DICE_cmp = False

    def _compute_V_norm_(self,residual,image):
        """
        Compute the V norm <Lv,w> using the formula:
        psi = Lv; ||v||_V = <psi, v>_{L^2} = < psi, L^{-1}\star psi >_{L^2}
        see Section 2.2.4 of my thesis disseration for more details.
      
        usage : _compute_V_norm_(residual, image)
            :residual: torch Tensor of shape [1,C,H,W] or [1,C,D,H,W]
            :image: torch Tensor of shape [1,C,H,W] or [1,C,D,H,W]
        :return: float
        """
        
        C = residual.shape[1]
        grad_source = tb.spacialGradient(image)
        grad_source_resi = (grad_source * residual.unsqueeze(2)).sum(dim=1) / C
        K_grad_source_resi = self.mp.kernelOperator(grad_source_resi)

        return (grad_source_resi * K_grad_source_resi).sum()

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
            dist = 0
            for t in range(self.mp.residuals_stock.shape[0]):
                dist += float(self._compute_V_norm_(
                    self.mp.residuals_stock[t][None],
                    self.mp.image_stock[t][None]
                ))
            return dist

    def __repr__(self) -> str:
        return self.__class__.__name__ +\
            '(cost_parameters : {mu ='+ str(self._get_mu_())     +\
            ', rho ='+str(self._get_rho_()) +\
            ', lambda =' + str(self.cost_cst)+'},'+\
            f'\ngeodesic integrator : '+ self.mp.__repr__()+\
            f'\nintegration method : '+      self.mp.step.__name__ +\
            f'\noptimisation method : '+ self.optimizer_method_name+\
            f'\n# geodesic steps =' +      str(self.mp.n_step) + ')'

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
        assert self.id_grid != None

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
                plt.figure()
                plt.imshow(self.mp.image[0,0].detach().cpu(),**DLT_KW_IMAGE)
                plt.show()


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

    def compute_landmark_dist(self,source_landmark,target_landmark=None,forward=True):
        # from scipy.interpolate import interpn
        import numpy as np
        # compute deformed landmarks
        if forward:
            deformation = self.mp.get_deformation()
        else: deformation = self.mp.get_deformator()
        deform_landmark = []
        for l in source_landmark:
            idx =  (0,) + tuple([int(j) for j in l.flip(0)])
            deform_landmark.append(deformation[idx].tolist())

        def land_dist(land_1,land_2):
            return (
                (land_1 - land_2.round()).abs()
            ).sum()/source_landmark.shape[0]

        self.source_landmark = source_landmark
        self.target_landmark = target_landmark
        self.deform_landmark = torch.Tensor(deform_landmark)
        if target_landmark is None:
            return self.deform_landmark
        self.landmark_dist = land_dist(target_landmark,self.deform_landmark)
        dist_source_target = land_dist(target_landmark,source_landmark)
        return self.deform_landmark, self.landmark_dist,dist_source_target-self.landmark_dist

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
        prod_seg = source_deformed * target_segmentation
        sum_seg = source_deformed + target_segmentation

        self.dice = float(2 * prod_seg.sum() / sum_seg.sum())
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
        path =ROOT_DIRECTORY+'/saved_optim/'
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
            print(f"To get file : sh my_metamorphosis/pull_from_server.sh -vl ipnp -f {file_name}")
        elif loc == '_attelecom':
            print(f"To get file : sh my_metamorphosis/pull_from_server.sh -vl gpu3 -f {file_name}")

        return file_name,path



    # ==================================================================
    #                 PLOTS
    # ==================================================================

    def get_total_cost(self):
        total_cost = self.to_analyse[1][:,0] + \
                    self.cost_cst * self.to_analyse[1][:,1]
        if self._get_mu_() != 0 :
            total_cost += self.cost_cst*(self._get_rho_())* self.to_analyse[1][:,2]
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

        ax[1,1].imshow(tb.imCmp(self.target,self.mp.image.detach().cpu()),**image_kw)
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
        self.mp.forward(self.source.clone(),residuals,save=True,plot=0)
        self.mp.plot_deform(self.target,temporal_nfigs)

    def plot(self):
        self.plot_cost()
        self.plot_imgCmp()
        # self.plot_deform()

class Residual_norm_function(ABC):

    @abstractmethod
    def __init__(self,mask,mu = 1,rho=None):
        if mu <= 0:
            raise ValueError(f"mu must be a non zero real positive value, got mu = {mu:.3f}")
        self.mu = mu
        self.rho = mu if rho is None else rho
        self.mask = mask
        if self.mask.shape[0] > 1: self.n_step = self.mask.shape[0]
        # self.anti_mask = (mask < 1e-5)

        # preparation of border
        # binary_mask = 1 - self.anti_mask.to(torch.float)
        # # plt.figure()
        # # plt.imshow(tb.imCmp(binary_mask[12][None],self.mask[12][None]))
        # # plt.show()
        # normal_vector_grad = - tb.spacialGradient(binary_mask)
        # # normal_vector_grad[:,0,1] *= -1
        # self.unit_normal_vector_grad = torch.nn.functional.normalize(normal_vector_grad,dim=2,p=2)[:,0]

    def __repr__(self):
        return f'{self.__class__.__name__}:(mu = {self.mu:.2E},rho = {self.rho:.2E})'

    def to_device(self,device):
        self.mask = self.mask.to(device)
        # self.unit_normal_vector_grad = self.unit_normal_vector_grad.to(device)

    @abstractmethod
    def f(self,t):
        pass

    def F(self,t):
        # F_mat = self.f(t)[self.anti_mask]/(self.mu * self.mask[self.anti_mask])
        # F_mat[self.anti_mask] = 1
        return self.mask[t] * self.f(t) / self.mu

    def F_div_f(self,t):
        return self.mu*self.mask[t]

    @abstractmethod
    def dt_F(self,t):
        pass

    # def border_normal_vector(self,t):
    #     return tb.im2grid(self.unit_normal_vector_grad[t][None])


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

# =======================================================================
# =======================================================================
#  CLASSICAL METAMORPHOSIS
# =======================================================================
# =======================================================================


class Metamorphosis_path(Geodesic_integrator):
    """ Class integrating over a geodesic shooting. The user can choose the method among
    'Eulerian', 'advection_semiLagrangian' and 'semiLagrangian.

    """
    def __init__(self,method,
                 mu=1.,
                 rho=1.,
                 sigma_v= (1,1,1),
                 n_step =10,
                 multiScale_average=False
                 ):
        """

        :param mu: Control parameter for geodesic shooting intensities changes
        For more details see eq. 3 of the article
        mu = 0 is LDDMM
        mu > 0 is metamorphosis
        :param sigma_v: sigma of the gaussian RKHS in the V norm
        :param n_step: number of time step in the segment [0,1]
         over the geodesic integration
        """
        # print('Before super() my weight is ',get_size(self))
        super().__init__(sigma_v,multiScale_average)
        # self.mu = mu if callable(mu) else lambda :mu
        # self.rho = rho if callable(rho) else lambda :rho
        self.mu = mu
        self.rho = rho
        if mu < 0: self.mu = 0
        if(mu == 0 and rho != 0):
            warnings.warn("mu as been set to zero in methamorphosis_path, "
                          "automatic reset of rho to zero."
                         )
            self.rho = 0
        self.n_step = n_step

        # inner methods

        # self._force_save = False
        sharp = False
        if method == 'Eulerian':
            self.step = self._step_fullEulerian
        elif method == 'advection_semiLagrangian':
            self.step = self._step_advection_semiLagrangian
        elif method == 'semiLagrangian':
            self.step = self._step_full_semiLagrangian
        elif method == 'sharp':
            sharp = True
            self.step = self._step_sharp_semiLagrangian
        elif method== 'tkt':
            pass
        else:
            raise ValueError("You have to specify the method used : 'Eulerian','advection_semiLagrangian'"
                             " or 'semiLagrangian'")
        self._init_sharp_(sharp)

    def _step_fullEulerian(self):
        self._update_field_()
        self._update_image_Eulerian_()
        self._update_residuals_Eulerian_()

        return (self.image,self.field,self.residuals)

    def _step_advection_semiLagrangian(self):
        self._update_field_()
        # Lagrangian scheme on images
        deformation = self.id_grid - self.field/self.n_step
        self._update_image_semiLagrangian_(deformation)
        self._update_residuals_Eulerian_()

        return (self.image,self.field,self.residuals)

    def _step_full_semiLagrangian(self):
        self._update_field_()
        # Lagrangian scheme on images and residuals
        deformation = self.id_grid - self.field/self.n_step
        self._update_image_semiLagrangian_(deformation)
        self._update_residuals_semiLagrangian_(deformation)

        return (self.image,self.field,self.residuals)

    def _step_sharp_semiLagrangian(self):
        # device = self.image.device
        self._update_field_()
        self._update_sharp_intermediary_field_()

        resi_cumul = 0
        if self._get_mu_() != 0:
            resi_cumul = self._compute_sharp_intermediary_residuals_()
            resi_cumul = resi_cumul.to(self.residuals.device)
        if self._i > 0: self._phis[self._i - 1] = None

        # # DEBUG ============================
        if self.debug:
            fig,ax = plt.subplots(2,self._i+1)
            print(ax.shape)
            print('phi :',len(self._phis[self._i]))

            for k_y, phi in enumerate(self._phis[self._i]):
                ax_p = ax[0,k_y] if self._i != 0 else ax[0]
                tb.gridDef_plot_2d(phi.detach(),ax=ax_p,step=5)
                ax_p.set_title(f'phi [{self._i}][{k_y}]')

                ax_im = ax[1,k_y] if self._i != 0 else ax[1]
                ax_im.imshow(tb.imgDeform(self.source.detach(),phi.detach(),
                                          dx_convention='pixel')[0,0]
                             ,**DLT_KW_IMAGE)
            if self._i !=0:
                fig,ax = plt.subplots(1,2,figsize =(10,5))
                aa = ax[0].imshow(self.residuals.detach()[0,0],**DLT_KW_RESIDUALS)
                fig.colorbar(aa,ax=ax[0],fraction=0.046, pad=0.04)

                # self._update_residuals_semiLagrangian_(phi_n_n)
                ab = ax[1].imshow(resi_cumul.detach()[0,0],**DLT_KW_RESIDUALS)
                ax[1].set_title("new residual")
                fig.colorbar(ab,ax=ax[1],fraction=0.046, pad=0.04)

                #
            plt.show()
        # gaaa = input('Appuyez sur entree')
        # # End DEBUG ===================================
        self._update_image_semiLagrangian_(self._phis[self._i][0],resi_cumul,
                                           sharp=True)
        # self.image = tb.imgDeform(self.source.to(self.image.device),
        #                           self._phis[self._i][0],
        #                           dx_convention='pixel')
        # if self.mu != 0: self.image += self.mu * resi_cumul/self.n_step
        self._update_residuals_semiLagrangian_(self._phis[self._i][self._i])

        return (self.image, self.field, self.residuals)

    def step(self):
        raise ValueError("You have to specify the method used : 'Eulerian','advection_semiLagrangian'"
                             " or 'semiLagrangian'")

    def _get_mu_(self):
        return self.mu

    def _get_rho_(self):
        return float(self.rho)

class Optimize_metamorphosis(Optimize_geodesicShooting):

    def __init__(self,source : torch.Tensor,
                     target : torch.Tensor,
                     geodesic : Metamorphosis_path,
                     cost_cst : float,
                     data_term=None,
                     optimizer_method : str = 'grad_descent',
                     # sharp=False
                     # mask = None # For cost_function masking
                 ):
        super().__init__(source,target,geodesic,cost_cst,data_term,optimizer_method)
        # self.mask = mask

    # def __repr__(self) -> str:
    #     return self.__class__.__name__ +\
    #         '(cost_parameters : {mu ='+ str(self._get_mu_())     +\
    #         ', rho ='+str(self._get_rho_()) +\
    #         ', lambda =' + str(self.cost_cst)+'},'+\
    #         '\ngeodesic integrator : '+ self.mp.__repr__()+\
    #         '\nintegration method : '+      self.mp.step.__name__ +\
    #         '\noptimisation method : '+ self.optimizer_method_name+\
    #         '\n# geodesic steps =' +      str(self.mp.n_step) + ')'

    def _get_mu_(self):
        return self.mp.mu

    def _get_rho_(self):
        return float(self.mp.rho)

    def _compute_V_norm_(self,*args):
        """

        usage 1: _compute_V_norm_(field)
            :field: torch Tensor of shape [1,H,W,2] or [1,D,H,W,3]
        usage 2: _compute_V_norm_(residual, image)
            :residual: torch Tensor of shape [1,C,H,W] or [1,C,D,H,W]
            :image: torch Tensor of shape [1,C,H,W] or [1,C,D,H,W]
        :return: float
        """
        if len(args) == 2 and not args[0].shape[-1] in [2,3] :
            residual, image = args[0],args[1]
            C = residual.shape[1]
            grad_source = tb.spacialGradient(image)
            grad_source_resi = (grad_source * residual.unsqueeze(2)).sum(dim=1) / C
            K_grad_source_resi = self.mp.kernelOperator(grad_source_resi)

            return (grad_source_resi * K_grad_source_resi).sum()
        elif len(args) == 1 and args[0].shape[-1] in [2,3]:
            field = args[0]
            k_field = self.mp.kernelOperator(field)
            return (k_field * field).sum()
        else:
            raise ValueError(f"Bad arguments, see usage in Doc got args = {args}")


    def cost(self, residuals_ini : torch.Tensor) -> torch.Tensor:
        r""" cost computation

        $H(z_0) =   \frac 12\| \im{1} - \ti \|_{L_2}^2 + \lambda \Big[ \|v_0\|^2_V + \mu ^2 \|z_0\|^2_{L_2} \Big]$

        :param residuals_ini: z_0
        :return: $H(z_0)$ a single valued tensor
        """
        #_,_,H,W = self.source.shape
        rho = self.mp.rho
        lamb = self.cost_cst
        if(self.mp.mu == 0 and rho != 0):
            warnings.warn("mu as been set to zero in methamorphosis_path, "
                          "automatic reset of rho to zero."
                         )
            rho = 0
        self.mp.forward(self.source,residuals_ini,save=False,plot=0)

        # Compute the data_term. Default is the Ssd
        self.data_loss = self.data_term()

        # Norm V
        self.norm_v_2 = self._compute_V_norm_(residuals_ini,self.source)


        # norm_2 on z
        if self.mp.mu != 0:
            # # Norm on the residuals only
            self.norm_l2_on_z = (residuals_ini**2).sum()/prod(self.source.shape[2:])
            self.total_cost = self.data_loss + \
                              lamb * (self.norm_v_2 + (rho) *self.norm_l2_on_z)
        else:
             self.total_cost = self.data_loss + lamb * self.norm_v_2
        # print('ssd :',self.ssd,' norm_v :',self.norm_v_2)
        return self.total_cost


# =======================================================================
# =======================================================================
#  WEIGHTED METAMORPHOSIS
# =======================================================================
# =======================================================================

class Residual_norm_identity(Residual_norm_function):

    def __init__(self,mask,mu,rho):
        if rho < 0:
            raise ValueError(f"rho must be a real positive value got rho = {rho:.3f}")
        super(Residual_norm_identity, self).__init__(mask, mu,rho)


    def __repr__(self):
        return f'{self.__class__.__name__}:(mu = {self.mu:.2E},rho = {self.rho:.2E})' \
               f'F = rho/mu = {self.rho/self.mu}, f = rho).'

    def f(self,t):
        return self.rho

    # def F(self,t):
    #     return self.rho/self.mu
    #
    # def inv_F(self,t):
    #     return  self.mu/self.rho

    def dt_F(self,t):
        return 0

class Residual_norm_borderBoost(Residual_norm_function):

    def __init__(self,mask,mu,rho):
        raise NotImplementedError("Need to redo all of this ....")
        if rho < 0:
            raise ValueError(f"rho must be a real positive value got rho = {rho:.3f}")
        super().__init__(mask, mu,rho)
        n_step = mask.shape[0]

        # Preparation of the time derivative F(M_t)
        # TODO: le spacial gradient risque de ne pas fonctionner sur le batch en 3D
        grad_mask = tb.spacialGradient(self.seg_tumour)
        self.grad_mask_norm = (grad_mask**2).sum(dim = 2).sqrt()
        # TODO : make sure that this n_step is the same that the geodesic integrator will use
        grad_dt_mask = tb.spacialGradient((self.seg_tumour[1:] - self.seg_tumour[:-1]) / n_step)
        grad_mask_times_grad_dt_mask = (grad_mask[1:] * grad_dt_mask).sum(dim=2)
        self.dt_F_mask = torch.nan_to_num(grad_mask_times_grad_dt_mask / self.grad_mask_norm[1:])

    def __repr__(self):
        return f'{self.__class__.__name__}:(mu = {self.mu:.2E},rho = {self.rho:.2E})' \
               f' rho/mu = {self.rho/self.mu}, mu/rho = {self.mu/self.rho}.'

    def cpu(self):
        self.grad_mask_norm = self.grad_mask_norm.to('cpu')
        self.dt_F_mask = self.dt_F_mask.to('cpu')
        super(Residual_norm_borderBoost, self).cpu()

    def to_device(self,device):
        self.grad_mask_norm = self.grad_mask_norm.to(device)
        self.dt_F_mask = self.dt_F_mask.to(device)
        super(Residual_norm_borderBoost, self).to_device(device)
        # self.mask = self.mask.to(device)
        # self.unit_normal_vector_grad = self.unit_normal_vector_grad.to(device)

    def f(self,t):
        return self.mask[t,0]*(1 + self.rho * self.grad_mask_norm[t,0])

    def F(self,t):
        return (self.mu +  self.rho * self.grad_mask_norm[t,0])/self.mu

    def dt_F(self,t):
        try:
            return self.dt_F_mask[t,0]
        except:
            return 1

# ==========================================================================
#
#  Constrained optimisation
#
#===========================================================================


class Constrained_meta_path(Geodesic_integrator):

    def __init__(self, residual_function: Residual_norm_function = None,
                 orienting_field: torch.Tensor = None,
                 orienting_mask: torch.Tensor = None,
                 mu=0,
                 rho=0,
                 gamma=0,
                 sigma_v=(1, 1, 1),
                 n_step=None,
                 border_effect=True,
                 sharp = False
    ):
        super(Constrained_meta_path, self).__init__(sigma_v)
        if residual_function is None:
            self.flag_W, self.mu, self.rho = False, mu, rho
            print("not Weighted")
        else:
            print(n_step)
            if not n_step is None:
                if hasattr(residual_function, 'n_step') and \
                        (residual_function.n_step != n_step):
                    raise ValueError(f"{residual_function.__class__.__name__}.n_step is {residual_function.n_step}"
                                     f"and {self.__class__.__name__} = {n_step}. They must be equal.")
            print("Weighted")
            self.rf = residual_function
            try:
                self.rf.set_geodesic_integrator(self)
            except AttributeError:
                pass
            self.flag_W = True
        if gamma == 0 or orienting_field is None:
            print("not oriented")
            self.gamma, self.orienting_mask, self.orienting_field = 0, None, None
            self.flag_O = False
            if self.flag_W : self.n_step = self.rf.n_step
        else:
            print("oriented")
            self.flag_O = True
            self.orienting_mask = orienting_mask
            self.orienting_field = orienting_field
            self.gamma = gamma
            self.n_step = orienting_field.shape[0]

        if self.flag_O and self.flag_W:
            if orienting_field.shape[0] != self.rf.n_step or\
                    orienting_mask.shape[0] != self.rf.n_step:
                raise ValueError(f"orienting_field, orienting_mask and the mask of the residual_function"
                                 f"must have same size got (resp.):{orienting_field.shape} "
                                 f",{orienting_mask.shape},{self.rf.mask.shape}"
                                 )

        self._init_sharp_(sharp)

        # self.n_step = n_step
        self.rf = residual_function



    def __repr__(self):
        if self.flag_W and self.flag_O:
            mode = 'Weighted & Oriented'
        elif self.flag_W:
            mode = 'weighted'
        elif self.flag_O:
            mode = 'oriented'
        else:
            mode = 'why are you using this class ?'

        if self.flag_W:
            param = self.rf.__repr__()
        else:
            param = f"parameters :(mu = {self._get_mu_():.2E},rho = {self._get_rho_():.2E}"
        if self.flag_O: param += f"gamma = {self._get_gamma_():.2E}"
        param += ")"

        return self.__class__.__name__ + f"({mode})\n" \
                                         f"\t {param}\n\t{self.kernelOperator.__repr__()} "

    def _get_mu_(self):
        if self.flag_W:
            return self.rf.mu
        else:
            return self.mu

    def _get_rho_(self):
        if self.flag_W:
            return float(self.rf.rho)
        else:
            return float(self.rho)

    def _get_gamma_(self):
        if self.flag_O:
            return self.gamma
        else:
            return 0

    def _update_residuals_weighted_semiLagrangian_(self, deformation):
        f_t = self.rf.f(self._i)
        fz_times_div_v = f_t * self.residuals * tb.Field_divergence(dx_convention='pixel')(self.field)[0, 0]
        div_fzv = -tb.imgDeform(f_t * self.residuals,
                                deformation,
                                dx_convention='pixel',
                                clamp=False)[0, 0] \
                  + fz_times_div_v / self.n_step
        z_time_dtF = self.residuals * self.rf.dt_F(self._i)
        self.residuals = - (div_fzv + z_time_dtF) / f_t

    def _update_image_weighted_semiLagrangian_(self, deformation,residuals = None,sharp=False):
        if residuals is None: residuals = self.residuals
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
                self.kernelOperator((- self.rf.f(self._i) * self.residuals * grad_image[0]))
            )/self._get_mu_()
        else:
            free_field = self._compute_vectorField_(self.residuals, grad_image)
            free_field *= self._field_cst_mult()
        oriented_field = 0
        if self.flag_O:
            mask_i = self.orienting_mask[self._i][..., None].clone()
            free_field *= 1 / (1 + (self.gamma * mask_i))

            oriented_field = self.orienting_field[self._i][None].clone()
            oriented_field *= (self.gamma * mask_i) / (1 + (self.gamma * mask_i))

        self.field = free_field + oriented_field

    def step(self):
        if self.flag_O or self.flag_W:
            self._update_field_oriented_weighted_()
        else:
            self._update_field_()

        if self.flag_sharp:
            self._update_sharp_intermediary_field_()
            def_z = self._phis[self._i][self._i]
            def_I = self._phis[self._i][0]
            if self._get_mu_() != 0:
                resi_to_add = self._compute_sharp_intermediary_residuals_()
            if self._i > 0: self._phis[self._i - 1] = None
        else:
            def_z = self.id_grid - self.field / self.n_step
            def_I = def_z
            resi_to_add = self.residuals

        if self.flag_W:
            self._update_image_weighted_semiLagrangian_(def_I,resi_to_add,sharp=self.flag_sharp)
            self._update_residuals_weighted_semiLagrangian_(def_z)

            # DEBUG
            # print(">> ",self._i)
            # if v_scalar_n.sum()> 0:# and self._i in [0,3,8,15]:
            #     fig,ax = plt.subplots(2,2)
            #     # tb.quiver_plot(self.rf.border_normal_vector(self._i).cpu(),ax=ax[0],step= 5)
            #     _1_= ax[0,0].imshow(self.residuals.detach().cpu())
            #     fig.colorbar(_1_,ax=ax[0,0],fraction=0.046, pad=0.04)
            #     ax[0,0].set_title('z')
            #     _2_ = ax[0,1].imshow((self.rf.inv_F(self._i)*(div_fzv)).cpu().detach())
            #     fig.colorbar(_2_,ax=ax[0,1],fraction=0.046, pad=0.04)
            #     ax[0,1].set_title('div_fzv')
            #     _3_ = ax[1,0].imshow((( z_time_dtF)).cpu().detach())
            #     fig.colorbar(_3_,ax=ax[1,0],fraction=0.046, pad=0.04)
            #     ax[1,0].set_title('z x dtF')
            #     _4_ = ax[1,1].imshow((border).cpu().detach())
            #     fig.colorbar(_4_,ax=ax[1,1],fraction=0.046, pad=0.04)
            #     ax[1,1].set_title('border')
            #     plt.show()
            # else:
            #     print("-------------ZERO------------------")
            # print(self._i,' : ',border.mean()/div_fzv.mean(), z_time_dtF.mean()/div_fzv.mean())

        else:
            self._update_image_semiLagrangian_(def_I,resi_to_add,
                                               sharp = self.flag_sharp)
            self._update_residuals_semiLagrangian_(def_z)

        return (self.image, self.field, self.residuals)

    def compute_field_sim(self):
        if not self.flag_O:
            print("Field similarity do not apply here flag_O = ",self.flag_O)
            return torch.inf
        N_time_HW = self.orienting_field.shape[0]*prod(self.orienting_field.shape[2:])
        return (self.field_stock - self.orienting_field).abs().sum()/N_time_HW

    def plot_field_sim(self):
        if self.flag_O:
            diff = self.field_stock - self.orienting_field
            diff = (diff.abs().sum(axis=-1)*self.orienting_mask[:,0]).sum(axis=0)
            print(diff.shape)
            fig,ax = plt.subplots()
            p = ax.imshow(diff,origin='lower')
            fig.colorbar(p,ax=ax,fraction=0.046, pad=0.04)
        else:
            print("Not oriented ! No field_sim plot !")



class Constrained_Optim(Optimize_geodesicShooting):

    def __init__(self, source, target, geodesic, cost_cst,data_term=None, optimizer_method='adadelta'):
        super().__init__(source, target, geodesic, cost_cst,data_term, optimizer_method)
        if self.mp.flag_O: self._cost_saving_ = self._oriented_cost_saving_


    def _get_mu_(self):
        return float(self.mp._get_mu_())

    def _get_rho_(self):
        return float(self.mp._get_rho_())

    def _get_gamma_(self):
        return float(self.mp._get_gamma_())

    def _oriented_cost_saving_(self, i, loss_stock):
        """ A variation of Optimize_geodesicShooting._default_cost_saving_

        :param i: index for saving the according values
                !!! if `loss_stock` is None, `loss_stock` will be initialized, and
                `i` must have the value of the number of iterations.
        :param loss_stock:
        :return: updated `loss_stock`
        """
        # Initialise loss_stock
        if loss_stock is None:
            d = 4
            return torch.zeros((i, d))

        loss_stock[i, 0] = self.data_loss.detach()
        loss_stock[i, 1] = self.norm_v_2.detach()
        if self._get_mu_() != 0:
            loss_stock[i, 2] = self.norm_l2_on_z.detach()
        if self._get_gamma_() != 0:
            loss_stock[i, 3] = self.fields_diff_norm_V.detach()


        return loss_stock

    def get_total_cost(self):

        total_cost = self.to_analyse[1][:,0] + \
                    self.cost_cst * self.to_analyse[1][:,1]
        if self._get_mu_() != 0 :
            total_cost += self.cost_cst*(self._get_rho_())* self.to_analyse[1][:,2]
        if self.mp.flag_O:
            total_cost += self.cost_cst*(self._get_gamma_())*self.to_analyse[1][:,3]

        return total_cost

    def cost(self, residuals_ini: torch.Tensor) -> torch.Tensor:

        lamb = self.cost_cst
        rho = self._get_rho_()

        self.mp.forward(self.source, residuals_ini, save=False, plot=0)

        # Compute the data_term. Default is the Ssd
        self.data_loss = self.data_term()

        # Norm V
        C = residuals_ini.shape[1]
        grad_source = tb.spacialGradient(self.source)
        grad_source_resi = (grad_source * residuals_ini.unsqueeze(2)).sum(dim=1) / C
        K_grad_source_resi = self.mp.kernelOperator(grad_source_resi)
        self.norm_v_2 = (grad_source_resi * K_grad_source_resi).sum()

        self.total_cost = self.data_loss + lamb * self.norm_v_2

        #  || v - w ||_V
        if self.mp.flag_O:
            fields_diff = grad_source_resi - tb.grid2im(self.mp.orienting_field[0][None])
            K_fields_diff = self.mp.kernelOperator(fields_diff)
            self.fields_diff_norm_V = (fields_diff * K_fields_diff).sum()/prod(self.source.shape[2:])

            self.total_cost += lamb * self._get_gamma_() * self.fields_diff_norm_V


        if self._get_mu_() != 0:
            # # Norm on the residuals only
            if self.mp.flag_W:
                self.norm_l2_on_z = (self.mp.rf.F(0) * residuals_ini ** 2).sum()/prod(self.source.shape[2:])
            else:
                self.norm_l2_on_z = (residuals_ini ** 2).sum()/prod(self.source.shape[2:])

            self.total_cost += lamb * rho * self.norm_l2_on_z

        # print('ssd :',self.ssd,' norm_v :',self.norm_v_2)
        return self.total_cost


    def forward(self,
                z_0,
                n_iter=10,
                grad_coef=1e-3,
                verbose=True,
                plot=False,
                sharp=None
                ):
        if self.mp.flag_W:
            self.mp.rf.to_device(z_0.device)
        if self.mp.flag_O:
            self.mp.orienting_mask = self.mp.orienting_mask.to(z_0.device)
            self.mp.orienting_field = self.mp.orienting_field.to(z_0.device)
        super(Constrained_Optim, self).forward(z_0, n_iter, grad_coef, verbose,plot)
        self.to_device('cpu')

    def to_device(self, device):
        if self.mp.flag_W:
            self.mp.rf.to_device(device)
        if self.mp.flag_O:
            self.mp.orienting_mask = self.mp.orienting_mask.to(device)
            self.mp.orienting_field = self.mp.orienting_field.to(device)
        super(Constrained_Optim, self).to_device(device)

    def plot_cost(self):
        plt.rcParams['figure.figsize'] = [10, 10]
        fig1, ax1 = plt.subplots(1, 2)

        cost_stock = self.to_analyse[1].detach().numpy()

        ssd_plot = cost_stock[:, 0]
        ax1[0].plot(ssd_plot, "--", color='blue', label='ssd')
        ax1[1].plot(ssd_plot, "--", color='blue', label='ssd')

        normv_plot = self.cost_cst * cost_stock[:, 1]
        ax1[0].plot(normv_plot, "--", color='green', label='normv')
        ax1[1].plot(cost_stock[:, 1], "--", color='green', label='normv')
        total_cost = ssd_plot + normv_plot
        if self._get_mu_() != 0:
            norm_l2_on_z = self.cost_cst * (self._get_rho_()) * cost_stock[:, 2]
            total_cost += norm_l2_on_z
            ax1[0].plot(norm_l2_on_z, "--", color='orange', label='norm_l2_on_z')
            ax1[1].plot(cost_stock[:, 2], "--", color='orange', label='norm_l2_on_z')

        if self._get_gamma_() != 0:
            fields_diff_norm_v = self.cost_cst * self._get_gamma_() * cost_stock[:, 3]
            total_cost += fields_diff_norm_v
            ax1[0].plot(fields_diff_norm_v, "--", color="purple", label='fields_diff_norm_v')
            ax1[1].plot(cost_stock[:, 3], "--", color='purple', label='fields_diff_norm_v')

        ax1[0].plot(total_cost, color='black', label=r'\Sigma')
        ax1[0].legend()
        ax1[1].legend()
        ax1[0].set_title("Lambda = " + str(self.cost_cst) +
                         " mu = " + str(self._get_mu_()) +
                         " rho = " + str(self._get_rho_()) +
                         "gamma = " + str(self._get_mu_()))



class Reduce_field_Optim(Optimize_geodesicShooting):

    def __init__(self,source, target, geodesic, cost_cst, optimizer_method,mask_reduce,gamma):
        super().__init__(source, target, geodesic, cost_cst, optimizer_method)
        self._cost_saving_ = self._reduce_cost_saving_
        self.gamma = gamma
        self.mask_reduce = mask_reduce
        if self.mp.flag_O:
            raise ValueError("Geodesic integrator was set incorrectly. Reduce field"
                             "should not be used with an oriented framework. mr.glag_O is True.")

    def _get_mu_(self):
        return float(self.mp._get_mu_())

    def _get_rho_(self):
        return float(self.mp._get_rho_())

    def _get_gamma_(self):
        return float(self.gamma)

    def _reduce_cost_saving_(self, i, loss_stock):
        """ A variation of Optimize_geodesicShooting._default_cost_saving_

        :param i: index for saving the according values
                !!! if `loss_stock` is None, `loss_stock` will be initialized, and
                `i` must have the value of the number of iterations.
        :param loss_stock:
        :return: updated `loss_stock`
        """
        # Initialise loss_stock
        if loss_stock is None:
            d = 4
            return torch.zeros((i, d))

        loss_stock[i, 0] = self.ssd.detach()
        loss_stock[i, 1] = self.norm_v_2.detach()
        loss_stock[i,3] = self.norm_l2_on_mask1.detach()
        if self._get_mu_() != 0:
            loss_stock[i, 2] = self.norm_l2_on_z.detach()


        return loss_stock

    def get_total_cost(self):
        return super().get_total_cost() + self.cost_cst * self._get_gamma_() * self.to_analyse[1][:,3]

    def cost(self, residuals_ini: torch.Tensor) -> torch.Tensor:

        lamb = self.cost_cst
        rho = self._get_rho_()

        self.mp.forward(self.source, residuals_ini, save=True, plot=0)
        # checkpoint(self.mp.forward,self.source.clone(),self.id_grid,residuals_ini)
        # self.ssd = .5*((self.target - self.mp.image)**2).sum()
        self.ssd = cf.SumSquaredDifference(self.target)(self.mp.image)

        # Norm V
        C = residuals_ini.shape[1]
        grad_source = tb.spacialGradient(self.source)
        grad_source_resi = (grad_source * residuals_ini.unsqueeze(2)).sum(dim=1) / C
        K_grad_source_resi = self.mp.kernelOperator(grad_source_resi)
        self.norm_v_2 = (grad_source_resi * K_grad_source_resi).sum()

        self.total_cost = self.ssd + lamb * self.norm_v_2

        # Reduce field norm
        mask_1 = tb.imgDeform(self.mask_reduce[0][None].cpu(),self.mp.get_deformator(),dx_convention='pixel')
        self.norm_l2_on_mask1 = (mask_1**2).sum()/prod(self.source.shape[2:])

        self.total_cost += lamb * self._get_gamma_() *self.norm_l2_on_mask1

        if self._get_mu_() != 0:
            # # Norm on the residuals only
            if self.mp.flag_W:
                self.norm_l2_on_z = (self.mp.rf.F(0) * residuals_ini ** 2).sum()/prod(self.source.shape[2:])
            else:
                self.norm_l2_on_z = (residuals_ini ** 2).sum()/prod(self.source.shape[2:])

            self.total_cost += lamb * rho * self.norm_l2_on_z

        # print('ssd :',self.ssd,' norm_v :',self.norm_v_2)
        return self.total_cost

    def forward(self,
                z_0,
                n_iter=10,
                grad_coef=1e-3,
                verbose=True,
                sharp=None
                ):
        if self.mp.flag_W:
            self.mp.rf.to_device(z_0.device)
        self.mask_reduce.to(z_0.device)
        super(Reduce_field_Optim, self).forward(z_0, n_iter, grad_coef, verbose)
        self.to_device('cpu')

    def to_device(self, device):
        if self.mp.flag_W:
            self.mp.rf.to_device(device)
        self.mask_reduce.to(device)
        super(Reduce_field_Optim, self).to_device(device)

    def plot_cost(self):
        plt.rcParams['figure.figsize'] = [10, 10]
        fig1, ax1 = plt.subplots(1, 2)

        cost_stock = self.to_analyse[1].detach().numpy()

        ssd_plot = cost_stock[:, 0]
        ax1[0].plot(ssd_plot, "--", color='blue', label='ssd')
        ax1[1].plot(ssd_plot, "--", color='blue', label='ssd')

        normv_plot = self.cost_cst * cost_stock[:, 1]
        ax1[0].plot(normv_plot, "--", color='green', label='normv')
        ax1[1].plot(cost_stock[:, 1], "--", color='green', label='normv')
        total_cost = ssd_plot + normv_plot
        if self._get_mu_() != 0:
            norm_l2_on_z = self.cost_cst * (self._get_rho_()) * cost_stock[:, 2]
            total_cost += norm_l2_on_z
            ax1[0].plot(norm_l2_on_z, "--", color='orange', label='norm_l2_on_z')
            ax1[1].plot(cost_stock[:, 2], "--", color='orange', label='norm_l2_on_z')

        if self._get_gamma_() != 0:
            reduce_field_norm = self.cost_cst * self._get_gamma_() * cost_stock[:, 3]
            total_cost += reduce_field_norm
            ax1[0].plot(reduce_field_norm, "--", color="purple", label='reduce_field_norm_2')
            ax1[1].plot(cost_stock[:, 3], "--", color='purple', label='reduce_field_norm_2')

        ax1[0].plot(total_cost, color='black', label=r'\Sigma')
        ax1[0].legend()
        ax1[1].legend()
        ax1[0].set_title("Lambda = " + str(self.cost_cst) +
                         " mu = " + str(self._get_mu_()) +
                         " rho = " + str(self._get_rho_()) +
                         "gamma = " + str(self._get_mu_()))

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
            if module == 'torch.storage' and name == '_load_from_bytes':
                return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
            else:
                if module == 'metamorphosis': module = 'my_metamorphosis.metamorphosis'
                if name == 'metamorphosis_path': name = 'Metamorphosis_path'
                # print('module :',module,' name : ', name)
                return super().find_class(module, name)

    if path is None:
        path =ROOT_DIRECTORY+OPTIM_SAVE_DIR
    if not file_name in os.listdir(path):
        raise ValueError("File "+file_name+" does not exist in "+path)
    with open(path+file_name,'rb') as f:
        # opti_dict = pickle.load(f)
        opti_dict = CPU_Unpickler(f).load()

    # TODO : Modifier ça optimize_geo est une classe abstraite maintenant.
    # prevoir de decider de quelle class on parle en fonction de "F".
    try:
        isinstance(opti_dict['mp'].rf,Residual_norm_function)
        optimizer = Constrained_Optim
        print('Constrained > Weighted')
    except AttributeError:
        optimizer = Optimize_metamorphosis
        print('Classic Metamorphosis loaded :' )
    print('DT:',opti_dict['data_term'])
    new_optim = optimizer(opti_dict['source'],
                            opti_dict['target'],
                            opti_dict['mp'],
                            cost_cst=opti_dict['cost_cst'],
                            data_term=opti_dict['data_term'],
                            optimizer_method=opti_dict['optimizer_method_name'])
    for k in FIELD_TO_SAVE[5:]:
        try:
            new_optim.__dict__[k] = opti_dict[k]
        except KeyError:
            print("old fashioned Metamorphosis : No data_term, default is Ssd")
            pass
    print("\n Je suis ici")
    if 'landmarks' in opti_dict.keys():
        print("opti_dict len",len(opti_dict['landmark']))
        new_optim.source_landmark = opti_dict['landmarks'][0]
        new_optim.target_landmark = opti_dict['landmarks'][1]
        new_optim.deform_landmark = opti_dict['landmarks'][2]

    new_optim.loaded_from_file = file_name
    if verbose: print('New optimiser loaded :\n',new_optim.__repr__())
    return new_optim



# ===================================================================
#
#       APIs
#
# ==================================================================
@time_it
def lddmm(source,target,residuals,
          sigma,cost_cst,
          integration_steps,n_iter,grad_coef,
          data_term =None,
          sharp=False,
          safe_mode = False,
          integration_method='semiLagrangian',
          multiScale_average = False
          ):
    if type(residuals) == int: residuals = torch.zeros(source.shape,device=source.device)
    residuals.requires_grad = True
    if sharp: integration_method = 'sharp'

    sigma = tb.format_sigmas(sigma,len(source.shape[2:]))

    mp = Metamorphosis_path(method=integration_method,
                        mu=0, rho=0,
                        sigma_v=sigma,
                        n_step=integration_steps,
                        multiScale_average=multiScale_average
                        )
    mr = Optimize_metamorphosis(source,target,mp,
                                cost_cst=cost_cst,
                                # optimizer_method='LBFGS_torch',
                                optimizer_method='adadelta',
                                data_term=data_term
                               )
    if not safe_mode:
        mr.forward(residuals,n_iter=n_iter,grad_coef=grad_coef)
    else:
        mr.forward_safe_mode(residuals,n_iter=n_iter,grad_coef=grad_coef)
    return mr

@time_it
def metamorphosis(source,target,residuals,
                  mu,rho,sigma,cost_cst,
                  integration_steps,n_iter,grad_coef,
                  data_term=None,
                  sharp=False,
                  safe_mode = True,
                  integration_method='semiLagrangian'
                  ):
    if type(residuals) == int: residuals = torch.zeros(source.shape,device=source.device)
    # residuals = torch.zeros(source.size()[2:],device=device)
    residuals.requires_grad = True
    if sharp: integration_method= 'sharp'
    sigma = tb.format_sigmas(sigma,len(source.shape[2:]))

    mp = Metamorphosis_path(method=integration_method,
                        mu=mu, rho=rho,
                        sigma_v=sigma,
                        n_step=integration_steps,
                        )
    mr = Optimize_metamorphosis(source,target,mp,
                                cost_cst=cost_cst,
                                data_term=data_term,
                                # optimizer_method='LBFGS_torch')
                                optimizer_method='adadelta')
    if not safe_mode:
        mr.forward(residuals,n_iter=n_iter,grad_coef=grad_coef)
    else:
        mr.forward_safe_mode(residuals,n_iter=n_iter,grad_coef=grad_coef)
    return mr


@time_it
def weighted_metamorphosis(source,target,residual,mask,
                           mu,rho,rf_method,sigma,cost_cst,
                           n_iter,grad_coef,data_term=None,sharp=False,
                           safe_mode=True):
    device = source.device
    sigma = tb.format_sigmas(sigma,len(source.shape[2:]))
    if rf_method == 'identity':
        rf = Residual_norm_identity(mask.to(device),mu,rho)
    elif rf_method == 'borderBoost':
        rf = Residual_norm_borderBoost(mask.to(device),mu,rho)
    else:
        raise ValueError(f"rf_method must be 'identity' or 'borderBoost'")
    if type(residual) == int: residual = torch.zeros(source.shape,device=device)
    residual.requires_grad = True

    mp_weighted = Constrained_meta_path(
        residual_function=rf,sigma_v=sigma,
        sharp=sharp
    )
    mr_weighted = Constrained_Optim(
        source,target,mp_weighted,
        cost_cst=cost_cst,optimizer_method='adadelta',
        data_term=data_term
    )
    if not safe_mode:
        mr_weighted.forward(residual,n_iter=n_iter,grad_coef=grad_coef)
    else:
        mr_weighted.forward_safe_mode(residual,n_iter=n_iter,grad_coef=grad_coef)
    return mr_weighted

@time_it
def oriented_metamorphosis(source,target,residual,mp_orienting,
                           mu,rho,gamma,sigma,cost_cst,
                           n_iter,grad_coef):
    mask = mp_orienting.image_stock.to(source.device)
    orienting_field = mp_orienting.field_stock.to(source.device)
    if type(residual) == int: residual = torch.zeros(source.shape)
    residual.requires_grad = True

    # start = time.time()
    mp_orient = Constrained_meta_path(orienting_mask=mask,
                                      orienting_field=orienting_field,
                                mu=mu,rho=rho,gamma=gamma,
                                sigma_v=(sigma,)*len(residual.shape)
                                # n_step=20 # n_step is defined from mask.shape[0]
                                )
    mr_orient = Constrained_Optim(source,target,mp_orient,
                                       cost_cst=cost_cst,
                                       # optimizer_method='LBFGS_torch')
                                       optimizer_method='adadelta')
    mr_orient.forward(residual,n_iter=n_iter,grad_coef=grad_coef)
    return mr_orient

@time_it
def constrained_metamorphosis(source,target,residual,
                           rf_method,mu,rho,mask_w,
                           mp_orienting,gamma,mask_o,
                           sigma,cost_cst,sharp,
                           n_iter,grad_coef):
    mask = mp_orienting.image_stock.to(source.device)
    orienting_field = mp_orienting.field_stock.to(source.device)
    sigma = tb.format_sigmas(sigma,len(source.shape[2:]))

    if rf_method == 'identity':
        rf_method = Residual_norm_identity(mask,mu,rho)
    elif rf_method == 'borderBoost':
        rf_method = Residual_norm_borderBoost(mask,mu,rho)
    else:
        raise ValueError(f"rf_method must be 'identity' or 'borderBoost'")
    if type(residual) == int: residual = torch.zeros(source.shape,device=source.device)
    residual.requires_grad = True

    # start = time.time()
    mp_constr = Constrained_meta_path(orienting_mask=mask,
                                      orienting_field=orienting_field,
                                      residual_function=rf_method,
                                mu=mu,rho=rho,gamma=gamma,
                                sigma_v=sigma,
                                sharp=sharp,
                                # n_step=20 # n_step is defined from mask.shape[0]
                                )
    mr_constr = Constrained_Optim(source,target,mp_constr,
                                       cost_cst=cost_cst,
                                       # optimizer_method='LBFGS_torch')
                                       optimizer_method='adadelta')
    mr_constr.forward(residual,n_iter=n_iter,grad_coef=grad_coef)
    return mr_constr