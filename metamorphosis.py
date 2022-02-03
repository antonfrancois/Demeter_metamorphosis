import torch
import matplotlib.pyplot as plt
import warnings
from math import prod
import pickle
import os, sys, csv#, time
sys.path.append(os.path.abspath('../'))
from datetime import datetime
from abc import ABC, abstractmethod

import my_torchbox as tb
import vector_field_to_flow as vff
from my_toolbox import update_progress,format_time, get_size
from my_optim import GradientDescent
import reproducing_kernels as rk
import cost_functions as cf
from constants import FIELD_TO_SAVE,ROOT_DIRECTORY
from decorators import deprecated

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
    def __init__(self):
        super().__init__()

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
        verbose=False):
        r""" This method is doing the temporal loop using the good method `_step_`

        :param image: (tensor array) of shape [1,1,H,W]. Source image ($I_0$)
        :param field_ini: to be deprecated, field_ini is id_grid
        :param residual_ini: (tensor array) of shape [H,W]. initial residual ($z_0$)
        :param save: (bool) option to save the integration intermediary steps.
        :param plot: (int) positive int lower than `self.n_step` to plot the indicated
                         number of intemediary steps. (if plot>0, save is set to True)

        """
        if len(residual_ini.shape) not in [2,3]:
            raise ValueError(f"residual_ini must be of shape [H,W] or [D,H,W] got {residual_ini.shape}")
        device = residual_ini.device
        self.source = image.detach()
        self.image = image.clone().to(device)
        self.residuals = residual_ini
        self.save = save

        self.id_grid = tb.make_regular_grid(residual_ini.shape,device=device)
        if field_ini is None:
            self.field =self.id_grid.clone()
        else:
            self.field = field_ini #/self.n_step

        if plot > 0:
            self.save = True

        if self.save:
            self.image_stock = torch.zeros((t_max*self.n_step,)+image.shape[1:])
            self.field_stock = torch.zeros((t_max*self.n_step,)+self.field.shape[1:])
            self.residuals_stock = torch.zeros((t_max*self.n_step,)+residual_ini.shape)


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

            # print('At iteration ',i,' my weight is ',get_size(self))

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
        return tb.im2grid(self.kernelOperator((-residuals * grad_image[0])))

    def _field_cst_mult(self):
        if self.mu == 0:
            return 1
        else:
            return self.rho/self.mu

    def _update_field_(self):
        grad_image = tb.spacialGradient(self.image,dx_convention='pixel')
        self.field = self._compute_vectorField_(self.residuals,grad_image)
        self.field *= self._field_cst_mult()

    def _update_residuals_Eulerian_(self):
        residuals_dt = - tb.Field_divergence(dx_convention='pixel')(
                            self.residuals[None,:,:,None]* self.field,
                            )[0,0]
        self.residuals = self.residuals + residuals_dt /self.n_step

    def _update_residuals_semiLagrangian_(self,deformation):
        div_v_times_z = self.residuals* tb.Field_divergence(dx_convention='pixel')(self.field)[0,0]

        self.residuals = tb.imgDeform(self.residuals[None,None],
                                      deformation,
                                      dx_convention='pixel',
                                      clamp=False)[0,0]\
                         - div_v_times_z/self.n_step

    def _update_image_Eulerian_(self):
        self.image =  self._image_Eulerian_integrator_(self.image,self.field,1/self.n_step,1)
        self.image = self.image + (self.residuals *self.mu)/ self.n_step

    def _update_image_semiLagrangian_(self,deformation):
        self.image = tb.imgDeform(self.image,deformation,dx_convention='pixel') \
                     + (self.residuals *self.mu)/self.n_step


    def get_deformation(self,n_step=-1,save=False):
        r"""Returns the deformation use it for showing results
        $\Phi = \int_0^1 v_t dt$

        :return: deformation [1,H,W,2] or [2,H,W,D,3]
        """
        if n_step == 0:
            return self.field_stock[0][None]/self.n_step
        temporal_integrator = vff.FieldIntegrator(method='temporal',save=save)
        return temporal_integrator(self.field_stock[:n_step]/self.n_step,forward=True)

    def get_deformator(self,save = False):
        r"""Returns the inverse deformation use it for deforming images
        $\Phi^{-1}$

        :return: deformation [T,H,W,2] or [T,H,W,D,3]
        """
        temporal_integrator = vff.FieldIntegrator(method='temporal',save=save)
        return temporal_integrator(self.field_stock/self.n_step,forward=False)

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
        plt.rcParams['figure.figsize'] = [size_fig*3,n_figs*size_fig]
        fig,ax = plt.subplots(n_figs,3)

        for i,t in enumerate(plot_id):
            i_s =ax[i,0].imshow(self.image_stock[t,0,:,:].detach().numpy(),
                                **kw_image_args)
            ax[i,0].set_title("t = "+str((t/(self.n_step-1)).item())[:3])
            ax[i,0].axis('off')
            fig.colorbar(i_s,ax=ax[i,0],fraction=0.046, pad=0.04)

            r_s =ax[i,1].imshow(self.residuals_stock[t].detach().numpy(),
                                **kw_residuals_args)
            ax[i,1].axis('off')

            fig.colorbar(r_s,ax=ax[i,1],fraction=0.046, pad=0.04)

            tb.gridDef_plot_2d(self.get_deformation(t),
                            add_grid=True,
                            ax=ax[i,2],
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

        tb.gridDef_plot_2d(full_deformation,step=int(max(self.image.shape)/40),ax = axes[0,0],
                         check_diffeo=True,dx_convention='2square')
        tb.quiver_plot(full_deformation -self.id_grid.cpu() ,step=int(max(self.image.shape)/40),
                        ax = axes[0,1],check_diffeo=False)

        # show S deformed by full_deformation
        S_deformed = tb.imgDeform(self.source.cpu(),full_deformator,
                                  dx_convention='pixel')
        axes[1,0].imshow(self.source[0,0,:,:].cpu(),cmap='gray',origin='lower',vmin=0,vmax=1)
        axes[1,1].imshow(target[0,0].cpu(),cmap='gray',origin='lower',vmin=0,vmax=1)
        axes[2,0].imshow(S_deformed[0,0,:,:],cmap='gray',origin='lower',vmin=0,vmax=1)
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


class Optimize_geodesicShooting(torch.nn.Module,ABC):
    """ Abstract method for geodesic shooting optimisation. It needs to be provided with an object
    inheriting from Geodesic_integrator

    """
    @abstractmethod
    def __init__(self,source : torch.Tensor,
                     target : torch.Tensor,
                     geodesic : Geodesic_integrator,
                     cost_cst,
                     optimizer_method : str = 'grad_descent'
                 ):
        super().__init__()
        self.mp = geodesic
        self.source = source
        self.target = target
        if len(self.mp.sigma_v) != len(source.shape[2:]):
            raise ValueError(f"Geodesic integrator :{self.mp.__class__.__name__}"
                             f"was initialised to be {len(self.mp.sigma_v)}D"
                             f" with sigma_v = {self.mp.sigma_v} and go image "
                             f"source.size() = {source.shape}"
                             )

        self.cost_cst = cost_cst
        self.optimizer_method_name = optimizer_method #for __repr__
        # optimize on the cost as defined in the 2021 paper.


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

        self.temporal_integrator = vff.FieldIntegrator(method='temporal',save=False)
        self.is_DICE_cmp = False

    @abstractmethod
    def cost(self,residual_ini):
        pass

    @abstractmethod
    def _get_mu_(self):
        pass

    @abstractmethod
    def _get_rho_(self):
        pass

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
        if self._get_mu_() != 0: # metamophosis
            loss_stock[i,0] = self.ssd.detach()
            loss_stock[i,1] = self.norm_v_2.detach()
            loss_stock[i,2] = self.norm_l2_on_z.detach()
        else: #LDDMM
            loss_stock[i,0] = self.ssd.detach()
            loss_stock[i,1] = self.norm_v_2.detach()
        return loss_stock

    def forward(self,z_0,n_iter = 10,grad_coef = 1e-3,verbose= True) :
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

        self.parameter = z_0 # optimized variable
        self._initialize_optimizer_(grad_coef,max_iter=n_iter)

        self.id_grid = tb.make_regular_grid(z_0.shape,z_0.device)

        self.cost(self.parameter)
        #
        d = 3 if self._get_mu_() != 0 else 2
        loss_stock = torch.zeros((n_iter,d))
        loss_stock = self._default_cost_saving_(0,loss_stock)

        for i in range(1,n_iter):
            # print("\n",i,"==========================")
            self._step_optimizer_()
            loss_stock = self._default_cost_saving_(i,loss_stock)

            if verbose:
                update_progress((i+1)/n_iter,message=('cost_val',loss_stock[i,0]))

        # for future plots compute shooting with save = True
        self.mp.forward(self.source.clone(),
                        self.parameter.detach().clone(),
                        save=True,plot=0)

        self.to_device('cpu')
        self.to_analyse = (self.parameter.detach(),loss_stock)

    def to_device(self,device):
        self.source = self.source.to(device)
        self.target = self.target.to(device)
        self.parameter = self.parameter.to(device)
        self.id_grid = self.id_grid.to(device)

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
            self.forward(z_0,n_iter,grad_coef,verbose)
        except OverflowError:
            if mode is None:
                print("Integration diverged : Stop.")
                self.to_analyse= 'Integration diverged'
            elif mode == "grad_coef":
                print(f"Integration diverged :"
                      f" set grad_coef to {grad_coef*0.1}")
                self.forward_safe_mode(z_0,n_iter,grad_coef*0.1,verbose,mode=mode)


    def compute_DICE(self,source_segmentation,target_segmentation):
        """ Compute the DICE score of a regristration. Given the segmentations of
        a structure  (ex: ventricules) that should be present in both source and target image.
        it gives a score close to one if the segmentations are well matching after transformation.


        :param source_segmentation: Tensor of source size?
        :param target_segmentation:
        :return: (float) DICE score.
        """
        self.is_DICE_cmp = True
        source_deformed = tb.imgDeform(source_segmentation[None,None],self.mp.get_deformator(),
                                       dx_convention='pixel')[0,0]
        source_deformed[source_deformed>0] =1
        prod_seg = source_deformed * target_segmentation
        sum_seg = source_deformed + target_segmentation

        self.DICE = 2*prod_seg.sum() / sum_seg.sum()
        return  self.DICE

    def get_DICE(self):
        if self.is_DICE_cmp :
            return self.DICE
        else:
            return 'not computed'

    def get_ssd_def(self):
        image_def = tb.imgDeform(self.source,self.mp.get_deformator(),dx_convention='pixel')
        return float(cf.sumSquaredDifference(self.target)(image_def))

    def save(self,source_name,target_name,message=None):

        if self.to_analyse == 'Integration diverged':
            print("Can't save optimisation that didn't converged")
            return 0

        path =ROOT_DIRECTORY+'/my_metamorphosis/saved_optim/'
        date_time = datetime.now()
        n_dim = '2D' if len(self.mp.sigma_v) ==2 else '3D'
        id_num = 0

        # flag for discriminating against different kinds of Optimizers
        try:
            isinstance(self.mp.rf,Residual_norm_function)
            modifier_str = self.mp.rf.__repr__()
        except AttributeError:
            modifier_str = 'None'

        # build file name
        def file_name_maker_(id_num):
            return n_dim+date_time.strftime("_%d_%m_%Y")+'_'+\
                    source_name+'_to_'+target_name+'_{:03d}'.format(id_num)+\
                    '.pk1'
        file_name = file_name_maker_(id_num)
        while file_name in os.listdir(path):
            id_num+=1
            file_name = file_name_maker_(id_num)

        # =================
        # write to csv
        csv_file = 'saves_overview.csv'
        state_dict = dict(
            time=date_time.strftime("%d/%m/%Y %H:%M:%S"),
            saved_file_name=file_name,
            source=source_name,
            target=target_name,
            n_dim= n_dim,
            shape=self.source.shape.__str__()[10:],
            modifier=modifier_str,
            method=self.optimizer_method_name,
            final_loss=float(self.total_cost.detach()),
            DICE = self.get_DICE(),
            mu=self._get_mu_(),
            rho=self._get_rho_(),
            lamb=self.cost_cst,
            sigma_v=self.mp.sigma_v.__str__(),
            n_step=self.mp.n_step,
            n_iter=len(self.to_analyse[1]),
            message= '' if message is None else message
        )
        with open(path+csv_file,mode='a') as csv_f:
            writer = csv.DictWriter(csv_f,state_dict.keys(),delimiter=';')
            # writer.writeheader()
            writer.writerow(state_dict)

        #=================
        # save the data

        # copy and clean dictonary containing all values
        dict_copy = {}

        for k in FIELD_TO_SAVE:
            # print(k,' >> ',self.__dict__.get(k))
            dict_copy[k] = self.__dict__.get(k)
            if torch.is_tensor(dict_copy[k]):
                dict_copy[k] = dict_copy[k].cpu().detach()
        dict_copy['mp'] = self.mp # For some reason 'mp' wasn't showing in __dict__
        # print(dict_copy.keys())
        with open(path+file_name,'wb') as f:
            pickle.dump(dict_copy,f,pickle.HIGHEST_PROTOCOL)
        print('Optimisation saved in '+path+file_name)
        return file_name,path



    # ==================================================================
    #                 PLOTS
    # ==================================================================

    def plot_cost(self):
        """ To display cost evolution during the optimisation.


        """
        plt.rcParams['figure.figsize'] = [10,10]
        fig1,ax1 = plt.subplots(1,2)

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

        ax1[0].plot(total_cost, color='black',label=r'\Sigma')
        ax1[0].legend()
        ax1[1].legend()
        ax1[0].set_title("Lambda = "+str(self.cost_cst)+
                    " mu = "+str(self._get_mu_()) +
                    " rho = "+str(self._get_rho_()))

    def plot_imgCmp(self):
        r""" Display and compare the deformed image $I_1$ with the target$
        """
        plt.rcParams['figure.figsize'] = [20,20]
        fig,ax = plt.subplots(2,2)
        image_kw = dict(cmap='gray',origin='lower',vmin=0,vmax=1)
        ax[0,0].imshow(self.source[0,0,:,:].detach().cpu().numpy(),
                       **image_kw)
        ax[0,0].set_title("source")
        ax[0,1].imshow(self.target[0,0,:,:].detach().cpu().numpy(),
                       **image_kw)
        ax[0,1].set_title("target")

        ax[1,0].imshow(tb.imCmp(self.target,self.mp.image.detach().cpu()),**image_kw)
        ax[1,0].set_title("comparaison deformed image with target")
        ax[1,1].imshow(self.mp.image[0,0].detach().cpu().numpy(),**image_kw)
        ax[1,1].set_title("deformed source image")

    def plot_deform(self,temporal_nfigs = 0):
        residuals = self.to_analyse[0]
        self.mp.forward(self.source.clone(),residuals,save=True,plot=0)
        self.mp.plot_deform(self.target,temporal_nfigs)

    def plot(self):
        _,_,H,W = self.source.shape

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
        self.anti_mask = (mask < 1e-5)

        # preparation of border
        binary_mask = 1 - self.anti_mask.to(torch.float)
        # plt.figure()
        # plt.imshow(tb.imCmp(binary_mask[12][None],self.mask[12][None]))
        # plt.show()
        normal_vector_grad = - tb.spacialGradient(binary_mask)
        # normal_vector_grad[:,0,1] *= -1
        self.unit_normal_vector_grad = torch.nn.functional.normalize(normal_vector_grad,dim=2,p=2)[:,0]

    def __repr__(self):
        return f'{self.__class__.__name__}:(mu = {self.mu:.2E},rho = {self.rho:.2E})'

    def to_device(self,device):
        self.mask = self.mask.to(device)
        self.unit_normal_vector_grad = self.unit_normal_vector_grad.to(device)

    @abstractmethod
    def f(self,t):
        pass

    def F(self,t):
        F_mat = self.f(t)[self.anti_mask]/(self.mu * self.mask[self.anti_mask])
        F_mat[self.anti_mask] = 1
        return F_mat

    def inv_F(self,t):
        return 1/self.F(t)

    @abstractmethod
    def dt_F(self,t):
        pass

    def border_normal_vector(self,t):
        return tb.im2grid(self.unit_normal_vector_grad[t][None])


# =======================================================================
# =======================================================================
#  CLASSICAL METAMORPHOSIS
# =======================================================================
# =======================================================================


class Metamorphosis_path(Geodesic_integrator):
    """ Class integrating over a geodesic shooting. The user can choose the method among
    'Eulerian', 'advection_semiLagrangian' and 'semiLagrangian.

    """
    def __init__(self,method, mu=1.,rho=1.,sigma_v= (1,1,1), n_step =10):
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
        super().__init__()
        # self.mu = mu if callable(mu) else lambda :mu
        # self.rho = rho if callable(rho) else lambda :rho
        self.mu = mu
        self.rho = rho
        if(mu == 0 and rho != 0):
            warnings.warn("mu as been set to zero in methamorphosis_path, "
                          "automatic reset of rho to zero."
                         )
            self.rho = 0
        self.n_step = n_step

        # inner methods

        # print('Before kernelOperator my weight is ',get_size(self))
        self.sigma_v = sigma_v# is used if the optmisation is later saved
        self.kernelOperator = rk.GaussianRKHS(sigma_v,
                                                 border_type='constant')

        if method == 'Eulerian':
            self.step = self._step_fullEulerian
        elif method == 'advection_semiLagrangian':
            self.step = self._step_advection_semiLagrangian
        elif method == 'semiLagrangian':
            self.step = self._step_full_semiLagrangian
        elif callable(method):
            self.step = method
        elif method== 'tkt':
            pass
        else:
            raise ValueError("You have to specify the method used : 'Eulerian','advection_semiLagrangian'"
                             " or 'semiLagrangian'")


    def _step_fullEulerian(self):
        self._update_field_()
        self._update_residuals_Eulerian_()
        self._update_image_Eulerian_()

        return (self.image,self.field,self.residuals)

    def _step_advection_semiLagrangian(self):
        self._update_field_()
        self._update_residuals_Eulerian_()
        # Lagrangian scheme on images
        deformation = self.id_grid - self.field/self.n_step
        self._update_image_semiLagrangian_(deformation)

        return (self.image,self.field,self.residuals)

    def _step_full_semiLagrangian(self):
        self._update_field_()
        # Lagrangian scheme on images and residuals
        deformation = self.id_grid - self.field/self.n_step
        self._update_residuals_semiLagrangian_(deformation)
        self._update_image_semiLagrangian_(deformation)

        return (self.image,self.field,self.residuals)

    def step(self):
        raise ValueError("You have to specify the method used : 'Eulerian','advection_semiLagrangian'"
                             " or 'semiLagrangian'")

    def _get_mu_(self):
        return self.mu

class Optimize_metamorphosis(Optimize_geodesicShooting):

    def __init__(self,source : torch.Tensor,
                     target : torch.Tensor,
                     geodesic : Metamorphosis_path,
                     cost_cst : float,
                     optimizer_method : str = 'grad_descent',
                     mask = None # For cost_function masking
                 ):
        super().__init__(source,target,geodesic,cost_cst,optimizer_method)
        self.mask = mask


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
        return self.mp.rho

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

        if  self.mask is None:
            self.ssd = cf.sumSquaredDifference(self.target)(self.mp.image)
        else:
            self.ssd = cf.sumSquaredDifference(self.target,cancer_seg=self.mask)(self.mp.image)

        # Norm V
        grad_source = tb.spacialGradient(self.source)
        grad_source_resi = grad_source[0]*residuals_ini
        K_grad_source_resi = self.mp.kernelOperator(grad_source_resi)

        self.norm_v_2 = (grad_source_resi * K_grad_source_resi).sum()

        # norm_2 on z
        if self.mp.mu != 0:
            # # Norm on the residuals only
            self.norm_l2_on_z = (residuals_ini**2).sum()/prod(self.source.shape[2:])
            self.total_cost = self.ssd + \
                              lamb*(self.norm_v_2 + (rho) *self.norm_l2_on_z)
        else:
             self.total_cost = self.ssd + lamb*self.norm_v_2
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
               f'F = rho/mu = {self.F(0)}, inv_F = {self.inv_F(0)}.'

    def f(self,t):
        return self.rho*self.mask[t,0]

    def F(self,t):
        return self.rho/self.mu

    def inv_F(self,t):
        return  self.mu/self.rho

    def dt_F(self,t):
        return 0

class Residual_norm_borderBoost(Residual_norm_function):

    def __init__(self,mask,mu,rho):
        if rho < 0:
            raise ValueError(f"rho must be a real positive value got rho = {rho:.3f}")
        super().__init__(mask, mu,rho)
        n_step = mask.shape[0]

        # Preparation of the time derivative F(M_t)
        # TODO: le spacial gradient risque de ne pas fonctionner sur le batch en 3D
        grad_mask = tb.spacialGradient(self.mask)
        self.grad_mask_norm = (grad_mask**2).sum(dim = 2).sqrt()
        # TODO : make sure that this n_step is the same that the geodesic integrator will use
        grad_dt_mask = tb.spacialGradient((self.mask[1:] - self.mask[:-1])/n_step)
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



class Weighted_meta_path(Geodesic_integrator):

    def __init__(self,residual_function,sigma_v= (1,1,1), n_step =10,border_effect=True):
        super().__init__()
        if hasattr(residual_function,'n_step') and residual_function.n_step != n_step:
            raise ValueError(f"{residual_function.__class__.__name__}.n_step is {residual_function.n_step}"
                             f"and {self.__class__.__name__} = {n_step}. They must be equal.")
        self.n_step = n_step
        self.rf = residual_function

        self.sigma_v = sigma_v# is used if the optmisation is later saved
        self.kernelOperator = rk.GaussianRKHS(sigma_v,
                                                 border_type='constant')
        self.border_effect= border_effect

    def __repr__(self):
        return self.__class__.__name__+'(\n'\
                +self.kernelOperator.__repr__()+',\n'\
                +self.rf.__repr__()+'\n)'

    def _get_mu_(self):
        return self.rf.mu

    def step(self):
        grad_image = tb.spacialGradient(self.image,dx_convention='pixel')
        self.field = self._compute_vectorField_(self.rf.F(self._i)*self.residuals,grad_image)

        # Lagrangian scheme on images and residuals
        deformation = self.id_grid - self.field/self.n_step
        fz_times_div_v= self.rf.F(self._i)*self.residuals* tb.Field_divergence(dx_convention='pixel')(self.field)[0,0]
        div_fzv = -tb.imgDeform(self.rf.F(self._i)*self.residuals[None,None],
                                      deformation,
                                      dx_convention='pixel',
                                      clamp=False)[0,0]\
                         + fz_times_div_v/self.n_step
        z_time_dtF = self.residuals*self.rf.dt_F(self._i)

        if self.border_effect:
            v_scalar_n = (self.field * self.rf.border_normal_vector(self._i)).sum(dim=-1)[0]
            border = self.residuals *v_scalar_n
        else:
            border = 0

        self.residuals = border - self.rf.inv_F(self._i) *( div_fzv + z_time_dtF)


        self.image = tb.imgDeform(self.image,deformation,dx_convention='pixel') +\
                     (self.rf.mu*self.rf.mask[self._i]*self.residuals )/self.n_step

        return (self.image,self.field,self.residuals)


class Weighted_optim(Optimize_geodesicShooting):

    def __init__(self,source : torch.Tensor,
                     target : torch.Tensor,
                     geodesic,
                     cost_cst,
                     optimizer_method : str = 'grad_descent'
                 ):
        super().__init__(source,target,geodesic,cost_cst,optimizer_method)


    def _get_mu_(self):
        return self.mp.rf.mu

    def _get_rho_(self):
        return self.mp.rf.rho

    def cost(self, residuals_ini : torch.Tensor) -> torch.Tensor:
        r""" cost computation

        $H(z_0) =   \frac 12\| \im{1} - \ti \|_{L_2}^2 + \lambda \Big[ \|v_0\|^2_V + \mu ^2 \|z_0\|^2_{L_2} \Big]$

        :param residuals_ini: z_0
        :return: $H(z_0)$ a single valued tensor
        """
        #_,_,H,W = self.source.shape
        lamb = self.cost_cst

        self.mp.forward(self.source,residuals_ini,save=False,plot=0)

        self.ssd = cf.sumSquaredDifference(self.target)(self.mp.image)

        # Norm V
        grad_source = tb.spacialGradient(self.source)
        grad_source_resi = grad_source[0]*residuals_ini
        K_grad_source_resi = self.mp.kernelOperator(grad_source_resi)


        self.norm_v_2 = (grad_source_resi * K_grad_source_resi).sum()

        # norm_2 on z
        if self.mp.rf.mask[0].sum() > 0:
            # # Norm on the residuals only
            self.norm_l2_on_z = (self.mp.rf.f(0)*residuals_ini**2).sum()/prod(self.source.shape[2:])
            # self.norm_l2_on_z = torch.tensor(0)
            self.total_cost = self.ssd + \
                              lamb*(self.norm_v_2 + self.norm_l2_on_z)
        else:
             self.total_cost = self.ssd + lamb*self.norm_v_2
        # print('ssd :',self.ssd,' norm_v :',self.norm_v_2)
        return self.total_cost

    def forward(self,z_0,n_iter = 10,grad_coef = 1e-3,verbose= True):
        self.mp.rf.to_device(z_0.device)
        if self.mp.border_effect:
            self.mp.rf.unit_normal_vector_grad = self.mp.rf.unit_normal_vector_grad.to(z_0.device)
        super(Weighted_optim,self).forward(z_0,n_iter,grad_coef,verbose)
        self.to_device('cpu')

    def to_device(self,device):
        self.mp.rf.to_device(device)
        super(Weighted_optim,self).to_device(device)

def load_optimize_geodesicShooting(file_name,path=None,verbose=True):
    """ load previously saved optimisation in order to plot it later."""
    if path is None:
        path =ROOT_DIRECTORY+'/my_metamorphosis/saved_optim/'
    if not file_name in os.listdir(path):
        raise ValueError("File "+file_name+" does not exist in "+path)
    with open(path+file_name,'rb') as f:
        opti_dict = pickle.load(f)

    # TODO : Modifier ça optimize_geo est une classe abstraite maintenant.
    # prevoir de decider de quelle class on parle en fonction de "F".
    try:
        isinstance(opti_dict['mp'].rf,Residual_norm_function)
        optimizer = Weighted_optim
        print('Weighted')
    except AttributeError:
        optimizer = Optimize_metamorphosis
        print('Classic Metamorphosis loaded :' )
    new_optim = optimizer(opti_dict['source'],
                            opti_dict['target'],
                            opti_dict['mp'],
                            cost_cst=opti_dict['cost_cst'],
                            optimizer_method=opti_dict['optimizer_method_name'])
    for k in FIELD_TO_SAVE[5:]:
        new_optim.__dict__[k] = opti_dict[k]

    if verbose: print('New optimiser loaded :\n',new_optim.__repr__())
    return new_optim


# ===================================================================
#
#       APIs
#
# ==================================================================

def lddmm(source,target,residuals,
          sigma,cost_cst,
          integration_steps,n_iter,grad_coef,
          safe_mode = False
          ):

    # residuals = torch.zeros(source.size()[2:],device=device)
    residuals.requires_grad = True

    mp = Metamorphosis_path(method='semiLagrangian',
                        mu=0, rho=0,
                        sigma_v=(sigma,)*len(source.shape[2:]),
                        n_step=integration_steps)
    mr = Optimize_metamorphosis(source,target,mp,
                                           cost_cst=cost_cst,
                                           # optimizer_method='LBFGS_torch')
                                           optimizer_method='adadelta')
    if not safe_mode:
        mr.forward(residuals,n_iter=n_iter,grad_coef=grad_coef)
    else:
        mr.forward_safe_mode(residuals,n_iter=n_iter,grad_coef=grad_coef)
    return mr

def metamorphosis(source,target,residuals,
                  mu,rho,sigma,cost_cst,
                  integration_steps,n_iter,grad_coef,
                  safe_mode = True
                  ):

    # residuals = torch.zeros(source.size()[2:],device=device)
    residuals.requires_grad = True

    mp = Metamorphosis_path(method='semiLagrangian',
                        mu=mu, rho=rho,
                        sigma_v=(sigma,)*len(source.shape[2:]),
                        n_step=integration_steps)
    mr = Optimize_metamorphosis(source,target,mp,
                                           cost_cst=cost_cst,
                                           # optimizer_method='LBFGS_torch')
                                           optimizer_method='adadelta')
    if not safe_mode:
        mr.forward(residuals,n_iter=n_iter,grad_coef=grad_coef)
    else:
        mr.forward_safe_mode(residuals,n_iter=n_iter,grad_coef=grad_coef)
    return mr

def weighted_metamorphosis(source,target,residual,mask,
                           mu,rho,rf_method,sigma,cost_cst,
                           n_iter,grad_coef,
                           safe_mode=True):
    residual.requires_grad = True
    if rf_method == 'identity':
        rf = Residual_norm_identity(mask,mu,rho)
    elif rf_method == 'borderBoost':
        rf = Residual_norm_borderBoost(mask,mu,rho)

    mp_weighted = Weighted_meta_path(
        rf,sigma_v=(sigma,sigma),n_step=mask.shape[0],
        border_effect=False
    )
    mr_weighted = Weighted_optim(
        source,target,mp_weighted,
        cost_cst=cost_cst,optimizer_method='adadelta'
    )
    if not safe_mode:
        mr_weighted.forward(residual,n_iter=n_iter,grad_coef=grad_coef)
    else:
        mr_weighted.forward_safe_mode(residual,n_iter=n_iter,grad_coef=grad_coef)
    return mr_weighted