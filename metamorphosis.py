import torch
import matplotlib.pyplot as plt
import time
import warnings

import my_torchbox as tb
import vector_field_to_flow as vff
from my_toolbox import update_progress,format_time
from my_optim import GradientDescent
import reproducing_kernels as rk


class metamorphosis_path:
    """ Class integrating over a geodesic shooting. The user can choose the method among
    'Eulerian', 'advection_semiLagrangian' and 'semiLagrangian.

    """
    def __init__(self,method, mu=1,sigma_v= 1, n_step =10):
        """

        :param mu: Control parameter for geodesic shooting intensities changes
        For more details see eq. 3 of the article
        mu = 0 is LDDMM
        mu > 0 is metamorphosis
        :param sigma_v: sigma of the gaussian RKHS in the V norm
        :param n_step: number of time step in the segment [0,1]
         over the geodesic integration
        """
        self.mu = mu
        self.n_step = n_step

        # inner methods
        # kernel_size = max(6,int(sigma_v*6))
        # kernel_size += (1 - kernel_size %2)
        # self.kernelOperator = flt.GaussianBlur2d((kernel_size,kernel_size),
        #                                          (sigma_v, sigma_v),
        #                                          border_type='constant')
        self.kernelOperator = rk.GaussianRKHS2d((sigma_v, sigma_v),
                                                 border_type='constant')

        if method == 'Eulerian':
            self.step = self._step_fullEulerian
        elif method == 'advection_semiLagrangian':
            self.step = self._step_advection_semiLagrangian
        elif method == 'semiLagrangian':
            self.step = self._step_full_semiLagrangian
        else:
            raise ValueError("You have to specify the method used : 'Eulerian','advection_semiLagrangian'"
                             " or 'semiLagrangian'")

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

        :param residuals: (tensor array) of shape [H,W]
        :param grad_image: (tensor array) of shape [B,2,2,H,W]
        :return: (tensor array) of shape [B,H,W,2]
        """
        return tb.im2grid(self.kernelOperator(-residuals * grad_image[0]))

    def _step_fullEulerian(self):
        grad_image = tb.spacialGradient(self.image,dx_convention='pixel')
        self.field = self._compute_vectorField_(self.residuals,grad_image)

        residuals_dt = - tb.field_divergence(
                            self.residuals[None,:,:,None]* self.field,
                            dx_convention='pixel'
                            )[0,0]
        self.residuals = self.residuals + residuals_dt /self.n_step

        self.image =  self._image_Eulerian_integrator_(self.image,self.field,1/self.n_step,1)
        self.image = self.image + (self.residuals *self.mu)/ self.n_step

        return (self.image,self.field,self.residuals)

    def _step_advection_semiLagrangian(self):
        grad_image = tb.spacialGradient(self.image,dx_convention='pixel')
        self.field = self._compute_vectorField_(self.residuals,grad_image)

        # Eulerian scheme on residuals
        residuals_dt = - tb.field_divergence(
                            self.residuals[None,:,:,None]* self.field,
                            dx_convention='pixel')[0,0]
        self.residuals = self.residuals + residuals_dt /self.n_step

        # Lagrangian scheme on images
        deformation = self.id_grid - self.field/self.n_step
        self.image = tb.imgDeform(self.image,deformation,dx_convention='pixel') \
                     + (self.residuals *self.mu)/self.n_step

        return (self.image,self.field,self.residuals)

    def _step_full_semiLagrangian(self):
        grad_image = tb.spacialGradient(self.image,dx_convention='pixel')
        self.field = self._compute_vectorField_(self.residuals,grad_image)
        # Lagrangian scheme on images and residuals

        deformation = self.id_grid - self.field/self.n_step
        div_v_times_z = self.residuals*tb.field_divergence(self.field)[0,0]

        self.residuals = tb.imgDeform(self.residuals[None,None],
                                      deformation,
                                      dx_convention='pixel')[0,0]\
                         - div_v_times_z/self.n_step

        self.image = tb.imgDeform(self.image,deformation,dx_convention='pixel') +\
                     (self.mu*self.residuals )/self.n_step

        return (self.image,self.field,self.residuals)

    def forward(self,image,residual_ini,field_ini=None,save=True,plot =0,t_max = 1):
        r""" This method is doing the temporal loop using the good method `_step_`

        :param image: (tensor array) of shape [1,1,H,W]. Source image ($I_0$)
        :param field_ini: to be deprecated, field_ini is id_grid
        :param residual_ini: (tensor array) of shape [H,W]. initial residual ($z_0$)
        :param save: (bool) option to save the integration intermediary steps.
        :param plot: (int) positive int lower than `self.n_step` to plot the indicated
                         number of intemediary steps. (if plot>0, save is set to True)

        """
        self.source = image.detach()
        self.image = image.clone()
        self.residuals = residual_ini
        self.save = save

        # TODO: probablement à supprimer
        H,W = residual_ini.shape
        self.id_grid = tb.make_regular_grid((1,H,W,2))
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

        self.t = 0
        for i,t in enumerate(torch.linspace(0,t_max,t_max*self.n_step)):
            self.t = t

            _,field_to_stock,residuals_dt = self.step()

            if self.image.isnan().any() or self.residuals.isnan().any():
                raise OverflowError("Some nan where produced ! the integration diverged",
                                    "changing the parameters is needed (decreasing n_step can help) ")

            if self.save:
                self.image_stock[i] = self.image[0].detach()
                self.field_stock[i] = field_to_stock[0].detach()
                self.residuals_stock[i] = residuals_dt.detach()

        if plot>0:
            self.plot(n_figs=plot)

        #return self.residuals

    def get_deformation(self,n_step=-1):
        r"""Returns the deformation use it for showing results
        $\Phi = \int_0^1 v_t dt$

        :return: deformation [1,H,W,2] or [2,H,W,D,3]
        """
        temporal_integrator = vff.FieldIntegrator(method='temporal',save=False)
        return temporal_integrator(self.field_stock[:n_step]/self.n_step,forward=True)

    def get_deformator(self):
        r"""Returns the inverde deformation use it for deforming images
        $\Phi^{-1}$

        :return: deformation [1,H,W,2] or [2,H,W,D,3]
        """
        temporal_integrator = vff.FieldIntegrator(method='temporal',save=False)
        return temporal_integrator(self.field_stock/self.n_step,forward=False)

    # ==================================================================
    #                       PLOTS
    # ==================================================================

    def plot(self,n_figs=5):

        plot_id = torch.quantile(torch.arange(self.image_stock.shape[0],dtype=torch.float),
                                 torch.linspace(0,1,n_figs)).round().int()


        kw_image_args = dict(cmap='gray',
                      extent=[-1,1,-1,1],
                      origin='lower',
                      vmin=0,vmax=1)
        kw_residuals_args = dict(cmap='RdYlBu_r',
                      extent=[-1,1,-1,1],
                      origin='lower',
                      vmin=self.residuals_stock.min(),
                      vmax=self.residuals_stock.max())
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

            tb.gridDef_plot(self.field_stock[t].unsqueeze(0),
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

        tb.gridDef_plot(full_deformation,step=int(max(self.image.shape)/40),ax = axes[0,0],
                         check_diffeo=True,dx_convention='2square')
        tb.quiver_plot(full_deformation -self.id_grid ,step=int(max(self.image.shape)/40),
                        ax = axes[0,1],check_diffeo=False)

        # show S deformed by full_deformation
        S_deformed = tb.imgDeform(self.source,full_deformator,dx_convention='pixel')
        axes[1,0].imshow(self.source[0,0,:,:],cmap='gray',origin='lower',vmin=0,vmax=1)
        axes[1,1].imshow(tb.imCmp(target,self.source),origin='lower',vmin=0,vmax=1)
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


# =======================================================================
# =======================================================================
#  OPTIMISATION
# =======================================================================
# =======================================================================

class optimize_geodesicShooting:

    def __init__(self,source : torch.Tensor,
                     target : torch.Tensor,
                     geodesic : metamorphosis_path,
                     cost_cst : dict,
                     optimizer_method : str = 'grad_descent',
                     cost = None
                 ):
        self.mp = geodesic
        self.source = source
        self.target = target
        self.cost_cst = cost_cst
        self.optimizer_method_name = optimizer_method #for __repr__
        # optimize on the cost as defined in the 2021 paper.
        if cost is None:
            self.cost = self.metamorphosis_cost

        # forward function choice among developed optimizers
        if optimizer_method == 'grad_descent':
            self._initialize_optimizer_ = self._initialize_grad_descent_
            self._step_optimizer_ = self._step_grad_descent_
        elif optimizer_method == 'LBFGS_torch':
            self._initialize_optimizer_ = self._initialize_LBFGS_
            self._step_optimizer_ = self._step_LBFGS_
        else:
            raise ValueError(
                "\noptimizer_method is " + optimizer_method +
                "You have to specify the optimizer_method used among"
                "{'grad_descent', 'LBFGS_torch'}"
                             )

        self.temporal_integrator = vff.FieldIntegrator(method='temporal',save=False)

    def __repr__(self) -> str:
        return self.__class__.__name__ +\
            '(cost_parameters : {mu ='+ str(self.mp.mu)     +\
            ', lambda,rho =' +   str(self.cost_cst)    +'},'+\
            '\nintegration method '+      self.mp.step.__name__ +\
            '\noptimisation method '+ self.optimizer_method_name+\
            '\n# geodesic steps =' +      str(self.mp.n_step) + ')'


    def metamorphosis_cost(self, residuals_ini : torch.Tensor) -> torch.Tensor:
        r""" cost computation

        $H(z_0) =   \frac 12\| \im{1} - \ti \|_{L_2}^2 + \lambda \Big[ \|v_0\|^2_V + \mu ^2 \|z_0\|^2_{L_2} \Big]$

        :param residuals_ini: z_0
        :return: $H(z_0)$ a single valued tensor
        """
        _,_,H,W = self.source.shape
        lamb, rho = self.cost_cst.values()
        if(self.mp.mu == 0 and rho != 0):
            warnings.warn("mu as been set to zero in methamorphosis_path, "
                          "automatic reset of rho to zero."
                         )
            rho = 0

        self.mp.forward(self.source.clone(),residuals_ini,save=False,plot=0)
        # checkpoint(self.mp.forward,self.source.clone(),self.id_grid,residuals_ini)

        self.ssd = .5*((self.target - self.mp.image)**2).sum()


        # Norm V
        grad_source = tb.spacialGradient(self.source)
        grad_source_resi = grad_source[0]*residuals_ini
        K_grad_source_resi = self.mp.kernelOperator(grad_source_resi)

        self.norm_v_2 = (grad_source_resi * K_grad_source_resi).sum()

        # norm_2 on z
        if self.mp.mu != 0:
            # # Norm on the residuals only
            self.norm_l2_on_z = (residuals_ini**2).sum()/(H*W)
            self.total_cost = self.ssd + \
                              lamb*(self.norm_v_2 + (rho) *self.norm_l2_on_z)
        else:
             self.total_cost = self.ssd + lamb*self.norm_v_2
        # print('ssd :',self.ssd,' norm_v :',self.norm_v_2)
        return self.total_cost

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #   Implemented OPTIMIZERS
    # GRADIENT DESCENT
    def _initialize_grad_descent_(self,dt_step,max_iter=20):
        self.optimizer = GradientDescent(self.cost,self.parameter,lr=dt_step)

    def _step_grad_descent_(self):
        self.optimizer.step()

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

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def forward(self,z_0,n_iter = 10,grad_coef = 1e-3,verbose= True) :
        r""" The function is an perform the optimisation with the desired method.
        The result is stored in the tuple self.analyze with two elements. First element is the optimized
        initial residual ($z_O$ in the article) used for the shooting.
        The second is a tensor with the values of the loss norms over time. The function
        plot_cost() is designed to show them automaticly.

        :param z_0: initial residual. It is the variable on which we optimise.
        `require_grad` must be set to True.
        :param n_iter: (int) number of optimizer iterations
        :param verbose: (bool) display advancement

        """
        def _default_cost_saving_(i):
            if self.mp.mu != 0: # metamophosis
                loss_stock[i,0] = self.ssd.detach()
                loss_stock[i,1] = self.norm_v_2.detach()
                loss_stock[i,2] = self.norm_l2_on_z.detach()
            else: #LDDMM
                loss_stock[i,0] = self.ssd.detach()
                loss_stock[i,1] = self.norm_v_2.detach()

        self.parameter = z_0 # optimized variable
        self._initialize_optimizer_(grad_coef,max_iter=n_iter)

        self.id_grid = tb.make_regular_grid(z_0.shape,z_0.device)

        self.cost(self.parameter)
        d = 3 if self.mp.mu != 0 else 2
        loss_stock = torch.zeros((n_iter,d))
        _default_cost_saving_(0)

        for i in range(1,n_iter):
            self._step_optimizer_()
            _default_cost_saving_(i)

            if verbose:
                update_progress((i+1)/n_iter)

        # for future plots compute shooting with save = True
        self.mp.forward(self.source.clone(),
                        self.parameter.detach().clone(),
                        save=True,plot=0)

        self.to_analyse = (self.parameter.detach(),loss_stock)

    # ==================================================================
    #                 PLOTS
    # ==================================================================

    def plot_cost(self):
        """ To display cost evolution during the optimisation.


        """
        plt.rcParams['figure.figsize'] = [10,10]
        fig1,ax1 = plt.subplots(1,2)

        lamb, rho = self.cost_cst.values()

        ssd_plot = self.to_analyse[1][:,0].numpy()
        ax1[0].plot(ssd_plot,"--",color = 'blue',label='ssd')
        ax1[1].plot(ssd_plot,"--",color = 'blue',label='ssd')

        normv_plot = lamb*self.to_analyse[1][:,1].detach().numpy()
        ax1[0].plot(normv_plot,"--",color = 'green',label='normv')
        ax1[1].plot(self.to_analyse[1][:,1].detach().numpy(),"--",color = 'green',label='normv')
        total_cost = ssd_plot +normv_plot
        if self.mp.mu != 0:
            norm_l2_on_z = lamb*(rho)* self.to_analyse[1][:,2].numpy()
            total_cost += norm_l2_on_z
            ax1[0].plot(norm_l2_on_z,"--",color = 'orange',label='norm_l2_on_z')
            ax1[1].plot(self.to_analyse[1][:,2].numpy(),"--",color = 'orange',label='norm_l2_on_z')

        ax1[0].plot(total_cost, color='black',label='\Sigma')
        ax1[0].legend()
        ax1[1].legend()
        ax1[0].set_title("Lambda = "+str(lamb)+
                    " mu = "+str(self.mp.mu) +
                    " rho = "+str(rho))

    def plot_imgCmp(self):
        r""" Display and compare the deformed image $I_1$ with the target$
        """
        plt.rcParams['figure.figsize'] = [20,10]
        fig,ax = plt.subplots(2,2)

        ax[0,0].imshow(self.source[0,0,:,:].detach().numpy(),
                       cmap='gray',origin='lower',vmin=0,vmax=1)
        ax[0,0].set_title("source")
        ax[0,1].imshow(self.target[0,0,:,:].detach().numpy(),
                       cmap='gray',origin='lower',vmin=0,vmax=1)
        ax[0,1].set_title("target")

        ax[1,0].imshow(tb.imCmp(self.target,self.mp.image.detach()),origin='lower',cmap='gray')
        ax[1,0].set_title("comparaison deformed image with target")
        ax[1,1].imshow(self.mp.image[0,0].detach().numpy(),origin='lower',cmap='gray',vmin=0,vmax=1)
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