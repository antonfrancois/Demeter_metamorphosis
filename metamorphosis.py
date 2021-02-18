import torch
from torch.utils.checkpoint import checkpoint # redondant mais necéssaire ...
import kornia.filters as flt
from math import log
import matplotlib.pyplot as plt
import time

import my_torchbox as tb
import vector_field_to_flow as vff
from my_toolbox import update_progress,format_time
from cost_functions import sumSquaredDifference
from my_optim import GradientDescent

class metamorphosis_path:
    """ Class integrating over a geodesic shooting. The user can choose the method among
    'Eulerian', 'advection_semiLagrangian' and 'semiLagrangian.

    """
    def __init__(self,method, mu=1,sigma_v= 1, delta_t =10,t_max =1):
        """

        :param mu: It is the parameter for the amont of metamorphosis
        mu = 0 is LDDMM
        mu = 1 is metamorphosis
        :param sigma_v: smooting sigma
        :param delta_t:
        """
        self.mu = mu
        self.delta_t = delta_t
        self.t_max=t_max

        # inner methods
        kernel_size = max(6,int(sigma_v*6))
        kernel_size += (1 - kernel_size %2)
        self.kernelOperator = flt.GaussianBlur2d((kernel_size,kernel_size),
                                                 (sigma_v, sigma_v),
                                                 border_type='constant')

        if method == 'Eulerian':
            self.step = self._step_fullEulerian
        elif method == 'advection_semiLagrangian':
            self.step = self._step_advection_semiLagrangian
            self.integrator = vff.FieldIntegrator(method='fast_exp')
        elif method == 'semiLagrangian':
            self.step = self._step_full_semiLagrangian
            self.integrator = vff.FieldIntegrator(method='fast_exp')
        else:
            raise ValueError("You have to specify the method used : 'Eulerian','advection_semiLagrangian'"
                             " or 'semiLagrangian'")

    def _image_Eulerian_integrator_(self,image,vector_field,t_max,n_step=None):

        dt = t_max/n_step
        for t in torch.linspace(0,t_max,n_step):
            grad_I = tb.spacialGradient(image,dx_convention='pixel')
            grad_I_scalar_v = (grad_I[0]*tb.grid2im(vector_field)).sum(dim=1)
            image = image - grad_I_scalar_v * dt
        return image

    def _compute_vectorField_(self,residuals,grad_image):
        return tb.im2grid(self.kernelOperator(-residuals * grad_image[0]))

    def _step_fullEulerian(self):
        grad_image = tb.spacialGradient(self.image,dx_convention='pixel')

        self.field = self._compute_vectorField_(self.residuals,grad_image)
        residuals_dt = - tb.field_divergence(
                            self.residuals[None,:,:,None]* self.field,
                            dx_convention='pixel'
                            )[0,0]

        # self.field = field_new
        self.residuals = self.residuals + residuals_dt /self.delta_t
        self.image =  self._image_Eulerian_integrator_(self.image,self.field,1/self.delta_t,1)
        # self.image = self.image + (self.residuals * self.mu**2)/ self.delta_t
        self.image = self.image + (self.residuals *self.mu**2)/ self.delta_t

        #seuillage entre 0 et 1
        # self.image = tb.thresholding(self.image)
#         #print('image grad = ',self.image.requires_grad,'nan ? =>',self.image.isnan().sum())
        return (self.image,self.field,self.residuals)

    def _step_advection_semiLagrangian(self):
        grad_image = tb.spacialGradient(self.image,dx_convention='pixel')
        self.field = self._compute_vectorField_(self.residuals,grad_image)        # Eulerian scheme
        residuals_dt = - tb.field_divergence(
                            self.residuals[None,:,:,None]* self.field,
                            dx_convention='pixel')[0,0]

        self.residuals = self.residuals + residuals_dt /self.delta_t

        # Lagrangian scheme on images and residuals
        # deformation = self.integrator(self.field.clone()/self.delta_t,
        #                               forward=False,verbose=False)
        deformation = self.id_grid - self.field/self.delta_t
        # self.image = tb.imgDeform(self.image,deformation) + (self.residuals * self.mu**2)/self.delta_t
        self.image = tb.imgDeform(self.image,deformation,dx_convention='pixel') \
                     + (self.residuals *self.mu**2)/self.delta_t
        #seuillage
        # self.image = tb.thresholding(self.image)
        return (self.image,self.field,self.residuals)

    def _step_full_semiLagrangian(self):
        grad_image = tb.spacialGradient(self.image,dx_convention='pixel')
        self.field = self._compute_vectorField_(self.residuals,grad_image)
        # Lagrangian scheme on images and residuals
        # deformation = self.integrator(self.field.clone(),forward=False,verbose=False)
        deformation = self.id_grid - self.field/self.delta_t
        div_v_times_z = self.residuals*tb.field_divergence(self.field)[0,0]

        self.residuals = tb.imgDeform(self.residuals[None,None],
                                      deformation,
                                      dx_convention='pixel')[0,0]\
                         - div_v_times_z/self.delta_t

        self.image = tb.imgDeform(self.image,deformation,dx_convention='pixel') +\
                     (self.residuals * self.mu**2)/self.delta_t

        #seuillage
        # self.image = tb.thresholding(self.image)
        # self.residuals = tb.thresholding(self.residuals,bounds=(-1,1))
        return (self.image,self.field,self.residuals)

    def forward(self,image,field_ini,residual_ini,save=True,plot =0):
        """ This method is doing the temporal loop using the good method 'step'

        :param image:
        :param field_ini: to be deprecated, field_ini is computed from I_0 and z_0
        :param residual_ini:
        :param save:
        :param plot:
        :return:
        """
        self.source = image.detach()
        self.image = image.clone()
        self.field = field_ini #/self.delta_t
        self.residuals = residual_ini
        self.save = save

        # TODO: probablement à supprimer
        self.id_grid = tb.make_regular_grid(field_ini.shape)

        if plot > 0:
            self.save = True

        if self.save:
            self.image_stock = torch.zeros((self.delta_t,)+image.shape[1:])
            self.field_stock = torch.zeros((self.delta_t,)+field_ini.shape[1:])
            self.residuals_stock = torch.zeros((self.delta_t,)+residual_ini.shape)
            # # iteration 0
            # self.image_stock[0] = image
            # self.field_stock[0] = field_ini
            # self.residuals_stock[0] = residual_ini

        self.t = 0

        for i,t in enumerate(torch.linspace(0,self.t_max,self.delta_t)):
            self.t = t

            # checkpoint for less ram usage with autograd
            _,field_to_stock,residuals_dt = self.step()

            if self.image.isnan().any() or self.residuals.isnan().any():
                raise OverflowError("Some nan where produced ! As there is no cheese",
                                    "change the values (increasing delta_t can help) ")

            if self.save:
                self.image_stock[i] = self.image[0].detach()
                self.field_stock[i] = field_to_stock[0].detach()
                self.residuals_stock[i] = residuals_dt.detach()

        if plot>0:
            self.plot(n_figs=plot)

        return self.residuals


    def plot(self,n_figs=5):

        plot_id = torch.quantile(torch.arange(self.delta_t,dtype=torch.float),
                                 torch.linspace(0,1,n_figs)).round().int()

        cmap = 'cividis'
        size_fig = 5
        plt.rcParams['figure.figsize'] = [size_fig*3,n_figs*size_fig]
        fig,ax = plt.subplots(n_figs,3)

        for i,t in enumerate(plot_id):
            i_s =ax[i,0].imshow(self.image_stock[t,0,:,:].detach().numpy(),
                                cmap=cmap,origin='lower',vmin=0,vmax=1)
            ax[i,0].set_title("t = "+str((t/(self.delta_t-1)).item())[:3])
            ax[i,0].axis('off')
            fig.colorbar(i_s,ax=ax[i,0],fraction=0.046, pad=0.04)

            r_s =ax[i,1].imshow(self.residuals_stock[t].detach().numpy(),
                                cmap=cmap,origin='lower')
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
        print('field_stock ',self.field_stock.shape)

        # field_stock_toplot = tb.pixel2square_convention(self.field_stock)
        # tb.gridDef_plot(field_stock_toplot[-1][None],dx_convention='2square')
        if temporal:
            full_deformation_t = temporal_integrator(self.field_stock/self.delta_t,
                                                     forward=True)
            full_deformator_t = temporal_integrator(self.field_stock/self.delta_t,
                                                    forward=False)
            full_deformation = full_deformation_t[-1].unsqueeze(0)
            full_deformator = full_deformator_t[-1].unsqueeze(0)
        else:
            full_deformation = temporal_integrator(self.field_stock/self.delta_t,
                                                   forward=True)
            full_deformator = temporal_integrator(self.field_stock/self.delta_t,
                                                  forward=False)

        plt.rcParams['figure.figsize'] = [20,30]
        fig , axes = plt.subplots(3,2)
        # show resulting deformation

        tb.gridDef_plot(full_deformation,step=int(max(self.image.shape)/40),ax = axes[0,0],
                         check_diffeo=True,dx_convention='2square')
        tb.quiver_plot(full_deformation -self.id_grid ,step=int(max(self.image.shape)/40),
                        ax = axes[0,1],check_diffeo=False)

        # show S deformed by full_deformation
        # tb.quiver_plot(full_deformator-regular_grid,step=7,ax =axes[1],color='red')
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

                # ax[0].imshow(tb.imgDeform(S,full_deformator_t[t].unsqueeze(0))[0,0,:,:],extent=[-1,1,-1,1],cmap='gray')
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


class grad_descent_metamorphosis:

    def __init__(self,source,target,geodesic,cost_l=1,grad_dt=1e-3):
        self.mp = geodesic
        self.source = source
        self.target = target
        # self.ssd = sumSquaredDifference(self.source,target)
        self.lr = grad_dt
        self.cost_l = cost_l

        self.temporal_integrator = vff.FieldIntegrator(method='temporal',save=False)


    def cost(self,residuals_ini):
        """ cost computation

        :param residuals_ini: z_0
        :return:
        """
        _,H,W,_ = self.source.shape

        self.mp.forward(self.source.clone(),self.id_grid,residuals_ini,save=False,plot=0)
        # checkpoint(self.mp.forward,self.source.clone(),self.id_grid,residuals_ini)

        # full_deformator = self.temporal_integrator(self.mp.field_stock,forward=False)
        # im_deform = tb.imgDeform(self.source.clone(),full_deformator)
        # self.ssd = .5*((self.target - im_deform)**2).sum()

        self.ssd = .5*((self.target - self.mp.image)**2).sum()

        # Norm V
        grad_source = tb.spacialGradient(self.source)

        grad_source_resi = grad_source[0]*residuals_ini
        K_grad_source_resi = self.mp.kernelOperator(grad_source_resi)

        self.norm_v_2 = (grad_source_resi * K_grad_source_resi).sum()

        # norm_2 on z
        if self.mp.mu != 0:
            # # Norm on the residuals only
            self.norm_l2_on_z = (residuals_ini**2).sum()
            self.total_cost = self.ssd + \
                              self.cost_l*(self.norm_v_2 + (self.mp.mu**2) *self.norm_l2_on_z)
        else:
             self.total_cost = self.ssd + self.cost_l*self.norm_v_2
        return self.total_cost



    def forward(self,x_0,n_iter = 10,verbose= True,gamma=None) :
        """ Boucle de pas de descente avec à chaque tour de boucle un appel à la fonction cost()

        :param x_0:
        :param n_iter:
        :return:
        """
        gd = GradientDescent(self.cost,x_0,lr=self.lr,gamma=gamma)
        self.id_grid = tb.make_regular_grid(x_0.shape,x_0.device)

        self.cost(x_0)
        if self.mp.mu != 0: # metamophosis
            loss_stock = torch.zeros((n_iter,3))
            loss_stock[0,0] = self.ssd.detach()
            loss_stock[0,1] = self.norm_v_2.detach()
            loss_stock[0,2] = self.norm_l2_on_z.detach()
        else: #LDDMM
            loss_stock = torch.zeros((n_iter,2))
            loss_stock[0,0] = self.ssd.detach()
            loss_stock[0,1] = self.norm_v_2.detach()


        for i in range(1,n_iter):
            gd.step()

            if self.mp.mu != 0: # metamorphoses
                loss_stock[i,0] = self.ssd.detach()
                loss_stock[i,1] = self.norm_v_2.detach()
                loss_stock[i,2] = self.norm_l2_on_z.detach()
            else: # LDDMM
                loss_stock[i,0] = self.ssd.detach()
                loss_stock[i,1] = self.norm_v_2.detach()

            if verbose:
                update_progress((i+1)/n_iter)

        # for future plots compute shooting with save = True
        self.mp.forward(self.source.clone(),self.id_grid,gd.x.detach().clone(),save=True,plot=0)

        self.to_analyse = (gd.x.detach(),loss_stock)


    # ==================================================================
    #                 PLOTS
    # ==================================================================

    def plot_cost(self):
        plt.rcParams['figure.figsize'] = [10,10]
        fig1,ax1 = plt.subplots()

        ssd_plot = self.to_analyse[1][:,0].numpy()
        ax1.plot(ssd_plot,"--",color = 'blue',label='ssd')

        normv_plot = self.cost_l*self.to_analyse[1][:,1].detach().numpy()
        ax1.plot(normv_plot,"--",color = 'green',label='normv')
        total_cost = ssd_plot +normv_plot
        if self.mp.mu != 0:
            norm_l2_on_z = self.cost_l*(self.mp.mu**2)* self.to_analyse[1][:,2].numpy()
            total_cost += norm_l2_on_z
            ax1.plot(norm_l2_on_z,"--",color = 'orange',label='norm_l2_on_z')

        ax1.plot(total_cost, color='black',label='\Sigma')
        ax1.legend()

    def plot_imgCmp(self):
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
        self.mp.forward(self.source.clone(),self.id_grid,residuals,save=True,plot=0)
        self.mp.plot_deform(self.target,temporal_nfigs)

    def plot(self):
        _,_,H,W = self.source.shape

        start = time.time()
        self.plot_cost()
        check_1 = time.time()
        print('\n time for plot ============ ')
        print("plot 1 ", format_time(check_1 - start))
        self.plot_imgCmp()
        check_2 = time.time()
        print("plot 2 ", format_time(check_2 - start))
        # self.plot_deform()
        end = time.time()
        print("plot 3 ", format_time(end - start))



