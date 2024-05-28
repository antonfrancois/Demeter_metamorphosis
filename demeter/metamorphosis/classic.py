import torch
import warnings
import matplotlib.pyplot as plt
from math import prod

from metamorphosis import Geodesic_integrator,Optimize_geodesicShooting

from utils.constants import *
import utils.torchbox as tb

class Metamorphosis_integrator(Geodesic_integrator):
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

        return (self.image,self.field,self.momentum)

    def _step_advection_semiLagrangian(self):
        self._update_field_()
        # Lagrangian scheme on images
        deformation = self.id_grid - self.field/self.n_step
        self._update_image_semiLagrangian_(deformation)
        self._update_residuals_Eulerian_()

        return (self.image,self.field,self.momentum)

    def _step_full_semiLagrangian(self):
        self._update_field_()
        # Lagrangian scheme on images and residuals
        deformation = self.id_grid - self.field/self.n_step
        self._update_image_semiLagrangian_(deformation)
        self._update_residuals_semiLagrangian_(deformation)

        return (self.image,self.field,self.momentum)

    def _step_sharp_semiLagrangian(self):
        # device = self.image.device
        self._update_field_()
        self._update_sharp_intermediary_field_()

        resi_cumul = 0
        if self._get_mu_() != 0:
            resi_cumul = self._compute_sharp_intermediary_residuals_()
            resi_cumul = resi_cumul.to(self.momentum.device)
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
                fig,ax = plt.subplots(1,3,figsize =(15,5))
                aa = ax[0].imshow(self.momentum.detach()[0,0], **DLT_KW_RESIDUALS)
                fig.colorbar(aa,ax=ax[0],fraction=0.046, pad=0.04)

                # self._update_residuals_semiLagrangian_(phi_n_n)
                ab = ax[1].imshow(resi_cumul.detach()[0,0],**DLT_KW_RESIDUALS)
                ax[1].set_title("new residual")
                fig.colorbar(ab,ax=ax[1],fraction=0.046, pad=0.04)

                ax[2].imshow(self.image.detach()[0,0],**DLT_KW_IMAGE)

                #
            plt.show()
        # aaa = input('Appuyez sur entree')
        # # End DEBUG ===================================
        self._update_image_semiLagrangian_(self._phis[self._i][0],resi_cumul,
                                           sharp=True)
        # self.image = tb.imgDeform(self.source.to(self.image.device),
        #                           self._phis[self._i][0],
        #                           dx_convention='pixel')
        # if self.mu != 0: self.image += self.mu * resi_cumul/self.n_step
        self._update_residuals_semiLagrangian_(self._phis[self._i][self._i])

        return (self.image, self.field, self.momentum)

    def step(self):
        raise ValueError("You have to specify the method used : 'Eulerian','advection_semiLagrangian'"
                             " or 'semiLagrangian'")

    def _get_mu_(self):
        return self.mu

    def _get_rho_(self):
        return float(self.rho)

class Metamorphosis_Shooting(Optimize_geodesicShooting):

    def __init__(self,source : torch.Tensor,
                     target : torch.Tensor,
                     geodesic : Metamorphosis_integrator,
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

    # def _compute_V_norm_(self,*args):
    #     """
    #
    #     usage 1: _compute_V_norm_(field)
    #         :field: torch Tensor of shape [1,H,W,2] or [1,D,H,W,3]
    #     usage 2: _compute_V_norm_(residual, image)
    #         :residual: torch Tensor of shape [1,C,H,W] or [1,C,D,H,W]
    #         :image: torch Tensor of shape [1,C,H,W] or [1,C,D,H,W]
    #     :return: float
    #     """
    #     if len(args) == 2 and not args[0].shape[-1] in [2,3] :
    #         residual, image = args[0],args[1]
    #         C = residual.shape[1]
    #         grad_source = tb.spacialGradient(image)
    #         grad_source_resi = (grad_source * residual.unsqueeze(2)).sum(dim=1) / C
    #         K_grad_source_resi = self.mp.kernelOperator(grad_source_resi)
    #
    #         return (grad_source_resi * K_grad_source_resi).sum()
    #     elif len(args) == 1 and args[0].shape[-1] in [2,3]:
    #         field = args[0]
    #         k_field = self.mp.kernelOperator(field)
    #         return (k_field * field).sum()
    #     else:
    #         raise ValueError(f"Bad arguments, see usage in Doc got args = {args}")


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
