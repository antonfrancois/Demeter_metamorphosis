r"""
This module contains the classical Metamorphosis algorithm. Defined as follows:

We define the Hamiltonian:

.. math::
    H(I,p,v,z) = - (p |\dot{ I}) - \frac{1}{2} (Lv|v)_{2} - \frac{1}{2}(z,z)_{2}.

By calculating the optimal trajectories we obtain the system:

.. math::
    \left\{ 	\begin{array}{rl} v &= - \sqrt{ \rho } K_{V} (p \nabla I)\\ \dot{p} &= -\sqrt{ \rho } \nabla \cdot (pv) \\ z &= \sqrt{ 1 - \rho } p \\  \dot{I} &=  - \sqrt{ \rho } v_{t} \cdot \nabla I_{t} + \sqrt{ 1-\rho } z.\end{array} \right.

In practice we minimize the Energy function:

.. math::
    E(p_0) = D(I_1,T) + \lambda \Big[ \|v_0\|^2_V + \|z_0\|^2_{L_2} \Big]

with $D(I_1,T)$ the data term given by the user, $\|v_0\|^2_V = (Lv,v)_2$ the RKHS norm of the velocity field parametrized
by L^{-1} = K_\sigma$  that we call the kernel operator and $\|z_0\|^2_{L_2}$ the $L_2$ norm of the momentum field.

"""



import torch
from math import prod, sqrt

from .abstract import Geodesic_integrator,Optimize_geodesicShooting

from demeter.constants import *
from ..utils import torchbox as tb

class Metamorphosis_integrator(Geodesic_integrator):
    r""" Class integrating over a geodesic shooting. The user can choose the method among
    'Eulerian', 'advection_semiLagrangian' and 'semiLagrangian.

    :param rho: (float) $\rho \in [0,1]$  Control parameter for geodesic shooting intensities changes
        $rho = 1$ is LDDMM
        $mu = 0$ is pure photomeric changes and any value in between is a mix of both.
    :param sigma_v: sigma of the gaussian RKHS in the V norm
    :param n_step: number of time step in the segment [0,1]
         over the geodesic integration
    """
    def __init__(self,method,
                 rho=1.,
                 # sigma_v= (1,1,1),
                 # multiScale_average=False,
                 **kwargs
                 ):

        # print('Before super() my weight is ',get_size(self))
        super().__init__(**kwargs)
        # self.mu = mu if callable(mu) else lambda :mu
        # self.rho = rho if callable(rho) else lambda :rho
        if rho < 0 or rho > 1:
            raise ValueError("This is the new version of Metamorphosis, rho must be in [0,1]")
        self.rho = rho
        # self.n_step = n_step

        # inner methods

        # self._force_save = False
        sharp = False
        self.method = method
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
        self.field *= sqrt(self.rho)
        ic(self.rho,self.field.min().item(), self.field.max().item())
        self._update_image_Eulerian_()
        # self._update_residuals_Eulerian_()
        self._update_momentum_Eulerian_()

        return (self.image,self.field,self.momentum)

    def _step_advection_semiLagrangian(self):
        self._update_field_()
        # Lagrangian scheme on images
        deformation = self.id_grid - self.rho * self.field/self.n_step
        self._update_image_semiLagrangian_(deformation)
        self._update_momentum_Eulerian_()

        return (self.image,self.rho * self.field,(1 - self.rho) * self.momentum)

    def _step_full_semiLagrangian(self):
        self._update_field_()
        # Lagrangian scheme on images and residuals
        deformation = self.id_grid - self.rho * self.field/self.n_step
        self._update_image_semiLagrangian_(deformation)

        self.momentum = self._compute_div_momentum_semiLagrangian_(
            deformation,
            self.momentum,
            sqrt(self.rho),
            sqrt(self.rho) * self.field
        )
        # self.momentum *= sqrt(self.rho)

        return (self.image,self.rho * self.field,(1 - self.rho) * self.momentum)

    def _step_sharp_semiLagrangian(self):
        # device = self.image.device
        self._update_field_()
        self._update_sharp_intermediary_field_()

        resi_cumul = 0
        if self._get_rho_() < 1:
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
                                          dx_convention=self.dx_convention)[0,0]
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
        #                           dx_convention=self.dx_convention)
        # if self.mu != 0: self.image += self.mu * resi_cumul/self.n_step
        self._update_momentum_semiLagrangian_(self._phis[self._i][self._i])

        return (self.image, self.field, self.momentum)

    def step(self):
        raise ValueError("You have to specify the method used : 'Eulerian','advection_semiLagrangian'"
                             " or 'semiLagrangian'")

    def _get_rho_(self):
        return float(self.rho)

class Metamorphosis_Shooting(Optimize_geodesicShooting):
    r"""
    Metamorphosis shooting class, implements the shooting algorithm for metamorphosis
    over the energy function:

    .. math::
        E(p_0) =   D( I_1,T)  + \lambda \Big[ \|v_0\|^2_V + \mu ^2 \|z_0\|^2_{L_2} \Big]


    with:
        - $D(I_1,T)$ the data term given by the user.
        - I_1 is obtained by integrating the geodesics over the geodesic equation using the provided integrator class.
        - $\|v_0\|^2_V = (Lv,v)_2$ the RKHS norm of the velocity field parametrized by L^{-1} = K_\sigma$  that we call the kernel operator
        - $\|z_0\|^2_{L_2} =\frac{1}{\# \Omega} \sum z_0^2$ the $L_2$ norm of the momentum field, with $\# \Omega$ the number of pixels in the image.

    Parameters
    ----------
    source : torch.Tensor
        source image
    target : torch.Tensor
        target image
    geodesic : Metamorphosis_integrator
        geodesic integrator
    cost_cst : float
        cost constant
    data_term : mt.DataCost, optional
        data term, by default Ssd
    optimizer_method : str, optional
        optimizer method, by default 'LBFGS_torch'
    hamiltonian_integration : bool, optional
        choose to integrate over first time step only or whole hamiltonian
         integration, by default False
    """

    def __init__(self,source,target,geodesic,**kwargs):
        # super().__init__(source,target,geodesic,cost_cst,data_term,optimizer_method,hamiltonian_integration,**kwargs)
        super().__init__(source,target,geodesic,**kwargs)


    # def __repr__(self) -> str:
    #     return self.__class__.__name__ +\
    #         '(cost_parameters : {mu ='+ str(self._get_mu_())     +\
    #         ', rho ='+str(self._get_rho_()) +\
    #         ', lambda =' + str(self.cost_cst)+'},'+\
    #         '\ngeodesic integrator : '+ self.mp.__repr__()+\
    #         '\nintegration method : '+      self.mp.step.__name__ +\
    #         '\noptimisation method : '+ self.optimizer_method_name+\
    #         '\n# geodesic steps =' +      str(self.mp.n_step) + ')'

    def _get_rho_(self):
        return float(self.mp.rho)

    def get_all_arguments(self):
        params_all  = super().get_all_arguments()
        params_spe = {
            'rho':self._get_rho_(),
            'method':self.mp.method,
        }
        return {**params_all,**params_spe}


    def cost(self, momentum_ini : torch.Tensor) -> torch.Tensor:
        r"""
        cost computation

        :param momentum_ini: Moment initial p_0
        :return: :math:`H(z_0)` a single valued tensor
        """
        lamb = self.cost_cst
        self.mp.forward(self.source,momentum_ini,
                        save=False,
                        plot=0,
                        hamiltonian_integration=self.flag_hamiltonian_integration,
                        )

        # Compute the data_term. Default is the Ssd
        self.data_loss = self.data_term()



        # norm_2 on z
        if self.flag_hamiltonian_integration:
            self.norm_v_2 = self.mp.norm_v
            self.norm_l2_on_z = self.mp.norm_z
            # self.total_cost = self.data_loss + lamb * (self.norm_v_2 + self.norm_l2_on_z)
        else:
            # Norm V
            self.norm_v_2 = .5 * self._compute_V_norm_(momentum_ini,self.source)

            # # Norm on the residuals only
            self.norm_l2_on_z = .5 * (momentum_ini**2).sum()/prod(self.source.shape[2:])

        self.total_cost = self.data_loss + \
                              lamb * (self.norm_v_2 + self.norm_l2_on_z)
        # print('ssd :',self.ssd,' norm_v :',self.norm_v_2)
        return self.total_cost
