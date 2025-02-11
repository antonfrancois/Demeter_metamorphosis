r"""
This module contains the classes for the constrained metamorphosis framework.
The constrained metamorphosis framework is a generalisation of the metamorphosis framework
that allows to add two types of constraints:
- One on the residual by controlling locally the amount of deformation versus the amount of photometric changes at given pixels.
- One on the field by adding a prior orienting field that will guide the deformation.
"""


from math import prod


from .abstract import Geodesic_integrator,Optimize_geodesicShooting

from demeter.constants import *
import demeter.utils. torchbox as tb
import  demeter.utils.cost_functions as cf


# ==========================================================================
#
#  Constrained optimisation
#
#===========================================================================


class ConstrainedMetamorphosis_integrator(Geodesic_integrator):
    r"""
    This class is an extension of the Geodesic_integrator class. It allows to
    integrate a geodesic with a constraint on the residual and the field using prior
    masks and fields.

    priors one can add are:
    - a residual mask: $M: \Omega \times t \rightarrow [0,1]$ a mask that for each time and pixel will control
    the amount of deformation versus the amount of photometric changes.
    - an orienting field: $w: \Omega \times t \rightarrow \mathbb{R}^d$ a field that will guide the deformation
    - an orienting mask: $Q: \Omega \times t \rightarrow [0,1]$ a mask that will control the
    prevalence of the orienting field over the deformation field.
    The geodesic equation is then given by:

    .. math::

        \left\{
            \begin{array}{rl}
                 v_{t} &= - K_{V} (\sqrt{ M_{t} } p_{t} \nabla I_{t} + Q_{t} w_{t})\\
                 \dot{p_{t}} &= -\sqrt{ M_t } \nabla \cdot (p_{t}v_{t}) \\
                 z_{t} &= \sqrt{ 1 - M_t } p_{t} \\
                 \dot{I_{t}} &=  - \sqrt{ M_t } v_{t} \cdot \nabla I_{t} + \sqrt{ 1-M_t } z_{t}.
            \end{array}
        \right.
    """

    def __init__(self, residual_mask: torch.Tensor  | int | float,
                 orienting_field: torch.Tensor = None,
                 orienting_mask: torch.Tensor = None,
                 sharp = False,
                 **kwargs
    ):
        super(ConstrainedMetamorphosis_integrator, self).__init__(n_step = None,**kwargs)
        if orienting_field is None:
            print("not oriented")
            self.orienting_mask, self.orienting_field = None, None
            self.flag_O = False
        else:
            print("oriented")
            self.flag_O = True
            self.orienting_mask = orienting_mask
            self.orienting_field = orienting_field
            self.n_step = orienting_mask.shape[0]
        if residual_mask is None:
            raise ValueError("You must set a residual_mask")
        if (isinstance(residual_mask,int | float)):
            self.flag_W = False
            if not self.flag_O:
                raise ValueError("You did not set any mask, did you want to use classical Metamorphosis ?")
            if isinstance(residual_mask,int) or isinstance(residual_mask,float):
                rho = residual_mask
            else: rho =1
            self.residual_mask = torch.ones((self.n_step,1)) * rho

            print("not Weighted")
        else:
            print("Weighted")
            self.residual_mask = residual_mask
            self.n_step = residual_mask.shape[0]
            self.flag_W = True

        # Verify that masks and have the same size
        if self.flag_O and self.flag_W:
            if orienting_mask.shape != residual_mask.shape:
                raise ValueError(f"orienting_mask and the residual_mask must have same size got (resp.):"
                                 f"{orienting_mask.shape},{residual_mask.shape}"
                                 )
            if orienting_mask.shape[0] != orienting_field.shape[0]:
                raise ValueError(f"orienting_field and orienting_mask must have same size in temporal dimension got (resp.):"
                                 f"{orienting_field.shape},{orienting_mask.shape}"
                                 )

        self._init_sharp_(sharp)

        # self.n_step = n_step

    def to_device(self,device):
        self.residual_mask = self.residual_mask.to(device)
        super().to_device(device)


    def __repr__(self):
        if self.flag_W and self.flag_O:
            mode = 'Weighted & Oriented'
        elif self.flag_W:
            mode = 'weighted'
        elif self.flag_O:
            mode = 'oriented'
        else:
            mode = 'why are you using this class ?'

        param = f"n_step = {self.n_step}"

        return self.__class__.__name__ + f"({mode})\n" \
                                         f"\t {param}\n\t{self.kernelOperator.__repr__()} "

    def _get_rho_(self):
        return 0

    def step(self):
        mask = self.residual_mask[self._i,0]

        if self.flag_O or self.flag_W:
            self._update_field_oriented_weighted_()
        else:
            self._update_field_()
            self.field *=  torch.sqrt(mask[...,None])
        assert self.field.isnan().any() == False, f"iter: {self._i}, field is nan"


        deform = (self.id_grid - torch.sqrt(mask)[...,None]* self.field / self.n_step)
        resi_to_add = (1 - mask) * self.momentum

        self._update_image_weighted_semiLagrangian_(deform,resi_to_add,sharp=False)

        assert self.image.isnan().any() == False, f"iter: {self._i}, image is nan"
        # self._update_momentum_semiLagrangian_(deform)
        self.momentum  = self._compute_div_momentum_semiLagrangian_(
            deform,
            self.momentum,
            torch.sqrt(mask)
        )
        assert self.momentum.isnan().any() == False, f"iter: {self._i}, momentum is nan"

        return (self.image,
                 torch.sqrt(mask)[...,None]*self.field,
                resi_to_add)


    def step_deprecated(self):
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
                # resi_to_add = self.kernelOperator(self._compute_sharp_intermediary_residuals_())
            if self._i > 0: self._phis[self._i - 1] = None
        else:
            def_z = self.id_grid - self.field / self.n_step
            def_I = def_z
            resi_to_add = self.momentum

        if self.flag_W:
            self._update_image_weighted_semiLagrangian_(def_I,resi_to_add,sharp=self.flag_sharp)
            self._update_momentum_weighted_semiLagrangian_(def_z)

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
            self._update_momentum_semiLagrangian_(def_z)

        return (self.image, self.field, self.momentum)

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



class ConstrainedMetamorphosis_Shooting(Optimize_geodesicShooting):
    r"""
    This class is an extension of the Optimize_geodesicShooting class. It allows to
    optimise a geodesic with constraints on the residual and the field using prior
    masks and fields.

    It implements and optiimise over the following cost function:
    $$H(I,p,v,z) = D(I_1,T) - \frac{1}{2} (Lv|v)_{2} - \frac{1}{2}|z|_{2} - \langle v,Qw\rangle_{2}. $$

    with:
        - $D(I_1,T)$ the data term given by the user.
        - I_1 is obtained by integrating the geodesics over the geodesic equation using the provided integrator class.
        - $\|v_0\|^2_V = (Lv,v)_2$ the RKHS norm of the velocity field parametrized by $L^{-1} = K_\sigma$  that we call the kernel operator
        - $\|z_0\|^2_{L_2} =\frac{1}{\# \Omega} \sum z_0^2$ the $L_2$ norm of the momentum field, with $\# \Omega$ the number of pixels in the image.
        - $\langle v,Qw\rangle_{2} = \frac{1}{\# \Omega} \sum v \cdot Qw$ the scalar product between the velocity field $v$ and the orienting field $w$ weighted by the orienting mask $Q$.

    """

    def __init__(self, source, target, geodesic, cost_cst,data_term=None, optimizer_method='adadelta',**kwargs):
        super().__init__(source, target, geodesic, cost_cst,data_term, optimizer_method)
        self._cost_saving_ = self._oriented_cost_saving_

    def _get_rho_(self):
        return "Weighted"

    def get_all_arguments(self):
        params_all  = super().get_all_arguments()
        params_const = {
             'sharp':self.mp.flag_sharp
        }
        return {**params_all,**params_const}

    # TODO : change cost saving to fill a dictionnary
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

        loss_stock[i, 2] = self.norm_l2_on_z.detach()
        if self.mp.flag_O:
            loss_stock[i, 3] = self.scaprod_v_w.detach()


        return loss_stock

    def get_total_cost(self):

        total_cost = self.to_analyse[1][:,0] + \
                    self.cost_cst * self.to_analyse[1][:,1]
        if self._get_mu_() != 0 :
            total_cost += self.cost_cst*(self._get_rho_())* self.to_analyse[1][:,2]
        if self.mp.flag_O:
            total_cost += self.cost_cst*(self._get_gamma_())*self.to_analyse[1][:,3]

        return total_cost

    def cost(self, momentum_ini: torch.Tensor) -> torch.Tensor:
        lamb = self.cost_cst

        self.mp.forward(self.source, momentum_ini,
                        save=False,
                        plot=0,
                        # hamiltonian_integration=self.flag_hamiltonian_integration
                        )

        # Compute the data_term. Default is the Ssd
        self.data_loss = self.data_term()

        # Norm V
        grad_source = tb.spatialGradient(self.source, dx_convention=self.dx_convention)
        field_momentum = (grad_source * momentum_ini.unsqueeze(2)).sum(dim=1) #/ C
        field = self.mp.kernelOperator(field_momentum)

        self.norm_v_2 = .5 * (field_momentum * field).sum() #* prod(self.dx)

        # Norm on the residuals only
        self.norm_l2_on_z = .5 * (momentum_ini**2).sum()/prod(self.source.shape[2:])


        self.total_cost = (self.data_loss +
                           lamb * (
                                   self.norm_v_2
                                   + self.norm_l2_on_z
                        )
        )
        # Oriented field norm
        if self.mp.flag_O:
            self.scaprod_v_w = (tb.im2grid(field) * self.mp.orienting_field[0][None]).sum(dim=-1)
            self.scaprod_v_w *= self.mp.orienting_mask[0]
            self.scaprod_v_w = self.scaprod_v_w.sum()#/prod(self.source.shape[2:])

            self.total_cost += lamb * self.scaprod_v_w

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
            self.mp.residual_mask.to(z_0.device)
        if self.mp.flag_O:
            self.mp.orienting_mask = self.mp.orienting_mask.to(z_0.device)
            self.mp.orienting_field = self.mp.orienting_field.to(z_0.device)
        super(ConstrainedMetamorphosis_Shooting, self).forward(z_0, n_iter, grad_coef, verbose,plot)
        self.to_device('cpu')

    def to_device(self, device):
        # if self.mp.flag_W:
        self.mp.residual_mask = self.mp.residual_mask.to(device)
        if self.mp.flag_O:
            self.mp.orienting_mask = self.mp.orienting_mask.to(device)
            self.mp.orienting_field = self.mp.orienting_field.to(device)
        super(ConstrainedMetamorphosis_Shooting, self).to_device(device)

    def plot_cost(self,y_log=False):
        fig1, ax1 = plt.subplots(1, 2,figsize=(10,10))
        if y_log:
            ax1[0].set_yscale('log')
            ax1[1].set_yscale('log')
        cost_stock = self.to_analyse[1].detach().numpy()

        ssd_plot = cost_stock[:, 0]
        ax1[0].plot(ssd_plot, "--", color='blue', label='ssd')
        ax1[1].plot(ssd_plot, "--", color='blue', label='ssd')

        normv_plot = self.cost_cst * cost_stock[:, 1]
        ax1[0].plot(normv_plot, "--", color='green', label='normv')
        ax1[1].plot(cost_stock[:, 1], "--", color='green', label='normv')
        total_cost = ssd_plot + normv_plot

        norm_l2_on_z = self.cost_cst * cost_stock[:, 2]
        total_cost += norm_l2_on_z
        ax1[0].plot(norm_l2_on_z, "--", color='orange', label='norm_l2_on_z')
        ax1[1].plot(cost_stock[:, 2], "--", color='orange', label='norm_l2_on_z')

        fields_diff_norm_v = self.cost_cst *  cost_stock[:, 3]
        total_cost += fields_diff_norm_v
        ax1[0].plot(fields_diff_norm_v, "--", color="purple", label='fields_diff_norm_v')
        ax1[1].plot(cost_stock[:, 3], "--", color='purple', label='fields_diff_norm_v')

        ax1[0].plot(total_cost, color='black', label=r'\Sigma')
        ax1[0].legend()
        ax1[1].legend()
        ax1[0].set_title("Lambda = " + str(self.cost_cst))

        return fig1,ax1


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
        grad_source = tb.spatialGradient(self.source)
        grad_source_resi = (grad_source * residuals_ini.unsqueeze(2)).sum(dim=1) / C
        K_grad_source_resi = self.mp.kernelOperator(grad_source_resi)
        self.norm_v_2 = (grad_source_resi * K_grad_source_resi).sum()

        self.total_cost = self.ssd + lamb * self.norm_v_2

        # Reduce field norm
        mask_1 = tb.imgDeform(self.mask_reduce[0][None].cpu(),
                              self.mp.get_deformator(),
                              dx_convention=self.dx_convention)
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
            self.mp.residual_mask.to(device)
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
