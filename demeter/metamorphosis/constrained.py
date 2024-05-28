import torch
import warnings
import matplotlib.pyplot as plt
from math import prod

from abc import ABC, abstractmethod
from metamorphosis import Geodesic_integrator,Optimize_geodesicShooting

from utils.constants import *
import utils.torchbox as tb
import utils.cost_functions as cf

class Residual_norm_function(ABC):

    @abstractmethod
    def __init__(self,mask,mu = 1,rho=None):
        if mu <= 0:
            raise ValueError(f"mu must be a non zero real positive value, got mu = {mu:.3f}")
        self.mu = mu
        self.rho = mu if rho is None else rho
        self.mask = mask
        if self.mask.shape[0] > 1: self.n_step = self.mask.shape[0]

    def __repr__(self):
        return f'{self.__class__.__name__}:(mu = {self.mu:.2E},rho = {self.rho:.2E})'

    def to_device(self,device):
        self.mask = self.mask.to(device)

    @abstractmethod
    def f(self,t):
        pass

    def F(self,t):
        return self.mask[t] * self.f(t) / self.mu

    def F_div_f(self,t):
        return self.mu*self.mask[t]

    @abstractmethod
    def dt_F(self,t):
        pass



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


class ConstrainedMetamorphosis_integrator(Geodesic_integrator):

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
        super(ConstrainedMetamorphosis_integrator, self).__init__(sigma_v)
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

    # def _update_residuals_weighted_semiLagrangian_(self, deformation):
    #     f_t = self.rf.f(self._i)
    #     fz_times_div_v = f_t * self.residuals * tb.Field_divergence(dx_convention='pixel')(self.field)[0, 0]
    #     div_fzv = -tb.imgDeform(f_t * self.residuals,
    #                             deformation,
    #                             dx_convention='pixel',
    #                             clamp=False)[0, 0] \
    #               + fz_times_div_v / self.n_step
    #     z_time_dtF = self.residuals * self.rf.dt_F(self._i)
    #     self.residuals = - (div_fzv + z_time_dtF) / f_t
    #
    # def _update_image_weighted_semiLagrangian_(self, deformation,residuals = None,sharp=False):
    #     if residuals is None: residuals = self.residuals
    #     image = self.source if sharp else self.image
    #     self.image = tb.imgDeform(image, deformation, dx_convention='pixel') + \
    #                  (self.rf.F_div_f(self._i) *  residuals) / self.n_step
    #                  # (self.rf.mu * self.rf.mask[self._i] * residuals) / self.n_step
    #
    #     # ga = (self.rf.mask[self._i] * self.residuals) / self.n_step
    #     # plt.figure()
    #     # p = plt.imshow(ga[0])
    #     # plt.colorbar(p)
    #     # plt.show()
    #
    # def _update_field_oriented_weighted_(self):
    #     grad_image = tb.spacialGradient(self.image, dx_convention='pixel')
    #     if self.flag_W:
    #         free_field = tb.im2grid(
    #             self.kernelOperator((- self.rf.f(self._i) * self.residuals * grad_image[0]))
    #         )/self._get_mu_()
    #     else:
    #         free_field = self._compute_vectorField_(self.residuals, grad_image)
    #         free_field *= self._field_cst_mult()
    #     oriented_field = 0
    #     if self.flag_O:
    #         mask_i = self.orienting_mask[self._i][..., None].clone()
    #         free_field *= 1 / (1 + (self.gamma * mask_i))
    #
    #         oriented_field = self.orienting_field[self._i][None].clone()
    #         oriented_field *= (self.gamma * mask_i) / (1 + (self.gamma * mask_i))
    #
    #     self.field = free_field + oriented_field

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
            self._update_residuals_semiLagrangian_(def_z)

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

    def __init__(self, source, target, geodesic, cost_cst,data_term=None, optimizer_method='adadelta'):
        super().__init__(source, target, geodesic, cost_cst,data_term, optimizer_method)
        if self.mp.flag_O: self._cost_saving_ = self._oriented_cost_saving_


    def _get_mu_(self):
        return float(self.mp._get_mu_())

    def _get_rho_(self):
        return float(self.mp._get_rho_())

    def _get_gamma_(self):
        return float(self.mp._get_gamma_())

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
    #
    #     # Computes only
    #     if len(args) == 2 and not args[0].shape[-1] in [2,3] :
    #         residual, image = args[0],args[1]
    #         C = residual.shape[1]
    #         grad_source = tb.spacialGradient(image)
    #         grad_source_resi = (grad_source * residual.unsqueeze(2)).sum(dim=1) / C
    #         K_grad_source_resi = self.mp.kernelOperator(grad_source_resi)
    #
    #         return (grad_source_resi * K_grad_source_resi).sum()
    #     elif len(args) == 1 and args[0].shape[-1] in [2,3]:
    #         print("\n Warning !!!!  make sure that kernelOrerator is self adjoint.")
    #         raise ValueError("This method to compute the V norm is wrong"
    #                          " with gaussian Kernels")
    #         field = args[0]
    #         k_field = self.mp.kernelOperator(field)
    #         return (k_field * field).sum()
    #     else:
    #         raise ValueError(f"Bad arguments, see usage in Doc got args = {args}")


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
        # print(f"k_grad_source_resi = {K_grad_source_resi.shape}")
        if self.mp.flag_O:
            # print(f"glag O  {(1 + self._get_gamma_() * self.mp.orienting_mask[0][None]).shape}")
            K_grad_source_resi *= (1 + self._get_gamma_() * self.mp.orienting_mask[0][None])
        self.norm_v_2 = (grad_source_resi * K_grad_source_resi).sum()
        # print(f"norm_v_2 = {self.norm_v_2.shape}")

        self.total_cost = self.data_loss + lamb * self.norm_v_2

        # #  || v - w ||_V
        # if self.mp.flag_O:
        #     # print("différence de champs")
        #     fields_diff = grad_source_resi - tb.grid2im(self.mp.orienting_field[0][None])
        #     K_fields_diff = self.mp.kernelOperator(fields_diff)
        #     self.fields_diff_norm_V = (fields_diff * K_fields_diff).sum()/prod(self.source.shape[2:])
        #
        #     self.total_cost += lamb * self._get_gamma_() * self.fields_diff_norm_V

        # # Perform the scalar product of the field with orienting field
        if self.mp.flag_O:
            # print("produit scalaires")
            a = grad_source_resi * tb.grid2im(self.mp.orienting_field[0][None])
            a *= self._get_gamma_() * self.mp.orienting_mask[0][None]
            self.fields_diff_norm_V  =  (a).sum()/prod(self.source.shape[2:])
            # self.fields_diff_norm_V *=
            self.total_cost += - lamb * self._get_gamma_() * self.fields_diff_norm_V


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
        super(ConstrainedMetamorphosis_Shooting, self).forward(z_0, n_iter, grad_coef, verbose,plot)
        self.to_device('cpu')

    def to_device(self, device):
        if self.mp.flag_W:
            self.mp.rf.to_device(device)
        if self.mp.flag_O:
            self.mp.orienting_mask = self.mp.orienting_mask.to(device)
            self.mp.orienting_field = self.mp.orienting_field.to(device)
        super(ConstrainedMetamorphosis_Shooting, self).to_device(device)

    def plot_cost(self):
        fig1, ax1 = plt.subplots(1, 2,figsize=(10,10))

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
                         "gamma = " + str(self._get_gamma_()))



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