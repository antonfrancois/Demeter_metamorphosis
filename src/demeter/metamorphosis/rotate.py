"""
Rotate.py
"""

import matplotlib.pyplot as plt
import torch

import __init__
from math import prod, sqrt

from demeter.metamorphosis import Geodesic_integrator,Optimize_geodesicShooting

from demeter.constants import *
import demeter.utils.torchbox as tb


# TODO: move to utils.rotate (file to be made)
def apply_rot_mat(grid,rot_mat):
    rotated_grid = torch.einsum('ij,bhwj->bhwi',rot_mat, grid)
    return rotated_grid

class RotatingMetamorphosis_integrator(Geodesic_integrator):
    """


    """
    def __init__(self,rho,**kwargs):
        super().__init__(**kwargs)
        self.rho = rho
        # self.n_step = n_step

    def _get_rho_(self):
        return float(self.rho)

    def to_device(self,device):
        try:
            self.rot_mat = self.rot_mat.to(device)
        except AttributeError:
            pass

    def step(self):
        print("\n")
        print("="*25)
        print('step',self._i)
        momentum_I = self.momenta['momentum_I'].clone()
        momentum_R = self.momenta['momentum_R'].clone()
        print('momentum_I',momentum_I.min().item(),momentum_I.max().item())
        print("momentum_I", momentum_I.shape)
        print("momentum_R",momentum_R)

        # ----------------------------------------------
        ## 0 Contrainte
        grad_image = tb.spatialGradient(self.image, dx_convention = self.dx_convention)
        IgradI_x = torch.einsum('ijkl,ijkm->ijklm', tb.im2grid(grad_image[0]), self.id_grid)
        x_IgradI = torch.einsum('ijkl,ijkm->ijklm',self.id_grid, tb.im2grid(grad_image[0]))

        if self._i == 0:
            print((IgradI_x - x_IgradI).shape)
            print((IgradI_x - x_IgradI).sum(dim=[1,2]))
            cst = (IgradI_x - x_IgradI)[...,0,1][None].clone()
            print('cst',cst.shape)
            c=  (momentum_I * cst).sum(dim=[2,3])
            print("c avant", c)
            sum_cst =  cst.sum(dim=[2,3])
            print("sum_cst ", cst.sum(dim=[2,3]))
            if sum_cst != 0:
                c = c /  sum_cst
            print("c final", c)
            momentum_I -= c
            print('momentum_I',momentum_I.shape)
        # -----------------------------------------------
        ## 1. Compute the vector field
        ## 1.1 Compute the gradient of the image by finite differences
        self.field,norm_V = self._compute_vectorField_(
            momentum_I, grad_image)
        # self.field *= 0
        # self.field *= sqrt(self.rho)
        print('field min max',self.field.min(),self.field.max())
        # ic(self.field.device)
        # ----------------------------------------------
        # 1.2 Compute the rotation

        # ic(grad_image.shape)
#         ic(grad_image.shape,self.id_grid.shape,momentum_I.shape)
        momI_gradI = tb.im2grid(momentum_I * grad_image[0])
        momIgradI_x = torch.einsum('ijkl,ijkm->ijklm', momI_gradI, self.id_grid)
        x_momIgradI = torch.einsum('ijkl,ijkm->ijklm',self.id_grid,momI_gradI)

        print('x_momIgradI',x_momIgradI.shape)
        mom_rotated = momentum_R @ self.rot_mat.T
        mom_rotated =  (mom_rotated - mom_rotated.T) /2
        print("rot mat",self.rot_mat)
        print("mom_rotated",mom_rotated)


        int_mom_I = .5 * sqrt(self.rho) * (momIgradI_x - x_momIgradI).sum(dim=[1,2])[0]
        print('int_mom_I',int_mom_I)
        self.d_rot = mom_rotated - int_mom_I
        if self._i == 0:
            self.d_rot_ini = self.d_rot.clone()
        print('d_rot',self.d_rot)
#         ic(self.d_rot.device)

        # self.d_rot = (
        #         momentum_R * self.rot_mat
        #                   - sqrt(self.rho) *
        #         momentum_I *(
        #                        .5 * (gradI_x - x_gradI).sum(dim=[1,2])
        #                   )
        #              )
        # self.rot_mat = self.rot_mat + self.d_rot @ self.rot_mat


        #     cst = 1/mom_rotated[0,1]
        #     momentum_I -= cst


        # -----------------------------------------------
        # 2. Compute the residuals
        self.residuals = (sqrt(1 - self.rho) *
                          momentum_I)
#         ic(self.residuals.device)
        print("dx_convention",self.dx_convention)
        print("grid min max",self.id_grid.min().item(),self.id_grid.max().item())

        # -----------------------------------------------
        # 3. Update the image
        # id_rot = apply_rot_mat(self.id_grid, - sqrt(self.rho) * self.d_rot)
        eye = torch.eye(2).to(self.image.device)
        # id_rot = self.id_grid - apply_rot_mat(self.id_grid, self.d_rot/self.n_step)
        id_rot = apply_rot_mat(self.id_grid, eye+ self.d_rot/self.n_step)

        deform_rot = id_rot - sqrt(self.rho) *  self.field/self.n_step


        self._update_image_semiLagrangian_(deform_rot,momentum=momentum_I)

        if self.flag_hamiltonian_integration:

            # Norm L2 on z
            norm_l2_on_z = .5 * (self.residuals ** 2).sum()

            # Norm L2 on R
            norm_l2_on_R = .5 * torch.trace( self.d_rot.T @ self.d_rot_ini)
            self.ham_value = norm_V + norm_l2_on_z + norm_l2_on_R

        # -----------------------------------------------
        ## 4. Update momentums
        ## 4.1 Update image momentum

        momentum_I =  (
            self._compute_div_momentum_semiLagrangian_(
                deform_rot,
                momentum_I,
                cst = - sqrt(self.rho),
                field = sqrt(self.rho) * self.field
            )
        )
#         ic(momentum_I.device)
        momentum_R = momentum_R - self.d_rot.T @ momentum_R  / self.n_step
#         ic(momentum_R.device)
        self.momenta['momentum_I'] = momentum_I.clone()
        self.momenta['momentum_R'] = momentum_R.clone()

        exp_A = torch.linalg.matrix_exp(self.d_rot/self.n_step)
        # exp_A =
        print("exp_A:" , exp_A)
        self.rot_mat = exp_A @ self.rot_mat


        print("rot mat * rot mat.T",self.rot_mat @ self.rot_mat.T)

        print('momentum_R',momentum_R)
        print("rot mat",self.rot_mat)
        print("arc ", torch.arcsin(exp_A[0,1])/torch.pi,
              torch.arccos(exp_A[0,1])/torch.pi)

        return (self.image,
                sqrt(self.rho) * self.field,
                sqrt(1 - self.rho) * self.residuals)

    def forward(self, image, momenta,**kwargs):
        self.rot_mat = torch.eye(2)
        # r = momenta['r']
        #  = torch.tensor(
        #     [[0,r],
        #     [-r,0]],
        #     dtype=torch.float32)
        momentum_R = momenta['momentum_R'].clone()
        momenta['momentum_R'] =  (momentum_R - momentum_R.T) /2
        self.to_device(momentum_R.device)

        return super().forward(image,momenta,**kwargs)

    def plot(self, n_figs=5):
        if n_figs == -1:
            n_figs = self.n_step
        plot_id = (
            torch.quantile(
                torch.arange(self.image_stock.shape[0], dtype=torch.float),
                torch.linspace(0, 1, n_figs),
            )
            .round()
            .int()
        )

        kw_image_args = dict(
            cmap="gray", extent=[-1, 1, -1, 1], origin="lower", vmin=0, vmax=1
        )
        # v_abs_max = (self.residuals_stock.abs().max()).max()
        # v_abs_max = torch.quantile(self.momenta.abs(), 0.99)
        momentum =  self.momenta['momentum_I']
        v_abs_max = torch.quantile(momentum.abs(), 0.99)
        kw_residuals_args = dict(
            cmap="RdYlBu_r",
            extent=[-1, 1, -1, 1],
            origin="lower",
            vmin=-v_abs_max,
            vmax=v_abs_max,
        )
        size_fig = 5
        # C = self.momentum_stock.shape[1]
        C = 1
        fig, ax = plt.subplots(
            n_figs,
            2 + C,
            constrained_layout=True,
            figsize=(size_fig * 3, n_figs * size_fig),
        )

        for i, t in enumerate(plot_id):
            i_s = ax[i, 0].imshow(
                self.image_stock[t, :, :, :].detach().permute(1, 2, 0).numpy(),
                **kw_image_args,
            )
            ax[i, 0].set_title("t = " + str((t / (self.n_step - 1)).item())[:3])
            ax[i, 0].axis("off")
            fig.colorbar(i_s, ax=ax[i, 0], fraction=0.046, pad=0.04)

            # for j in range(C):
            #     r_s = ax[i, j + 1].imshow(
            #         self.momentum_stock[t, j].detach().numpy(), **kw_residuals_args
            #     )
            #     ax[i, j + 1].axis("off")

            # fig.colorbar(r_s, ax=ax[i, -2], fraction=0.046, pad=0.04)

            tb.gridDef_plot_2d(
                self.get_deformation(to_t = t + 1),
                add_grid=False,
                ax=ax[i, -1],
                step=int(min(self.field_stock.shape[2:-1]) / 30),
                check_diffeo=True,
                dx_convention=self.dx_convention,
            )

        return fig, ax

    def plot_rot(self):
        fig, ax = plt.subplots(1,2)

        shape = self.source.shape[2:]
        id_grid = tb.make_regular_grid(shape, dx_convention = "2square")
        rot = self.mp.rot_mat
        rot_grid_end = apply_rot_mat(id_grid, rot)
        ax[0].imshow(self.mp.image[0,0], cmap='gray', origin="lower")
        tb.gridDef_plot_2d(rot_grid_end,
                           ax=ax[0],
                           step=25,
                           dx_convention="2square",
                           color='red')

        source_rot = tb.imgDeform(self.source,rot_grid_end,dx_convention="2square")
        ax[1].imshow(
            tb.imCmp(
                source_rot,
                self.target,
                method= "seg"
            ),
             cmap='gray', origin="lower"
        )
        ax[1].set_title("rotated_source vs target")


        plt.show()

class RotatingMetamorphosis_Optimizer(Optimize_geodesicShooting):

    def __init__(self,**kwargs):
        print(kwargs.keys())
        super().__init__(**kwargs)
        self._cost_saving_ = self._rotating_cost_saving_

    def _get_rho_(self):
        return float(self.mp.rho)

    def get_all_arguments(self):
        params_all  = super().get_all_arguments()
        params_spe = {
            'rho':self._get_rho_(),
        }
        return {**params_all,**params_spe}

    def get_all_parameters(self):
        pass

    def cost(self,momenta,**kwargs):

        rho = self._get_rho_()
        self.to_device(momenta['momentum_I'].device)
        ic("momentum_I",momenta['momentum_I'].device)
        self.mp.forward(self.source,momenta,
                        save=False,
                        plot=0,
                        hamiltonian_integration=self.flag_hamiltonian_integration,
                        )
        # Compute the data_term. Default is the Ssd
        self.data_loss = self.data_term()
        ic(self.data_loss)

        if self.flag_hamiltonian_integration:
            self.total_cost = self.data_loss + (self.cost_cst) * self.mp.ham_integration
        else:
            # Norm V
            self.norm_v_2 = .5 * rho  * self._compute_V_norm_(momenta['momentum_I'],self.source)

            # Norm L2 on z
            volDelta = prod(self.dx)
            z = sqrt(1 - rho) * (momenta['momentum_I']/volDelta)
            self.norm_l2_on_z = .5 * (z ** 2).sum() * volDelta

            # Norm L2 on R
            self.norm_l2_on_R = .5 * torch.trace( self.mp.d_rot_ini.T @ self.mp.d_rot_ini)
            # self.norm_l2_on_R *= 1000
            ic(self.norm_l2_on_R)

        self.total_cost = self.data_loss + \
                          self.cost_cst * (self.norm_v_2 + self.norm_l2_on_z + self.norm_l2_on_R)

        return self.total_cost

    def _rotating_cost_saving_(self,i, loss_stock):
        if loss_stock is None:
            d = 4
            return torch.zeros((i+1, d))

        loss_stock[i, 0] = self.data_loss.detach()
        loss_stock[i, 1] = self.norm_v_2.detach()
        loss_stock[i, 2] = self.norm_l2_on_z.detach()
        loss_stock[i, 3] = self.norm_l2_on_R.detach()

        return loss_stock

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
        ax1[0].plot(fields_diff_norm_v, "--", color="purple", label='normL2_on_R')
        ax1[1].plot(cost_stock[:, 3], "--", color='purple', label='normL2_on_R')

        ax1[0].plot(total_cost, color='black', label=r'\Sigma')
        ax1[0].legend()
        ax1[1].legend()
        ax1[0].set_title("Lambda = " + str(self.cost_cst))

        return fig1,ax1

