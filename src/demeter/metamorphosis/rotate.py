"""
Rotate.py
"""

import matplotlib.pyplot as plt
import torch

import __init__
from math import prod, sqrt

from demeter.metamorphosis import Geodesic_integrator,Optimize_geodesicShooting

from demeter.utils.constants import *
import demeter.utils.torchbox as tb


# TODO: move to utils.rotate (file to be made)
def apply_rot_mat(grid,rot_mat):
    rotated_grid = torch.einsum('ij,bhwj->bhwi',rot_mat, grid)
    return rotated_grid

class RotatingMetamorphosis_integrator(Geodesic_integrator):
    """

    """
    def __init__(self,rho,n_step, **kwargs):
        super().__init__(**kwargs)
        self.rho = rho
        self.n_step = n_step

    def _get_rho_(self):
        return float(self.rho)

    def step(self):
        print("\n")
        print("="*25)
        print('step',self._i)
        momentum_I = self.momenta['momentum_I'].clone()
        momentum_R = self.momenta['momentum_R'].clone()

        # -----------------------------------------------
        ## 1. Compute the vector field
        ## 1.1 Compute the gradient of the image by finite differences
        grad_image = tb.spatialGradient(self.image, dx_convention = self.dx_convention)
        self.field = self._compute_vectorField_(
            momentum_I, grad_image)
        self.field *= sqrt(self.rho)
        print('field min max',self.field.min(),self.field.max())
        # 1.2 Compute the rotation

        # ic(grad_image.shape)
        momI_gradI = tb.im2grid(momentum_I * grad_image[0])
#         ic(grad_image.shape,self.id_grid.shape,momentum_I.shape)

        momIgradI_x = torch.einsum('ijkl,ijkm->ijklm', momI_gradI, self.id_grid)
        x_momIgradI = torch.einsum('ijkl,ijkm->ijklm',self.id_grid,momI_gradI)
        mom_rot = momentum_R @ self.rot_mat.T
        mom_rot =  (mom_rot - mom_rot.T) /2
        print("rot mat",self.rot_mat)
        print("mom_rot",mom_rot)
#         ic(gabon.shape)
        int_mom_I = .5 * sqrt(self.rho) * (momIgradI_x - x_momIgradI).sum(dim=[1,2])[0]
        print('int_mom_I',int_mom_I)
#         ic(congo.shape)
        self.d_rot = mom_rot - int_mom_I
        if self._i == 0:
            self.d_rot_ini = self.d_rot.clone()
        print('d_rot',self.d_rot)


        # self.d_rot = (
        #         momentum_R * self.rot_mat
        #                   - sqrt(self.rho) *
        #         momentum_I *(
        #                        .5 * (gradI_x - x_gradI).sum(dim=[1,2])
        #                   )
        #              )
        # self.rot_mat = self.rot_mat + self.d_rot @ self.rot_mat
        self.rot_mat = torch.linalg.matrix_exp(self.d_rot/self.n_step) @ self.rot_mat
        print('momentum_R',momentum_R)
        print("rot mat",self.rot_mat)

        # -----------------------------------------------
        # 2. Compute the residuals
        self.residuals = (sqrt(1 - self.rho) *
                          momentum_I)

        print("dx_convention",self.dx_convention)
        print("grid min max",self.id_grid.min().item(),self.id_grid.max().item())

        # -----------------------------------------------
        # 3. Update the image
        # id_rot = apply_rot_mat(self.id_grid, - sqrt(self.rho) * self.d_rot)
        id_rot = self.id_grid - apply_rot_mat(self.id_grid,  self.d_rot)

        deform_rot = id_rot - sqrt(self.rho) *  self.field/self.n_step

        # ic(self.id_grid.min().item(), self.id_grid.max().item(),
        #    id_rot.min().item(), id_rot.max().item(),
        #    deform_rot.min().item(), deform_rot.max().item()
        #    )

        # fig,ax = plt.subplots(1,3)
        # ax[0].imshow(self.image[0,0].detach().numpy())
        # ax[0].set_title('image BEFORE, step = '+str(self._i))
        # ax[1].imshow(momentum_I[0,0].detach().numpy())
        # ax[1].set_title('momentum_I, step = '+str(self._i))
        # tb.gridDef_plot(deform_rot,ax=ax[2],step=10)
        # plt.show()
        self._update_image_semiLagrangian_(deform_rot,momentum=momentum_I)

        # fig,ax = plt.subplots(1,3)
        # ax[0].imshow(self.image[0,0].detach().numpy())
        # ax[0].set_title('image AFTER, step = '+str(self._i))
        # ax[1].imshow(momentum_I[0,0].detach().numpy())
        # ax[1].set_title('momentum_I, step = '+str(self._i))
        # tb.gridDef_plot(deform_rot,ax=ax[2],step=10)
        # plt.show()

        # -----------------------------------------------
        ## 4. Update momentums
        ## 4.1 Update image momentum

        momentum_I = - sqrt(self.rho) * (
            self._compute_div_momentum_semiLagrangian_(deform_rot,
                                                       momentum_I)
        )
        momentum_R = momentum_R + self.d_rot.T @ self.rot_mat / self.n_step
        self.momenta['momentum_I'] = momentum_I.clone()
        self.momenta['momentum_R'] = momentum_R.clone()

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

        return super().forward(image,momenta,**kwargs)

class RotatingMetamorphosis_Optimizer(Optimize_geodesicShooting):

    def __init__(self,**kwargs):
        print(kwargs.keys())
        super().__init__(**kwargs)

    def _get_rho_(self):
        return float(self.mp.rho)

    def get_all_parameters(self):
        pass

    def cost(self,momenta,**kwargs):

        rho = self._get_rho_()
        self.mp.forward(self.source,momenta, save=False, plot=0)
        # Compute the data_term. Default is the Ssd
        self.data_loss = self.data_term()
        ic(self.data_loss)

        # Norm V
        self.norm_v_2 = .5 * rho  * self._compute_V_norm_(momenta['momentum_I'],self.source)

        # Norm L2 on z
        volDelta = prod(self.dx)
        z = sqrt(1 - rho) * (momenta['momentum_I']/volDelta)
        self.norm_l2_on_z = .5 * (z ** 2).sum() * volDelta

        # Norm L2 on R
        self.norm_l2_on_R = .5 * torch.trace( self.mp.d_rot_ini.T @ self.mp.d_rot_ini)

        self.total_cost = self.data_loss + \
                          self.cost_cst * (self.norm_v_2 + self.norm_l2_on_z + self.norm_l2_on_R)

        return self.total_cost

