"""
Rotate.py
"""
from logging import warning

import matplotlib.pyplot as plt
import torch

import __init__
from math import prod, sqrt

from kornia.geometry import scale

from demeter.utils.decorators import time_it
from demeter.metamorphosis import Geodesic_integrator,Optimize_geodesicShooting

from demeter.constants import *
import demeter.utils.torchbox as tb
from demeter.utils.toolbox import plot_loss_with_multiple_y_axes


def prepare_momenta(image_shape,
                    diffeo : bool = True,
                    rotation : bool = True,
                    translation : bool = True,
                    scaling : bool = True,
                    affine : bool = False,
                    rot_prior = None,
                    trans_prior= None,
                    scale_prior= None,
                    affine_prior= None,
                    device = "cuda:0",
                    requires_grad = True):
    dim = 2 if len(image_shape) == 4 else 3
    if affine:
        scaling = rotation = False
        translation = True
        warning("affine is true, scaling and rotation set to False, translation set to True")

        if affine_prior is None:
            if rot_prior is not None:
                affine_prior = rot_prior
            else:
                affine_prior = torch.zeros((dim,dim))
        if not torch.is_tensor(affine_prior):
            affine_prior = torch.tensor(affine_prior)
    else:
        if rot_prior is None:
            rot_prior = torch.zeros((dim,)) if dim == 3 else torch.tensor([0.])
        if not torch.is_tensor(rot_prior):
            rot_prior = torch.tensor(rot_prior)
        if scale_prior is None:
            scale_prior = torch.zeros((dim,))
        if not torch.is_tensor(scale_prior):
            scale_prior = torch.tensor(scale_prior)

    if trans_prior is None:
        trans_prior = torch.zeros((dim,))
    if not torch.is_tensor(trans_prior):
        trans_prior = torch.tensor(trans_prior)


    momenta = {}
    kwargs = {
        "dtype":torch.float32,
        "device":device
    }
    if diffeo:
        momenta["momentum_I"]= torch.zeros(image_shape,**kwargs)
    if rotation:
        if len(rot_prior.shape)==2:
            momenta["momentum_R"] = rot_prior.to(kwargs["dtype"]).to(kwargs["device"])
        elif len(rot_prior.shape)<=1:
            if dim == 2:
                momenta["momentum_R"] = torch.tensor(
                [ [0, rot_prior],
                             [-rot_prior, 0],
                         ],
                        dtype=torch.float32, device='cuda:0')
            elif dim == 3:
                r1, r2, r3 = rot_prior
                momenta["momentum_R"] = torch.tensor(
                [[0,-r1, -r2 ],
                         [r1, 0, -r3],
                         [r2, r3, 0]],
                        dtype=torch.float32, device='cuda:0')
        else:
            raise ValueError("Rotation prior must be 2 or 1 dimensional")
    if affine:
        if len(affine_prior.shape)==2:
            momenta["momentum_A"] = affine_prior.to(kwargs["dtype"]).to(kwargs["device"])
        elif len(affine_prior.shape)<=1:
            if dim == 2:
                momenta["momentum_A"] = torch.tensor(
                [ [0, affine_prior],
                        [-affine_prior, 0],
                         ],
                        dtype=torch.float32, device='cuda:0')
            elif dim == 3:
                r1, r2, r3 = affine_prior
                momenta["momentum_A"] = torch.tensor(
                [[0,-r1, -r2 ],
                         [r1, 0, -r3],
                         [r2, r3, 0]],
                        dtype=torch.float32, device='cuda:0')
        else:
            raise ValueError("Rotation prior must be 2 or 1 dimensional")
        # momenta["momentum_A"] = affine_prior.to(kwargs["dtype"]).to(kwargs["device"])
    if translation:
        momenta["momentum_T"]= trans_prior.to(kwargs["dtype"]).to(kwargs["device"])
    if scaling:
        momenta["momentum_S"] = scale_prior.to(kwargs["dtype"]).to(kwargs["device"])

    for keys in momenta.keys():
        momenta[keys].requires_grad=requires_grad

    return momenta


class RigidMetamorphosis_integrator(Geodesic_integrator):
    """


    """
    def __init__(self,rho, constraints = True,**kwargs):
        super().__init__(**kwargs)
        self.rho = rho
        # self.n_step = n_step
        self.constraints = constraints

    def _get_rho_(self):
        return float(self.rho)

    def to_device(self,device):
        try:
            self.rot_mat = self.rot_mat.to(device)
            self.translation = self.translation.to(device)
            self.scale = self.scale.to(device)
        except AttributeError:
            pass
        super().to_device(device)

    def projection(self, c, p):
        cp = (c * p).sum()
        norm_c = (c **2).sum()
        if norm_c != 0:
            cst = c * cp /  norm_c
        else:
            cst = 0
        return p - cst

    def _contrainte_(self, momentum_I, source):
        grad_source = tb.spatialGradient(source, dx_convention = self.dx_convention)
        IgradI_x = tb.multiply_grid_vectors(tb.im2grid(grad_source[0]), self.id_grid)
        x_IgradI = tb.multiply_grid_vectors(self.id_grid, tb.im2grid(grad_source[0]))

        # contrainte rotation
        if self._dim == 2:
            c_list = [(IgradI_x - x_IgradI)[...,0,1][None]]
        elif self._dim == 3:
            _k = [0, 0, 1]
            _l = [1, 2, 2]
            c_list = (IgradI_x - x_IgradI)[...,_k,_l].permute(4,0,1,2,3)
            c_list = [c for c in c_list]

        #contrainte translation
        for i in range(grad_source.shape[2]):
            c_list.append(grad_source[:,:,i])


        # Orthonormaliser la liste
        c_ortho_list = [c_list[0] / (c_list[0] **2).sum().sqrt()]
        if len(c_list) > 1:
            for c in c_list[1:]:
                c_tilde = c
                for co in c_ortho_list:
                    c_tilde -= (c * co).sum() * co
                c_norm = (c_tilde**2).sum().sqrt()
                c_ortho_list.append(
                    c_tilde / c_norm if c_norm != 0 else c_tilde
                )

        # check orthonormalisation
        # print("\t len ortho_list", len(c_ortho_list))
        # print("\t gradS p^I :",(momentum_I * grad_source).sum(dim=[-1,-2])[0,0])



        for c in c_ortho_list:
            momentum_I = self.projection(c, momentum_I)
            # print("\t", 'momentum_I',momentum_I.shape)

        if self._dim == 3:
            assert (c_ortho_list[0] * c_ortho_list[1]).sum() < 1e-5, f"(c_otho_list[0] * c_otho_list[1]).sum() = {(c_ortho_list[0] * c_ortho_list[1]).sum()}"
            assert (c_ortho_list[0] * c_ortho_list[2]).sum() < 1e-5, f"(c_otho_list[0] * c_otho_list[2]).sum() = {(c_ortho_list[0] * c_ortho_list[2]).sum()}"
            assert (c_ortho_list[2] * c_ortho_list[1]).sum() < 1e-5, f"(c_otho_list[2] * c_otho_list[1]).sum() = {(c_ortho_list[2] * c_ortho_list[1]).sum()}"
        # for i, c in enumerate(c_list):
        #     assert (c * momentum_I).sum() < 1e-4, f"(c_{i} * momentum_I).sum() = {(c * momentum_I).sum()}"
            # if (c * momentum_I).sum() > 1e-5:
            #     print( f"(c_{i} * momentum_I).sum() = {(c * momentum_I).sum()}")

        return momentum_I

    def _step_old(self):
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
        if self._i == 0:
            momentum_I = self._contrainte_(momentum_I, self.source)


        # -----------------------------------------------
        ## 1. Compute the vector field
        ## 1.1 Compute the gradient of the image by finite differences
        grad_image = tb.spatialGradient(self.image, dx_convention = self.dx_convention)
        self.field,norm_V = self._compute_vectorField_(
            momentum_I, grad_image)
        # self.field *= 0
        # self.field *= sqrt(self.rho)
        print('field min max',self.field.min(),self.field.max())
        # ----------------------------------------------
        # 1.2 Compute the rotation

        momI_gradI = tb.im2grid(momentum_I * grad_image[0])
        # momIgradI_x = torch.einsum('ijkl,ijkm->ijklm', momI_gradI, self.id_grid)
        # x_momIgradI = torch.einsum('ijkl,ijkm->ijklm',self.id_grid,momI_gradI)
        momIgradI_x = tb.multiply_grid_vectors( momI_gradI, self.id_grid)
        x_momIgradI = tb.multiply_grid_vectors(self.id_grid,momI_gradI)


        print('x_momIgradI',x_momIgradI.shape)
        mom_rotated = momentum_R @ self.rot_mat.T
        mom_rotated =  (mom_rotated - mom_rotated.T) /2
        print("rot mat",self.rot_mat)
        print("mom_rotated",mom_rotated)

        to_sum_dim = [ 1, 2] if self._dim == 2 else [1,2,3]
        int_mom_I = .5 * sqrt(self.rho) * (momIgradI_x - x_momIgradI).sum(dim=to_sum_dim)[0]
        print('int_mom_I',int_mom_I)
        self.d_rot = mom_rotated - int_mom_I
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


        #     cst = 1/mom_rotated[0,1]
        #     momentum_I -= cst


        # -----------------------------------------------
        # 2. Compute the residuals
        self.residuals = (sqrt(1 - self.rho) *
                          momentum_I)
        print("dx_convention",self.dx_convention)
        print("grid min max",self.id_grid.min().item(),self.id_grid.max().item())

        # -----------------------------------------------
        # 3. Update the image
        # id_rot = apply_rot_mat(self.id_grid, - sqrt(self.rho) * self.d_rot)
        eye = torch.eye(self._dim).to(self.image.device)
        # id_rot = self.id_grid - apply_rot_mat(self.id_grid, self.d_rot/self.n_step)
        id_rot = tb.grid_from_rotation(self.id_grid, eye + self.d_rot / self.n_step)

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
        momentum_R = momentum_R - self.d_rot.T @ momentum_R  / self.n_step
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

    def _compute_step_rotation_translation(self,
                                           momentum_R, rot_mat,
                                           momentum_T , translation,
                                           momentum_S, scale):
        momR_rotated = momentum_R @ rot_mat.T
        pre_rot_inf = momR_rotated #+ momT_translated

        momT_translated = momentum_T @ translation.T
        pre_rot_inf += momT_translated


        if self.debug:
            print("rot mat", rot_mat)
            print("momR_rotated",momR_rotated)
        # momR_rotated =  (momR_rotated - momR_rotated.T) /2
        #
        d_rot = (pre_rot_inf - pre_rot_inf.T) / 2
        # print('d_rot',self.d_rot)

        scale_inf = momentum_S * scale #+ momentum_T * translation
        scale = scale * (1 +  scale_inf / self.n_step)

        if self._i == 0:
            self._rot_inf_ini = d_rot.clone()
            self.scale_inf_ini = scale_inf.clone()

        exp_A = torch.linalg.matrix_exp(d_rot/self.n_step)
        rot_mat = exp_A @ rot_mat
        translation =translation +  ((d_rot.T + torch.diag(scale_inf)) @ translation + momentum_T) /self.n_step #


        norm_l2_on_R = .5 * torch.trace(d_rot.T @ self._rot_inf_ini)
        norm_l2_on_S = .5 * (scale_inf ** 2).sum()

        momentum_R = momentum_R - d_rot.T @ momentum_R  / self.n_step
        momentum_T = momentum_T - d_rot.T @ momentum_T / self.n_step
        momentum_S = momentum_S - scale_inf * momentum_S /self.n_step
        return momentum_R, momentum_T, momentum_S, rot_mat, translation, scale, d_rot, norm_l2_on_R, norm_l2_on_S

    def _compute_step_affine(self,
                               momentum_A, A_mat,
                               momentum_T , translation):
        d_affine = momentum_A @ A_mat.T
        # d_translation = momentum_T



        if self._i == 0:
            self._rot_inf_ini = d_affine.clone()

        exp_A = torch.linalg.matrix_exp(d_affine/self.n_step)
        A_mat = exp_A @ A_mat
        translation = translation +  momentum_T /self.n_step #

        norm_l2_on_A = .5 * torch.trace(d_affine.T @ self._rot_inf_ini)
        # if self.debug:
        #     ic(self._i,momentum_A,momentum_T,
        #        exp_A, A_mat,
        #        translation, norm_l2_on_A
        #        )

        momentum_A = (momentum_A - d_affine.T @ momentum_A  / self.n_step)
        # momentum_T = momentum_T # Momentum T is constant
        return momentum_A, momentum_T, A_mat, translation, norm_l2_on_A

    def step(self, image, momentum_I, momentum_R, momentum_T, momentum_S, momentum_A, rot_mat, translation, scale):
        """
        One integration step. Fully checkpoint-compliant: fixed number of outputs.
        """
        if self.debug:
            print("\n" + "="*25)
            print('step', self._i)

        # --- Apply constraints ---
        if self._i == 0 and self.constraints:
            momentum_I = self._contrainte_(momentum_I, self.source)

        # --- Rotation / Translation update ---
        if momentum_A is not None:
            ic(momentum_A.requires_grad)
            momentum_A, momentum_T, rot_mat, translation, norm_l2_on_A = self._compute_step_affine(
                               momentum_A, rot_mat,
                               momentum_T , translation
            )
        else:
            momentum_R, momentum_T, momentum_S, rot_mat, translation, scale, d_rot, norm_R_2, norm_S_2 = \
                self._compute_step_rotation_translation(
                    momentum_R, rot_mat,
                    momentum_T, translation,
                    momentum_S, scale
                )

        # --- Vector field and residuals ---
        # grad_image = tb.spatialGradient(image, dx_convention=self.dx_convention)
        # field, norm_V = self._compute_vectorField_(momentum_I, grad_image)
        field = self._update_field_(momentum_I, image)

        residuals = (1 - self.rho) * momentum_I

        # --- Image update ---
        deformation = self.id_grid - self.rho * field / self.n_step
        image = self._update_image_semiLagrangian_(momentum_I, image, deformation, residuals)

        # --- Momentum update ---
        momentum_I = self._compute_div_momentum_semiLagrangian_(
            deformation,
            momentum_I,
            cst=-sqrt(self.rho),
            field=sqrt(self.rho) * field
        )

        # --- Hamiltonian Integration ---
        if self.flag_hamiltonian_integration:
            norm_l2_on_z = .5 * (residuals ** 2).sum()
            self.ham_value = self.norm_v_i + norm_l2_on_z + norm_R_2

        # --- Always output the same things ---
        return (
            momentum_I,
            momentum_R,
            momentum_T,
            momentum_S,
            momentum_A,
            image,
            self.rho * field,
            residuals,
            rot_mat,
            translation,
            scale
        )

    def _forward_initialize_integration(self, image, momenta, device, save, sharp, hamiltonian_integration, plot):
        self.debug = True
        if self.debug:
            ic("debug is defined here", self.debug)

        self._dim = 2 if len(image.shape) == 4 else 3
        self.rot_mat = torch.eye(self._dim)
        # r = momenta['r']
        #  = torch.tensor(
        #     [[0,r],
        #     [-r,0]],
        #     dtype=torch.float32)
        # self.flag_field = True if "momentum_I" in momenta.keys() else False
        if "momentum_R" in momenta.keys():
            momentum_R = momenta['momentum_R'].clone()
            momenta['momentum_R'] =  (momentum_R - momentum_R.T) /2
            device = momenta['momentum_R'].device
            print("Will do rotation")
        else:
            device = momenta['momentum_A'].device
            print("Will do affine transformation")
            ic(momenta['momentum_A'].requires_grad)
        self.to_device(device)

        # if "momentum_T" in momenta:
        #     self.translation = torch.zeros_like(momenta['momentum_T'])
        #     self.flag_translation = True
        # else:
        #     self.translation = None
        #     self.flag_translation = False
        self.translation = torch.zeros((self._dim,), device=device)
        self.scale = torch.ones((self._dim,), device=device)

        self.flag_field =  True if  "momentum_I" in momenta.keys() else False

        super()._forward_initialize_integration(image, momenta, device, save, sharp, hamiltonian_integration, plot)

    def _forward_direct_step(self):
        # print("_forward_direct_step in rotate")
        if "momentum_A" in self.momenta.keys():
            momentum_A = self.momenta["momentum_A"]
            momentum_T = self.momenta["momentum_T"]
            flag_affine = True
            momentum_R = momentum_S = None
        elif "momentum_R" in self.momenta.keys():
            flag_affine = False
            momentum_R = self.momenta["momentum_R"]
            if "momentum_T" in self.momenta.keys():
                momentum_T = self.momenta["momentum_T"]
            else:
                momentum_T = torch.zeros((momentum_R.shape[-1],), device=momentum_R.device)
            if "momentum_S" in self.momenta.keys():
                momentum_S = self.momenta["momentum_S"]
            else:
                momentum_S = torch.zeros((momentum_R.shape[-1],), device=momentum_R.device)
            momentum_A = None
        else:
            raise ValueError("Momenta must contain at least momentum_A OR momentum_R.")



        if  self.flag_field:

            momentum_I = self.momenta["momentum_I"]
            momentum_I,momentum_R,momentum_T, momentum_S, momentum_A, self.image, self.field, self.residuals, self.rot_mat, self.translation, self.scale \
                = self.step(
                self.image,
                momentum_I,
                momentum_R,
                momentum_T,
                momentum_S,
                momentum_A,
                self.rot_mat,
                self.translation,
                self.scale
            )
            self.momenta["momentum_I"] = momentum_I
        else:
            if flag_affine:
                momentum_A, momentum_T, self.rot_mat, self.translation, norm_l2_on_A = self._compute_step_affine(
                   momentum_A, self.rot_mat,
                   momentum_T , self.translation
                )
                self.momenta["momentum_A"] = momentum_A
                self.momenta["momentum_T"] = momentum_T

            else:
                momentum_R, momentum_T, momentum_S, self.rot_mat, self.translation, self.scale, _, _,_ =\
                    self._compute_step_rotation_translation(
                        momentum_R, self.rot_mat,
                        momentum_T,  self.translation,
                        momentum_S, self.scale
                    )

                self.momenta["momentum_R"] = momentum_R
                self.momenta["momentum_T"] = momentum_T
                self.momenta["momentum_S"] = momentum_S


    def _forward_checkpointed_step(self):
        # print("_forward_checkpointed_step, rotate")
        if not "momentum_I" in self.momenta.keys():
            # print("go to direct step")
            self._forward_direct_step()

        # print("Going to checjpoint")
        momentum_I, momentum_R, momentum_T, image, field, residuals, rot_mat, translation = torch.utils.checkpoint.checkpoint(
            self.step,
            self.image,
            self.momenta["momentum_I"],
            self.momenta["momentum_R"],
            self.momenta.get("momentum_T", torch.zeros_like(self.momenta["momentum_R"])),
            self.rot_mat,
            self.translation,
            use_reentrant=False,
        )
        # print("[CHECKPOINT outputs]")
        # for x in [momentum_I, momentum_R, momentum_T, image, field, residuals, rot_mat, translation]:
        #     print("\t", x.shape, x.requires_grad)
        #     if x.requires_grad:
        #         x.register_hook(lambda grad: print(f"\tGrad computed for tensor of shape {x.shape}\n"))
        # Update attributes after checkpoint
        self.image = image
        self.field = field
        self.residuals = residuals
        self.momenta["momentum_I"] = momentum_I
        self.momenta["momentum_R"] = momentum_R
        self.momenta["momentum_T"] = momentum_T
        self.rot_mat = rot_mat
        self.translation = translation

    def get_rigidor(self, grid = None):
        """
        return a grid ready to apply the rotation and translation estimated

        Example:
        ---------
        >>>rot_def = mr.mp.get_rigidor()
        >>>rotated_source = tb.imgDeform(source,rot_def,dx_convention='2square')
        """
        if grid is None:
            grid = self.id_grid
        return tb.grid_from_rotation_translation_scaling(
            grid, self.rot_mat.T, - self.translation, 1/self.scale
        )

    def get_rigid(self, grid = None):
        if grid is None:
            grid = self.id_grid
        return tb.grid_from_rotation_translation_scaling(
            grid, self.rot_mat, self.translation, self.scale)

    def _save_step(self):
        if self.flag_field:
            super()._save_step()

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
        rot_grid_end = tb.grid_from_rotation(id_grid, rot)
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

class RigidMetamorphosis_Optimizer(Optimize_geodesicShooting):

    def __init__(self, cost_field_cst = .5, cost_affine_cst = 1, **kwargs):
        super().__init__(**kwargs)
        self._cost_saving_ = self._rotating_cost_saving_
        self.cst_field = cost_field_cst
        self.cost_affine_cst = cost_affine_cst

    def _get_rho_(self):
        return float(self.mp.rho)

    def get_all_arguments(self):
        params_all  = super().get_all_arguments()
        params_spe = {
            'rho':self._get_rho_(),
            'cst_field':self.cst_field,
        }
        return {**params_all,**params_spe}

    def get_all_parameters(self):
        pass

    def cost(self,momenta,**kwargs):

        rho = self._get_rho_()
        try:
            device = momenta['momentum_R'].device
        except KeyError:
            device = momenta['momentum_A'].device
        ic(device)
        self.to_device(device)
        self.mp.forward(self.source,momenta,
                        save=False,
                        plot=0,
                        hamiltonian_integration=self.flag_hamiltonian_integration,
                        )
        # Compute the data_term. Default is the Ssd
        self.data_loss = self.data_term()

        if self.flag_hamiltonian_integration:
            self.total_cost = self.data_loss + self.cost_cst * self.mp.ham_integration
        else:
            if self.mp.flag_field:
                # Norm V
                self.norm_v_2 = .5 * rho  * self._compute_V_norm_(momenta['momentum_I'],self.source)

                # Norm L2 on z
                volDelta = prod(self.dx)
                z = sqrt(1 - rho) * (momenta['momentum_I']/volDelta)
                self.norm_l2_on_z = .5 * (z ** 2).sum() * volDelta
            else:
                self.norm_v_2 = torch.tensor([0], device=self.data_loss.device)
                self.norm_l2_on_z = torch.tensor([0], device=self.data_loss.device)

            # Norm L2 on R
            self.norm_l2_on_R = .5 * torch.trace( self.mp._rot_inf_ini.T @ self.mp._rot_inf_ini)
            try:
                self.norm_S_2 = .5 *  ((self.mp.scale_inf_ini)**2).sum()
            except AttributeError:
                self.norm_S_2 = torch.tensor([0], device=self.data_loss.device)

        self.total_cost = self.data_loss + \
                          self.cost_cst * (
                                 self.cst_field *  (self.norm_v_2 +  self.norm_l2_on_z) +
                                 self.cost_affine_cst * (self.norm_l2_on_R + self.norm_S_2)
                          )

        return self.total_cost

    def _rotating_cost_saving_(self,i, loss_stock):


        if loss_stock is None:
            d = 5
            loss_stock = {
            "data_loss":torch.zeros((i,)),
            "norm_v_2":torch.zeros((i,)),
            "norm_l2_on_z":torch.zeros((i,)),
            "norm_l2_on_R":torch.zeros((i,)),
            "norm_S_2":torch.zeros((i,)),
            }
            return loss_stock


        loss_stock["data_loss"][i] = self.data_loss.detach().cpu()
        loss_stock["norm_v_2"][i] = self.norm_v_2.detach().cpu()
        loss_stock["norm_l2_on_z"][i] = self.norm_l2_on_z.detach().cpu()
        loss_stock["norm_l2_on_R"][i] = self.norm_l2_on_R.detach().cpu()
        loss_stock["norm_S_2"][i] = self.norm_S_2.detach().cpu()


        print("\t\tdata_loss :", self.data_loss.detach())
        print("\t\tnorm_v_2 :", self.norm_v_2.detach())
        print("\t\tnorm_l2_on_z :", self.norm_l2_on_z.detach())
        print("\t\tnorm_l2_on_R :", self.norm_l2_on_R.detach())
        print("\t\tnorm_S_2 :", self.norm_S_2.detach())

        return loss_stock



    def plot_cost(self,y_log=False):
        def _handle_old_lossstock_(cost_stock):
            # print(cost_stock)
            if isinstance(cost_stock ,dict):
                return cost_stock
            cost_stock = self.to_analyse[1].detach().numpy()
            loss_stock = {
                "data_loss":cost_stock[:,0],
                "norm_v_2":cost_stock[:,1],
                "norm_l2_on_z":cost_stock[:,2],
                "norm_l2_on_R":cost_stock[:,3],
                # "norm_S_2":cost_stock[:,4],
            }
            return loss_stock

        fig1, ax1 = plt.subplots(1, 2,figsize=(10,5))
        if y_log:
            ax1[0].set_yscale('log')
            ax1[1].set_yscale('log')
        cost_stock = _handle_old_lossstock_(self.to_analyse[1])
        # names= ["data_loss",  "norm_v_2",  "norm_l2_on_z",  "norm_l2_on_R",  "norm_S_2"]
        colors = plt.cm.tab10.colors

        dt = cost_stock["data_loss"]
        nv = cost_stock["norm_v_2"] * self.cost_cst * self.cst_field
        nz = cost_stock["norm_l2_on_z"] * self.cost_cst * self.cst_field
        nr = cost_stock["norm_l2_on_R"] * self.cost_cst * self.cost_affine_cst

        ax1[0].plot(dt, '--', label="data_loss" ,color=colors[0])
        ax1[0].plot(nv,'--', label="norm_v_2" ,color=colors[1])
        ax1[0].plot(nz,'--', label="norm_l2_on_z" ,color=colors[2])
        ax1[0].plot(nr,'--', label="norm_l2_on_R" ,color=colors[3])
        try:
            ns = cost_stock["norm_S_2"] * self.cost_cst * self.cost_affine_cst
            ax1[0].plot(ns,'--', label="norm_S_2" ,color=colors[4])
            total = dt + nv + nz + nr + ns
        except KeyError:
            total = dt + nv + nz + nr
        ax1[0].plot(total, label="sum" ,color="black")
        ax1[0].legend()

        plot_loss_with_multiple_y_axes(cost_stock, "Losses", ax = ax1[1])

        return fig1,ax1


    def compute_DICE(
        self, source_segmentation, target_segmentation, plot=False, forward=True, verbose=True
    ):
        """Compute the DICE score of a regristration. Given the segmentations of
        a structure  (ex: ventricules) that should be present in both source and target image.
        it gives a score close to one if the segmentations are well matching after transformation.
        Compute the Dice scores:
        - Rigid + diffeo
        - Rigid only

        :param source_segmentation: Tensor of source size?
        :param target_segmentation:
        :return: (dict[float]) a dict of DICE scores with the names:
        {
            "reg dice", "rigid dice"
        }
        """
        self.is_DICE_cmp = True
        if len(source_segmentation.shape) == 2 or (len(source_segmentation.shape)) == 3:
            source_segmentation = source_segmentation[None, None]

        self.source_segmentation = source_segmentation
        self.target_segmentation = target_segmentation

        # print(f"diffeo dice : {diffeo_dice}")
        rigidor = self.mp.get_rigidor()
        self.source_seg_rotated = tb.imgDeform(source_segmentation,rigidor,
                                      dx_convention='2square',
                                      mode="nearest"
                                      )
        rotation_dice = tb.average_dice(self.source_seg_rotated,
                                        target_segmentation,
                                        message = "(rotation only)",
                                        verbose= verbose)
        print(f"Rigid dice : {rotation_dice}")

        device = source_segmentation.device
        # Option 1:
        # deformator = self.mp.get_deformator() if forward else self.mp.get_deformation()
        # source_seg_deformed = tb.imgDeform(
        #     self.source_segmentation, deformator.to(device),
        #     dx_convention=self.dx_convention,
        #     mode = 'nearest'
        # )
        #
        # rigidor = self.mp.get_rigidor()
        # self.source_seg_deformed = tb.imgDeform(
        #     source_seg_deformed, rigidor.to(device),
        #     dx_convention=self.dx_convention,
        #     mode = 'nearest'
        # )

        # Option 1 bis:
        deformator = self.mp.get_deformator()
        deformator = self.mp.get_rigidor(deformator)
        self.source_seg_deformed = tb.imgDeform(
            self.source_segmentation, deformator.to(device),
            dx_convention=self.dx_convention,
            mode = 'nearest'
        )
        # Option 2:
        # deformator = self.mp.get_deformator()
        # self.source_seg_deformed = tb.imgDeform(
        #     self.source_seg_rotated, deformator.to(device),
        #     dx_convention=self.dx_convention,
        #     mode = 'nearest'
        # )




        reg_dice = tb.average_dice(self.source_seg_deformed,
                                   target_segmentation,
                                   message = "(all)",
                                   verbose= verbose)
        self.dice = (rotation_dice, reg_dice)
        T,C,D,H,W = source_segmentation.shape
        if plot:
            fig, ax = plt.subplots(2,4)
            ax[0,0].imshow(source_segmentation[0,0, D//2].detach().cpu(), cmap=DLT_SEG_CMAP)
            ax[0,0].set_title('Source')
            ax[0,1].imshow(target_segmentation[0,0, D//2].detach().cpu(), cmap=DLT_SEG_CMAP)
            ax[0,1].set_title('Target')
            ax[1,0].imshow(self.source_seg_deformed[0,0, D//2].detach().cpu(), cmap=DLT_SEG_CMAP)
            ax[1,0].set_title('Deformed')
            ax[1,1].imshow(self.source_seg_rotated[0,0, D//2].detach().cpu(), cmap=DLT_SEG_CMAP)
            ax[1,1].set_title('Rotated')

            st = tb.SegmentationComparator()(source_segmentation[:,:,D//2].detach().cpu(), target_segmentation[:,:,D//2])
            ax[0,2].imshow(st[0])
            ax[0,2].set_title('source vs target')
            rt = tb.SegmentationComparator()(
                self.source_seg_rotated[:,:,D//2],
                target_segmentation[:,:,D//2])
            ax[1,2].imshow(rt[0])
            ax[1,2].set_title('rot vs target')
            dt = tb.SegmentationComparator()(
                self.source_seg_deformed[:,:,D//2],
                target_segmentation[:,:,D//2])
            ax[1,3].imshow(dt[0])
            ax[1,3].set_title('def vs target')

            plt.show()
        return (rotation_dice, reg_dice), (self.source_seg_rotated, self.source_seg_deformed)
