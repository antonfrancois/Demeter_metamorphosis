"""
Rotate.py
"""
from logging import warning
from numbers import Number

import matplotlib.pyplot as plt
import torch

import __init__
from math import prod, sqrt


from demeter.utils.decorators import time_it
from demeter.metamorphosis import Geodesic_integrator, Optimize_geodesicShooting

from demeter.constants import *
import demeter.utils.torchbox as tb
from demeter.utils.toolbox import plot_loss_with_multiple_y_axes


def prepare_momenta(image_shape,
                    diffeo: bool = True,
                    rotation: bool = True,
                    translation: bool = True,
                    scaling: bool = True,
                    affine: bool = False,
                    rot_prior=None,
                    trans_prior=None,
                    scale_prior=None,
                    affine_prior=None,
                    device="cuda:0",
                    requires_grad=True):
    dim = 2 if len(image_shape) == 4 else 3
    if affine:
        scaling = rotation = False
        translation = True
        warning("affine is true, scaling and rotation set to False, translation set to True")

        if affine_prior is None:
            if rot_prior is not None:
                affine_prior = rot_prior
            else:
                affine_prior = torch.zeros((dim, dim))
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
        "dtype": torch.float32,
        "device": device
    }
    if diffeo:
        momenta["momentum_I"] = torch.zeros(image_shape, **kwargs)
    if rotation:
        if len(rot_prior.shape) == 2:
            momenta["momentum_R"] = rot_prior.to(kwargs["dtype"]).to(kwargs["device"])
        elif len(rot_prior.shape) <= 1:
            if dim == 2:
                momenta["momentum_R"] = torch.tensor(
                    [[0, rot_prior],
                     [-rot_prior, 0],
                     ],
                    dtype=torch.float32, device='cuda:0')
            elif dim == 3:
                r1, r2, r3 = rot_prior
                momenta["momentum_R"] = torch.tensor(
                    [[0, -r1, -r2],
                     [r1, 0, -r3],
                     [r2, r3, 0]],
                    dtype=torch.float32, device='cuda:0')
        else:
            raise ValueError("Rotation prior must be 2 or 1 dimensional")
    if affine:
        if len(affine_prior.shape) == 2:
            momenta["momentum_A"] = affine_prior.to(kwargs["dtype"]).to(kwargs["device"])
        elif len(affine_prior.shape) <= 1:
            if dim == 2:
                momenta["momentum_A"] = torch.tensor(
                    [[0, affine_prior],
                     [-affine_prior, 0],
                     ],
                    dtype=torch.float32, device='cuda:0')
            elif dim == 3:
                r1, r2, r3 = affine_prior
                momenta["momentum_A"] = torch.tensor(
                    [[0, -r1, -r2],
                     [r1, 0, -r3],
                     [r2, r3, 0]],
                    dtype=torch.float32, device='cuda:0')
        else:
            raise ValueError("Rotation prior must be 2 or 1 dimensional")
        # momenta["momentum_A"] = affine_prior.to(kwargs["dtype"]).to(kwargs["device"])
    if translation:
        momenta["momentum_T"] = trans_prior.to(kwargs["dtype"]).to(kwargs["device"])
    if scaling:
        momenta["momentum_S"] = scale_prior.to(kwargs["dtype"]).to(kwargs["device"])

    for keys in momenta.keys():
        momenta[keys].requires_grad = requires_grad

    return momenta


class Affine_Metamorphosis_integrator(Geodesic_integrator):
    """


    """

    def __init__(self, rho, constraints=True, **kwargs):
        super().__init__(**kwargs)
        self.rho = rho
        # self.n_step = n_step
        self.constraints = constraints

    def _get_rho_(self):
        return float(self.rho)

    def to_device(self, device):
        try:
            self.rot_mat = self.rot_mat.to(device)
            self.translation = self.translation.to(device)
            self.scale = self.scale.to(device)
        except AttributeError:
            pass
        super().to_device(device)

    def projection(self, c, p):
        cp = (c * p).sum()
        norm_c = (c ** 2).sum()
        if norm_c != 0:
            cst = c * cp / norm_c
        else:
            cst = 0
        return p - cst

    def _contrainte_(self, momentum_I, source):
        grad_source = tb.spatialGradient(source, dx_convention=self.dx_convention)
        IgradI_x = tb.multiply_grid_vectors(tb.im2grid(grad_source[0]), self.id_grid)
        x_IgradI = tb.multiply_grid_vectors(self.id_grid, tb.im2grid(grad_source[0]))

        # contrainte rotation
        if self._dim == 2:
            c_list = [(IgradI_x - x_IgradI)[..., 0, 1][None]]
        elif self._dim == 3:
            _k = [0, 0, 1]
            _l = [1, 2, 2]
            c_list = (IgradI_x - x_IgradI)[..., _k, _l].permute(4, 0, 1, 2, 3)
            c_list = [c for c in c_list]

        #contrainte translation
        for i in range(grad_source.shape[2]):
            c_list.append(grad_source[:, :, i])

        # Orthonormaliser la liste
        c_ortho_list = [c_list[0] / (c_list[0] ** 2).sum().sqrt()]
        if len(c_list) > 1:
            for c in c_list[1:]:
                c_tilde = c
                for co in c_ortho_list:
                    c_tilde -= (c * co).sum() * co
                c_norm = (c_tilde ** 2).sum().sqrt()
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
            assert (c_ortho_list[0] * c_ortho_list[
                1]).sum() < 1e-5, f"(c_otho_list[0] * c_otho_list[1]).sum() = {(c_ortho_list[0] * c_ortho_list[1]).sum()}"
            assert (c_ortho_list[0] * c_ortho_list[
                2]).sum() < 1e-5, f"(c_otho_list[0] * c_otho_list[2]).sum() = {(c_ortho_list[0] * c_ortho_list[2]).sum()}"
            assert (c_ortho_list[2] * c_ortho_list[
                1]).sum() < 1e-5, f"(c_otho_list[2] * c_otho_list[1]).sum() = {(c_ortho_list[2] * c_ortho_list[1]).sum()}"
        # for i, c in enumerate(c_list):
        #     assert (c * momentum_I).sum() < 1e-4, f"(c_{i} * momentum_I).sum() = {(c * momentum_I).sum()}"
        # if (c * momentum_I).sum() > 1e-5:
        #     print( f"(c_{i} * momentum_I).sum() = {(c * momentum_I).sum()}")

        return momentum_I

    def _compute_step_affine(self,
                             momentum_A, A_mat,
                             momentum_T, translation,
                             ):

        # 1. Compute infinitesimal affine mat M
        d_affine = momentum_A @ A_mat.T + momentum_T @ translation.T # d_affine = M

        if self._i == 0:
            self._rot_inf_ini = d_affine.clone()

        #  norm_l2_on_A

        norm_l2_on_A = .5 * torch.trace(d_affine.T @ self._rot_inf_ini)
        if self.debug:
            ic(self._i, momentum_A, momentum_T,
               exp_A, A_mat,
               translation, norm_l2_on_A
               )

        # Momenta update
        # 6. momentum_T (p^\tau)
        momentum_T = momentum_T - (d_affine.T @ momentum_T)/self.n_step

        # 7. momentum_A (p^M):
        momentum_A = (momentum_A - d_affine.T @ momentum_A / self.n_step)

        # X. Compute affine mat (A)
        exp_A = torch.linalg.matrix_exp(d_affine / self.n_step)
        A_mat = exp_A @ A_mat

        # X. Compute translation = b
        translation = translation + (d_affine @ translation + momentum_T) / self.n_step

        if self.debug:
            print("A_mat :", A_mat)
            print("translation :", translation)
        # momentum_A -= torch.autograd.grad(field,momentum_A)
        # momentum_T = momentum_T # Momentum T is constant
        return momentum_A, momentum_T, A_mat, translation, norm_l2_on_A

    def step(self, image, momentum_I, momentum_A, momentum_T, A_mat, translation):
        """
        One integration step. Fully checkpoint-compliant: fixed number of outputs.
        """
        if self.debug:
            print("\n" + "=" * 25)
            print('step', self._i)

        # --- Apply constraints ---
        if self._i == 0 and self.constraints:
            momentum_I = self._contrainte_(momentum_I, self.source)

        # --- Vector field and residuals ---
        # grad_image = tb.spatialGradient(image, dx_convention=self.dx_convention)
        # field, norm_V = self._compute_vectorField_(momentum_I, grad_image)
        # field = self._update_field_(momentum_I, image)

        # --- Affine part update, not depending on I---
        momentum_A, momentum_T, A_mat, translation, norm_l2_on_A = \
            self._compute_step_affine(
                momentum_A, A_mat,
                momentum_T, translation
        )

        # 3. Compute field (v)
        grad_image = tb.spatialGradient(image, dx_convention=self.dx_convention)
        # ic(grad_image.min().item(), grad_image.max().item(),self.dx_convention)
        field_momentum = (grad_image * momentum_I.unsqueeze(2)).sum(dim=1)
        field =  self.kernelOperator(field_momentum)
        norm_v = None
        if self.flag_hamiltonian_integration:
            self.norm_v_i = .5 * self.rho * (field_momentum.clone() * field.clone()).sum()

        field = -tb.im2grid(field)

        # 4. Compute image (I)
        deformation = self.id_grid - self.rho * field / self.n_step
        if self._get_rho_() != 1:
            raise NotImplementedError("Residuals are not implemented nor theorized yet in this settings")
        else:
            residuals = torch.zeros_like(image)
        image = self._update_image_semiLagrangian_(momentum_I, image, deformation, residuals)

        # Momenta update
        N = prod(image.shape[2:])
        d = len(image.shape[2:])
        B = image.shape[0]
        if B != 1:
            Warning(
                "_compute_step_affine_ does not support batch, B must be 1 got {}. Only first dim is considered".format(
                    B))
            B = 1

        # 5. momentum_I (p^I)
        momentum_I = self._compute_div_momentum_semiLagrangian_(
            deformation,
            momentum_I,
            cst = -sqrt(self.rho),
            field = sqrt(self.rho) * field
        )

        # 6. momentum_T (p^\tau)
        inv_A_mat = torch.linalg.inv(A_mat)
        field_momentum = field_momentum.reshape(B, d, 1, N)  # [B, (2,3), 1, H * W (* D)] R^(d x 1)
        jaco_field = tb.field_jacobian(field).reshape(B, d, d, N)
        inv_A_matT = inv_A_mat.T

        momentum_T += (inv_A_matT @ torch.einsum("ijn, jn -> i", # TODO : check efficiency einsum
             jaco_field[0].transpose(0,1),
             field_momentum[0,:,0]
             )) / self.n_step

        # 7. momentum_A (p^M):
        # >>> Integral 1
        # inv_A_matT = inv_A_mat.T
        field_flat = field.reshape(B, 1, d, N)  # [B, 1, (2,3), H * W (* D)] R^(1 x d)
        integral_1 =  (field_momentum * field_flat).sum(dim=-1)[0]

        # >>>> Integral 2
        x_flat = self.id_grid.reshape(B, 1, d, N)
        integral_2 = torch.einsum("ijn, jkn -> ik",
                                          jaco_field[0].transpose(0,1),
                                          (field_momentum * x_flat)[0]
                                )

        momentum_A -= inv_A_matT @ (integral_1 - integral_2)



        # --- Always output the same things ---
        return (
            momentum_I,
            momentum_A,
            momentum_T,
            image,
            self.rho * field,
            residuals,
            A_mat,
            translation,
        )

    def _forward_initialize_integration(self, image, momenta, device, save, sharp, hamiltonian_integration, plot):
        # self.debug = True
        if self.debug:
            ic("debug is defined here", self.debug)

        self._dim = 2 if len(image.shape) == 4 else 3
        self.rot_mat = torch.eye(self._dim)

        if "momentum_R" in momenta.keys():
            momenta["momentum_A"] = momenta["momentum_R"]
            del momenta["momentum_R"]

        device = momenta['momentum_A'].device
        self.to_device(device)

        self.translation = torch.zeros((self._dim,), device=device)

        self.flag_field = True if "momentum_I" in momenta.keys() else False

        super()._forward_initialize_integration(image, momenta, device, save, sharp, hamiltonian_integration, plot)

    def _forward_direct_step(self):
        # print("_forward_direct_step in rotate")
        if "momentum_A" in self.momenta.keys():
            momentum_A = self.momenta["momentum_A"]

            flag_affine = True
            momentum_R = momentum_S = None
        else:
            raise ValueError("Momenta must contain at least momentum_A.")
        if "momentum_T" in self.momenta.keys():
            momentum_T = self.momenta["momentum_T"]
        else:
            momentum_T = torch.zeros((self._dim,), device=self.device)

        if self.flag_field:
            momentum_I = self.momenta["momentum_I"]
            momentum_I, momentum_A, momentum_T, self.image, self.field, self.residuals, self.rot_mat, self.translation \
                = self.step(
                self.image,
                momentum_I,
                momentum_A,
                momentum_T,
                self.rot_mat,
                self.translation,
            )
            self.momenta["momentum_I"] = momentum_I
            self.momenta["momentum_A"] = momentum_A
            if momentum_T is not None:
                self.momenta["momentum_T"] = momentum_T
        else:
            momentum_A, momentum_T, self.rot_mat, self.translation, norm_l2_on_A = self._compute_step_affine(
                momentum_A, self.rot_mat,
                momentum_T, self.translation,
            )
            self.momenta["momentum_A"] = momentum_A
            self.momenta["momentum_T"] = momentum_T


    def _forward_checkpointed_step(self):
        raise NotImplementedError("To rework")
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

    def get_affine_deformator(self, grid=None):
        """
        return a grid ready to apply the rotation and translation estimated

        Example:
        ---------
        >>>rot_def = mr.mp.get_affine_deformator()
        >>>rotated_source = tb.imgDeform(source,rot_def,dx_convention='2square')
        """
        if grid is None:
            grid = self.id_grid
        mat = torch.linalg.inv(self.rot_mat)
        return tb.grid_from_rotation_translation(grid, mat, - self.translation)

    def get_affine_deformation(self, grid=None):
        if grid is None:
            grid = self.id_grid
        return tb.grid_from_rotation_translation(
            grid, self.rot_mat.to(grid.device), self.translation.to(grid.device))

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
        momentum = self.momenta['momentum_I']
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
                self.get_deformation(to_t=t + 1),
                add_grid=False,
                ax=ax[i, -1],
                step=int(min(self.field_stock.shape[2:-1]) / 30),
                check_diffeo=True,
                dx_convention=self.dx_convention,
            )

        return fig, ax

    def plot_rot(self):
        fig, ax = plt.subplots(1, 2)

        shape = self.source.shape[2:]
        id_grid = tb.make_regular_grid(shape, dx_convention="2square")
        rot = self.mp.rot_mat
        rot_grid_end = tb.matrix_time_grid(id_grid, rot)
        ax[0].imshow(self.mp.image[0, 0], cmap='gray', origin="lower")
        tb.gridDef_plot_2d(rot_grid_end,
                           ax=ax[0],
                           step=25,
                           dx_convention="2square",
                           color='red')

        source_rot = tb.imgDeform(self.source, rot_grid_end, dx_convention="2square")
        ax[1].imshow(
            tb.imCmp(
                source_rot,
                self.target,
                method="seg"
            ),
            cmap='gray', origin="lower"
        )
        ax[1].set_title("rotated_source vs target")

        plt.show()


class Affine_Metamorphosis_Optimizer(Optimize_geodesicShooting):

    def __init__(
            self,
            cost_field_cst=1,
            cost_affine_cst=1,
            adam_dt_step_field=None,
            adam_dt_step_affine=None,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self._cost_saving_ = self._rotating_cost_saving_
        self.cost_field_cst = cost_field_cst
        self.cost_affine_cst = cost_affine_cst
        self.adam_dt_step_field = adam_dt_step_field
        self.adam_dt_step_affine = adam_dt_step_affine

    def _get_rho_(self):
        return float(self.mp.rho)

    def get_all_arguments(self):
        params_all = super().get_all_arguments()
        params_spe = {
            'rho': self._get_rho_(),
            'cost_field_cst': self.cost_field_cst,
            'cost_affine_cst': self.cost_affine_cst,
            'adam_dt_step_field': self.adam_dt_step_field,
            'adam_dt_step_affine': self.adam_dt_step_affine,
        }
        return {**params_all, **params_spe}

    def get_all_parameters(self):
        pass

    def _initialize_Adam_(self, dt_step):
        """Adam with optional per-block step sizes for field vs affine momenta."""

        def _as_float_lr(value, name):
            if isinstance(value, Number):
                return float(value)
            if isinstance(value, torch.Tensor) and value.numel() == 1:
                return float(value.item())
            if isinstance(value, (list, tuple)) and len(value) == 1:
                return _as_float_lr(value[0], name)
            raise TypeError(
                f"{name} must be a scalar float-like value, got {type(value)}: {value}"
            )

        if not isinstance(self.parameter, dict):
            return super()._initialize_Adam_(dt_step)

        base_lr = _as_float_lr(dt_step, "dt_step")
        field_lr = (
            base_lr
            if self.adam_dt_step_field is None
            else _as_float_lr(self.adam_dt_step_field, "adam_dt_step_field")
        )
        affine_lr = (
            base_lr
            if self.adam_dt_step_affine is None
            else _as_float_lr(self.adam_dt_step_affine, "adam_dt_step_affine")
        )

        field_params = []
        affine_params = []
        for k, p in self.parameter.items():
            if not isinstance(p, torch.Tensor):
                continue
            if k == "momentum_I":
                field_params.append(p)
            elif k in {"momentum_R", "momentum_T", "momentum_S", "momentum_A"}:
                affine_params.append(p)

        param_groups = []
        if field_params:
            param_groups.append({"params": field_params, "lr": field_lr})
        if affine_params:
            param_groups.append({"params": affine_params, "lr": affine_lr})

        if not param_groups:
            return super()._initialize_Adam_(dt_step)

        self.optimizer = torch.optim.Adam(
            param_groups,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0,
        )

    def cost(self, momenta, **kwargs):

        rho = self._get_rho_()

        device = momenta['momentum_A'].device
        # self.flag_decoupled = False
        # self.flag_affine = True
        self.to_device(device)

        self.mp.forward(self.source, momenta,
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
                self.norm_v_2 = .5 * rho * self._compute_V_norm_(momenta['momentum_I'], self.source)

                # Norm L2 on z
                volDelta = prod(self.dx)
                z = sqrt(1 - rho) * (momenta['momentum_I'] / volDelta)
                self.norm_l2_on_z = .5 * (z ** 2).sum() * volDelta

            else:
                self.norm_v_2 = torch.tensor(0, device=self.data_loss.device)
                self.norm_l2_on_z = torch.tensor(0, device=self.data_loss.device)



            # torch.trace(momenta["momentum_A"] @ A_mat.T + momenta["momentum_T"] @ translation.T)
            self.norm_A  = .5 * torch.trace(momenta["momentum_A"])
            # Stable L2 norm: avoid undefined gradient at exactly zero.
            self.norm_T = .5 * torch.sqrt((momenta["momentum_T"] ** 2).sum() + 1e-12)

                # if self.mp.flag_field:
                #     # Norm V
                #     self.norm_v_2 = .5 * rho *

        self.total_cost = self.data_loss + \
                          self.cost_cst * (
                            self.cost_field_cst * (self.norm_v_2 + self.norm_l2_on_z) +
                            self.cost_affine_cst * (self.norm_A + self.norm_T)
                          )
        return self.total_cost

    def _rotating_cost_saving_(self, i, loss_stock):

        if loss_stock is None:
            d = 5
            loss_stock = {
                "data_loss": torch.zeros((i,)),
                "norm_v_2": torch.zeros((i,)),
                "norm_l2_on_z": torch.zeros((i,)),
                "norm_A": torch.zeros((i,)),
                "norm_T": torch.zeros((i,)),
            }
            return loss_stock

        loss_stock["data_loss"][i] = self.data_loss.detach().cpu()
        loss_stock["norm_v_2"][i] = self.norm_v_2.detach().cpu()
        loss_stock["norm_l2_on_z"][i] = self.norm_l2_on_z.detach().cpu()
        loss_stock["norm_A"][i] =  self.norm_A.detach().cpu()
        loss_stock["norm_T"][i] =  self.norm_T.detach().cpu()

        # print("\t\tdata_loss :", self.data_loss.detach())
        # print("\t\tnorm_v_2 :", self.norm_v_2.detach())
        # print("\t\tnorm_l2_on_z :", self.norm_l2_on_z.detach())
        # print("\t\tnorm_l2_on_R :", self.norm_l2_on_R.detach())
        # print("\t\tnorm_S_2 :", self.norm_S_2.detach())

        return loss_stock

    def plot_cost(self, y_log=False, verbose=False):
        def _handle_old_lossstock_(cost_stock):
            # print(cost_stock)
            if isinstance(cost_stock, dict):
                return cost_stock
            cost_stock = self.to_analyse[1].detach().numpy()
            loss_stock = {
                "data_loss": cost_stock[:, 0],
                "norm_v_2": cost_stock[:, 1],
                "norm_l2_on_z": cost_stock[:, 2],
                "norm_l2_on_R": cost_stock[:, 3],
                # "norm_S_2":cost_stock[:,4],
            }
            return loss_stock

        fig1, ax1 = plt.subplots(1, 2, figsize=(10, 5))
        if y_log:
            ax1[0].set_yscale('log')
            ax1[1].set_yscale('log')
        cost_stock = _handle_old_lossstock_(self.to_analyse[1])
        if verbose:
            for ls in cost_stock:
                print(ls)
        # names= ["data_loss",  "norm_v_2",  "norm_l2_on_z",  "norm_l2_on_R",  "norm_S_2"]
        colors = plt.cm.tab10.colors

        dt = cost_stock["data_loss"]
        nv = cost_stock["norm_v_2"] * self.cost_cst * self.cost_field_cst
        nz = cost_stock["norm_l2_on_z"] * self.cost_cst * self.cost_field_cst
        nr = cost_stock["norm_A"] * self.cost_cst * self.cost_affine_cst

        ax1[0].plot(dt, '--', label="data_loss", color=colors[0])
        ax1[0].plot(nv, '--', label="norm_V", color=colors[1])
        ax1[0].plot(nz, '--', label="norm_Z", color=colors[2])
        ax1[0].plot(nr, '--', label="norm_A", color=colors[3])
        try:
            ns = cost_stock["norm_T"] * self.cost_cst * self.cost_affine_cst
            ax1[0].plot(ns, '--', label="norm_T", color=colors[4])
            total = dt + nv + nz + nr + ns
        except KeyError:
            total = dt + nv + nz + nr
        ax1[0].plot(total, label="sum", color="black")
        ax1[0].legend()

        plot_loss_with_multiple_y_axes(cost_stock, "Losses", ax=ax1[1])

        fig1.suptitle(
            f"cost_cst = {self.cost_cst:.2f}, affine_cst = {self.cost_affine_cst:.2f}, field_cst = {self.cost_field_cst:.2f}")

        return fig1, ax1

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
        rigidor = self.mp.get_affine_deformation()
        self.source_seg_rotated = tb.imgDeform(source_segmentation, rigidor,
                                               dx_convention='2square',
                                               mode="nearest"
                                               )
        rotation_dice = tb.average_dice(self.source_seg_rotated,
                                        target_segmentation,
                                        message="(rotation only)",
                                        verbose=verbose)
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
        deformator = self.mp.get_affine_deformator(deformator)
        self.source_seg_deformed = tb.imgDeform(
            self.source_segmentation, deformator.to(device),
            dx_convention=self.dx_convention,
            mode='nearest'
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
                                   message="(all)",
                                   verbose=verbose)
        self.dice = (rotation_dice, reg_dice)
        T, C, D, H, W = source_segmentation.shape
        if plot:
            fig, ax = plt.subplots(2, 4)
            ax[0, 0].imshow(source_segmentation[0, 0, D // 2].detach().cpu(), cmap=DLT_SEG_CMAP)
            ax[0, 0].set_title('Source')
            ax[0, 1].imshow(target_segmentation[0, 0, D // 2].detach().cpu(), cmap=DLT_SEG_CMAP)
            ax[0, 1].set_title('Target')
            ax[1, 0].imshow(self.source_seg_deformed[0, 0, D // 2].detach().cpu(), cmap=DLT_SEG_CMAP)
            ax[1, 0].set_title('Deformed')
            ax[1, 1].imshow(self.source_seg_rotated[0, 0, D // 2].detach().cpu(), cmap=DLT_SEG_CMAP)
            ax[1, 1].set_title('Rotated')

            st = tb.SegmentationComparator()(source_segmentation[:, :, D // 2].detach().cpu(),
                                             target_segmentation[:, :, D // 2])
            ax[0, 2].imshow(st[0])
            ax[0, 2].set_title('source vs target')
            rt = tb.SegmentationComparator()(
                self.source_seg_rotated[:, :, D // 2],
                target_segmentation[:, :, D // 2])
            ax[1, 2].imshow(rt[0])
            ax[1, 2].set_title('rot vs target')
            dt = tb.SegmentationComparator()(
                self.source_seg_deformed[:, :, D // 2],
                target_segmentation[:, :, D // 2])
            ax[1, 3].imshow(dt[0])
            ax[1, 3].set_title('def vs target')

            plt.show()
        return (rotation_dice, reg_dice), (self.source_seg_rotated, self.source_seg_deformed)
