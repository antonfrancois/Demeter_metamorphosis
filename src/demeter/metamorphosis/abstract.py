r"""
The implementation of Metamorphoses in **Demeter** is based on the minimization of a Hamiltonian:
$$H(q,p,v,z) =  (p|\dot q) - R(v,z)$$

where $q : (\Omega, [0,1]) \mapsto \mathcal M$ is the temporal image valued in $\mathcal M$, $R$ is a regularization function, $v$ is a vector field, and $z$ is a control on the photometric part.

In the case of LDDMM and considering $\mathcal M = \mathbb R$, the Hamiltonian is:
$$H(q,p,v,z) =  (p|\dot q) - \frac 12\|v\|_V^2 - \frac 12\|z\|_Z^2$$

An optimal trajectory or geodesic under the conditions given by $H$ is:

$$\left\{\begin{array}{rl} \dot q_t &= - \nabla q_t \cdot v_t + z_t\\ \dot z_t &= - \mathrm{div}(z_t  v_t) \\
p_t &= z_t\\
v_t &= -K_V\left( z_t\nabla q_t \right)  \end{array}\right.$$

These equations are written in the continuous case. In this document, all discretization choices made during the implementation are detailed.

To solve the registration problem, a geodesic shooting strategy is used. For this, a relaxed version of $H$ is minimized:
$$E(p_0) = D_T(I_1) + \frac{\lambda}{2} \left( \|v_0\|_V^2 +\|z_0\|_Z^2  \right)$$

Where $D_T$ is a data attachment term and $T$ is a target image, $I_1$ is the image at the end of the geodesic integration, and $p_0$ is the initial momentum. Note that in the case of Metamorphoses valued in images, $p = z$.

You may have noticed that in the above equation $E(p_{0})$ depends only on the initial momentum. Indeed, thanks to a conservation property of norms during the calculation of optimal trajectories in a Hamiltonian which states: Let $v$ and $z$ follow a geodesic given by $H$, then
$$\forall t \in [0,1], \|v_{0}\|^2_{V} = \|v_{t}\|^2_{V}; \|v_{0}\|^2_{2} = \|z_{t}\|^2_{2}. $$

This property is used to save computation time. In practice, due to numerical scheme choices, norm conservation may not be achieved. In this case, it is possible to optimize over the set of norms and $E$ becomes:
$$E(p_0) = D_T(I_1) + \frac \lambda2 \int_{0}^1 \left( \|v_t\|_V^2 +\|z_t\|_Z^2  \right) dt.$$

The $I_{t},v_t,z_{t}$ are still deduced from $p_0$. It is possible to switch between the two in the code using the `hamiltonian_integration` option in the children of `Optimize_geodesicShooting`.
"""


import torch
import matplotlib.pyplot as plt
import warnings
from math import prod, sqrt
import pickle
import gc
import os, sys, csv  # , time
from icecream import ic

from datetime import datetime
from abc import ABC, abstractmethod

from torch.onnx.symbolic_opset9 import detach

from ..utils.optim import GradientDescent
from demeter.constants import *
from ..utils import torchbox as tb
from ..utils import vector_field_to_flow as vff
from ..utils.toolbox import (
    update_progress,
    fig_to_image,
    save_gif_with_plt,
)
from ..utils.decorators import time_it, monitor_gpu
from ..utils import cost_functions as cf
from ..utils import fill_saves_overview as fill_saves_overview

from ..metamorphosis import data_cost as dt
from .var_classes import Momenta, TorchDataClass

# TO DRAW THE BACKWARD GRAPH
# from torchviz import make_dot

# =========================================================================
#
#            Abstract classes
#
# =========================================================================
# See them as a toolkit

def _get_device_from_momenta(momenta_ini: TorchDataClass):
    if isinstance(momenta_ini, TorchDataClass):
        for tensor in momenta_ini.as_dict().values():
            if tensor.is_cuda:
                return tensor.device
        # fall back to first tensor device or cpu
        first = next(iter(momenta_ini.as_dict().values()), None)
        return first.device if first is not None else "cpu"
    raise TypeError(
        "momenta_ini must be a TorchDataClass instance, "
        f"got {type(momenta_ini)}"
    )


def _momenta_to_device(momenta: TorchDataClass, device: str) -> TorchDataClass:
    return type(momenta)(**{k: v.to(device) for k, v in momenta.as_dict().items()})


def _momenta_detach(momenta: TorchDataClass) -> TorchDataClass:
    return type(momenta)(
        **{k: v.detach().clone() for k, v in momenta.as_dict().items()}
    )


def _zero_like_momenta(momenta: Momenta, device: str) -> Momenta:
    return Momenta(**{k: torch.zeros_like(v).to(device) for k, v in momenta.as_dict().items()})


def _primary_tensor(momenta: Momenta) -> torch.Tensor:
    values = momenta.as_dict().values()
    if not values:
        raise ValueError("Momenta has no tensor fields")
    return next(iter(values))

def free_GPU_memory(mr):
    mr.to_device('cpu')
    del mr
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

class Geodesic_integrator(torch.nn.Module, ABC):
    """The Geodesic_integrator class is an abstract class that inherits from
    torch.nn.Module and ABC (Abstract Base Class). It is designed to define
    the way of integrating geodesics in the context of metamorphosis optimization.
    If you want to implement a new geodesic integrator, you inherit from this class
    and implement the abstract methods, with a focus on the step method which
    contains the code numerical scheme for a step of the geodesic integration.

     Here are the main features of the class:
    - Initialization:
        The constructor initializes the basic parameters needed for
        geodesic integration, such as the kernel operator (kernelOperator),
         the number of steps (n_step), and the spatial differentiation convention
          (dx_convention).
    - Abstract Methods:
        The class contains abstract methods like step, which
        must be implemented by derived classes to define the specific steps of the
        integration.
    - Temporal Integration:
        The forward method performs the temporal loop
        using the appropriate _step_ method to integrate the source image along
        the geodesic.
    - Generic functions useful for integration:
        Methods _compute_vectorField_,
        _update_image_semiLagrangian_ ot _compute_vectorField_multimodal_
        implements updates of the field, the momentum and the image.
    - Plots and Visualization:
        The class includes methods for visualizing the
        integration results, such as plot, plot_deform, and save_to_gif.

    .. note::
        The integrated plot and visualization methods are implemented for 2d
        images only. If you want to use them for 3d images, you need to use
         others functions like the ones in image_3d_visualization.py.


    Parameters
    ----------
    kernelOperator : reproducing_kernel.ReproducingKernel
        The kernel operator used to compute the vector field.
    n_step : int
        The number of steps for the geodesic integration.
    dx_convention : str, optional
        The spatial differentiation convention, by default "pixel".
    save_gpu_memory : bool, optional
        If True, we use torch.checkpoints to use less gpu memory. We save memory
        by avoiding storing the gradient graph and recomputing it at each iteration.
        Setting to True, make the code slower but allows to parse bigger images.

    """

    @abstractmethod
    def __init__(self,
                 kernelOperator,
                 n_step,
                 dx_convention="pixel",
                 save_gpu_memory = False,
                 debug= False,
                 **kwargs):
        super().__init__()
        self._force_save = False
        self._detach_image = True
        self.dx_convention = dx_convention
        self.save_gpu_memory = save_gpu_memory
        self.debug = debug

        self.kernelOperator = kernelOperator
        self.n_step = n_step


    def _init_sharp_(self, sharp):
        # print(f'sharp = {sharp}')
        if sharp is None:
            try:
                sharp = self.flag_sharp
            except AttributeError:
                sharp = False
        if not sharp:
            self.flag_sharp = False
            return 0
        if self.__class__.__name__ == "Metamorphosis_path":
            self.step = self._step_sharp_semiLagrangian
        self.flag_sharp = True
        self.save = True
        self._force_save = True
        self._phis = [[None] * i for i in range(1, self.n_step + 1)]
        self._resi_deform = []

    @abstractmethod
    def step(self, image, momentum):
        pass

    def check_nan(self, input):
        if isinstance(input, torch.Tensor):
            if input.isnan().any():
                raise OverflowError("Some nan where produced ! the integration diverged",
                                "changing the parameters is needed. "
                                "You can try:"
                                "\n- increasing n_step (deformation more complex"
                                "\n- decreasing grad_coef (convergence slower but more stable)"
                                "\n- increasing sigma_v (catching less details)"
                                )
        elif isinstance(input, Momenta):
            if input.has_nan():
                raise OverflowError("Some nan where produced ! the integration diverged,"
                                    f"Momentum nan report {input.nan_report()}",)

    def _test_nan_(self, *tensors):

        for tensor in tensors:
            if isinstance(tensor, torch.Tensor):
                self.check_nan(tensor)
            elif isinstance(tensor, Momenta):
                for v in tensor.as_dict().values():
                    self.check_nan(v)

    def _flatten_momenta(self,momenta_dict):
        flat_tensors = []
        shapes = {}
        for k, v in momenta_dict.items():
            shapes[k] = v.shape
            flat_tensors.append(v.reshape(-1))
        flat = torch.cat(flat_tensors)
        self.momenta_shape = shapes
        return flat


    def _unflatten_momenta(self,flat_tensor, device=None):
        out = {}
        idx = 0
        for k, shape in self.momenta_shape.items():
            numel = torch.tensor(shape).prod().item()
            out[k] = flat_tensor[idx:idx+numel].reshape(shape)
            if device is not None:
                out[k] = out[k].to(device)
            idx += numel
        return out

    # def forward(self,
    #              image,
    #               momenta,
    #             save=True,
    #             plot=0,
    #             t_max=1,
    #             verbose=False,
    #             sharp=None,
    #             # debug=False,
    #             hamiltonian_integration=False
    #     ):
    #     r""" This method is doing the temporal loop using the good method `_step_`
    #
    #     Parameters
    #     ----------
    #     image : tensor array of shape [1,1,H,W]
    #         Source image ($I_0$)
    #     momentum_ini : tensor array of shape [1,1,H,W]
    #         Momentum ($p_0$) or residual ($z_0$)
    #     save : bool, optional
    #         Option to save the integration intermediary steps, by default True
    #         it saves the image, field and momentum at each step in the attributes
    #         `image_stock`, `field_stock`, `residuals_stock` and `momentum_stock`.
    #     plot : int, optional
    #         Positive int lower than `self.n_step` to plot the indicated number of
    #         intermediary steps, by default 0
    #     t_max : int, optional
    #         The integration will be made on [0,t_max], by default 1
    #     verbose : bool, optional
    #         Option to print the progress of the integration, by default False
    #     sharp : bool, optional
    #         Option to use the sharp integration, by default None
    #     debug : bool, optional
    #         Option to print debug information, by default False
    #     hamiltonian_integration : bool, optional
    #         Choose to integrate over first time step only or whole hamiltonian, in
    #         practice when True, the Regulation norms of the Hamiltonian are computed
    #         and saved in the good attributes (usually `norm_v` and `norm_z`),
    #          by default False
    #
    #     """
    #     # if len(momenta.shape) not in [4, 5]:
    #     #     raise ValueError(f"residual_ini must be of shape [B,C,H,W] or [B,C,D,H,W] got {momenta.shape}")
    #     device = next((tensor.device for tensor in momenta.values() if tensor.is_cuda), 'cpu')
    #     # print(f'sharp = {sharp} flag_sharp : {self.flag_sharp},{self._phis}')
    #     self._init_sharp_(sharp)
    #     self.source = image.detach().to(device)
    #     self.image = image.clone().to(device)
    #     self.momenta = momenta
    #     # self.debug = debug
    #     self.flag_hamiltonian_integration = hamiltonian_integration
    #     try:
    #         self.save = True if self._force_save else save
    #     except AttributeError:
    #         self.save = save
    #
    #     self.id_grid = tb.make_regular_grid(self.image.shape[2:],
    #                                         dx_convention=self.dx_convention,
    #                                         device=device)
    #     assert self.id_grid != None
    #
    #     # field initialization to a regular grid
    #     field = self.id_grid.clone().to(device)
    #
    #     if plot > 0:
    #         self.save = True
    #
    #     if self.save:
    #         self.image_stock = torch.zeros((t_max * self.n_step,) + image.shape[1:])
    #         self.field_stock = torch.zeros(
    #             (t_max * self.n_step,) + field.shape[1:]
    #         )
    #         self.momentum_stock = [
    #             {k: torch.zeros_like(v) for k, v in momenta.items()}
    #             for _ in range(t_max * self.n_step)
    #         ]
    #         self.residuals_stock = torch.zeros((t_max * self.n_step,) + image.shape[1:])
    #
    #     if self.flag_hamiltonian_integration:
    #         self.norm_v = 0
    #         self.norm_z = 0
    #
    #     for i, t in enumerate(torch.linspace(0, t_max, t_max * self.n_step)):
    #         self._i = i
    #
    #         # print(self.step.__name__)
    #         if self.save_gpu_memory:
    #             use_reentrant =  False # set to true speed thing up but don't work with rigid.
    #             momenta, self.image , self.field, self.residuals = torch.utils.checkpoint.checkpoint(
    #                 self.step,
    #                 self.image,
    #                 momenta,
    #                 use_reentrant = use_reentrant,
    #             )
    #             self.momenta = self._unflatten_momenta(momenta)
    #         else:
    #             self.momenta, self.image, field, residuals = self.step(self.image, self.momenta)
    #
    #         self._test_nan_(self.image, self.momenta)
    #
    #         if self.flag_hamiltonian_integration:
    #             self.norm_v += self.norm_v_i / self.n_step
    #             self.norm_z += self.norm_z_i / self.n_step
    #             # self.ham_integration += self.ham_value / self.n_step
    #         # ic(self._i,self.field.min().item(),self.field.max().item(),
    #         #    self.momentum.min().item(),self.momentum.max().item(),
    #         #     self.image.min().item(),self.image.max().item())
    #
    #
    #         if self.save:
    #             if self._detach_image:
    #                 self.image_stock[i] = self.image[0].detach().to("cpu")
    #             else:
    #                 self.image_stock[i] = self.image[0]
    #             self.field_stock[i] = field[0].detach().to("cpu")
    #             for k, v in self.momenta.items():
    #                 self.momentum_stock[i][k] = v.detach().to("cpu")
    #             self.residuals_stock[i] = residuals[0].detach().to("cpu")
    #
    #         if verbose:
    #             update_progress(i / (t_max * self.n_step))
    #             if self.flag_hamilt_integration:
    #                 print('ham :', self.ham_value.detach().cpu().item(),
    #                   self.norm_v.detach().cpu().item(),
    #                   self.norm_z.detach().cpu().item())
    #
    #     # try:
    #     #     _d_ = device if self._force_save else 'cpu'
    #     #     self.field_stock = self.field_stock.to(device)
    #     # except AttributeError: pass
    #
    #     if plot > 0:
    #         self.plot(n_figs=plot)

    def forward(self,
                image,
                momenta: Momenta,
                save=True,
                plot=0,
                t_max=1,
                verbose=False,
                sharp=None,
                hamiltonian_integration=False,
                progress_callback=None):
        r""" This method is doing the temporal loop using the good method `_step_`

        Parameters
        ----------
        image : tensor array of shape [1,1,H,W]
            Source image ($I_0$)
        momentum_ini : tensor array of shape [1,1,H,W]
            Momentum ($p_0$) or residual ($z_0$)
        save : bool, optional
            Option to save the integration intermediary steps, by default True
            it saves the image, field and momentum at each step in the attributes
            `image_stock`, `field_stock`, `residuals_stock` and `momentum_stock`.
        plot : int, optional
            Positive int lower than `self.n_step` to plot the indicated number of
            intermediary steps, by default 0
        t_max : int, optional
            The integration will be made on [0,t_max], by default 1
        verbose : bool, optional
            Option to print the progress of the integration, by default False
        sharp : bool, optional
            Option to use the sharp integration, by default None
        debug : bool, optional
            Option to print debug information, by default False
        hamiltonian_integration : bool, optional
            Choose to integrate over first time step only or whole hamiltonian, in
            practice when True, the Regulation norms of the Hamiltonian are computed
            and saved in the good attributes (usually `norm_v` and `norm_z`),
             by default False
        progress_callback : callable, optional
            Called after each step with ``(completed_steps, total_steps)``.

        """
        if not isinstance(momenta, Momenta):
            raise TypeError(f"'momenta' must be a Momenta instance, got {type(momenta)}")
        if progress_callback is not None and not callable(progress_callback):
            raise TypeError("progress_callback must be callable")
        device = _get_device_from_momenta(momenta)
        self._forward_initialize_integration(
            image,
            _momenta_to_device(momenta, device),
            device,
            save,
            sharp,
            hamiltonian_integration,
            plot,
        )
        total_steps = t_max * self.n_step
        for i, t in enumerate(torch.linspace(0, t_max, total_steps)):
            self._i = i
            self._forward_single_step(verbose)
            if progress_callback is not None:
                progress_callback(i + 1, total_steps)

        if plot > 0:
            self.plot(n_figs=plot)


    # ------------------ PRIVATE HELPERS ------------------





    def _forward_initialize_integration(self, image, momenta: Momenta, device, save, sharp, hamiltonian_integration, plot):
        self._init_sharp_(sharp)
        self.source = image.detach().to(device)
        self.image = image.clone().to(device)
        self.momenta = momenta
        self.flag_hamiltonian_integration = hamiltonian_integration
        self.save = True if self._force_save else save


        self.id_grid = tb.make_regular_grid(self.image.shape[2:], dx_convention=self.dx_convention, device=device)
        assert self.id_grid is not None
        self.field = self.id_grid.clone().to(device)

        if plot > 0:
            self.save = True

        if self.save:
            T = self.n_step
            shape_image = self.image.shape[1:]
            shape_field = self.field.shape[1:]
            self.image_stock = torch.zeros(
                (T,) + shape_image, dtype=self.image.dtype
            )
            self.field_stock = torch.zeros(
                (T,) + shape_field, dtype=self.field.dtype
            )
            self.momentum_stock = [_zero_like_momenta(momenta, device="cpu") for _ in range(T)]
            self.residuals_stock = torch.zeros(
                (T,) + shape_image, dtype=self.image.dtype
            )

        if self.flag_hamiltonian_integration:
            self.norm_v = 0
            self.norm_z = 0


    def _forward_single_step(self, verbose):
        if self.save_gpu_memory:
            self._forward_checkpointed_step()
        else:
            self._forward_direct_step()

        self._test_nan_(self.image, self.momenta)

        if self.flag_hamiltonian_integration:
            self.norm_v += self.norm_v_i / self.n_step
            self.norm_z += self.norm_z_i / self.n_step

        if self.save:
            self._save_step()

        if verbose:
            self._log_step()


    def _forward_checkpointed_step(self):
        # print("_forward_checkpointed_step, abstract")

        use_reentrant = True # have to be false for rigid
        momenta, self.image, self.field, self.residuals = torch.utils.checkpoint.checkpoint(
            self.step,
            self.image,
            self.momenta,
            use_reentrant=use_reentrant,
        )
        return 0

    def _forward_direct_step(self):
        self.momenta, self.image, self.field, self.residuals = self.step(self.image, self.momenta)
        return 0

    def _save_step(self):
        i = self._i
        self.image_stock[i] = self.image[0].detach().cpu() if self._detach_image else self.image[0]
        self.field_stock[i] = self.field[0].detach().cpu()
        if isinstance(self.momenta, Momenta):
            detached = {k: v.detach().cpu() for k, v in self.momenta.as_dict().items()}
            self.momentum_stock[i] = Momenta(**detached)
        self.residuals_stock[i] = self.residuals[0].detach().cpu()


    def _log_step(self):
        update_progress(self._i / self.n_step)
        if getattr(self, "flag_hamilt_integration", False):
            print('ham :', self.ham_value.detach().cpu().item(),
                  self.norm_v.detach().cpu().item(),
                  self.norm_z.detach().cpu().item())


    def _image_Eulerian_integrator_(self, image, vector_field, t_max, n_step):
        """ image integrator using an Eulerian scheme

        :param image: (tensor array) of shape [T,1,H,W]
        :param vector_field: (tensor array) of shape [T,H,W,2]
        :param t_max: (float) the integration will be made on [0,t_max]
        :param n_step: (int) number of time steps in between [0,t_max]

        :return: (tensor array) of shape [T,1,H,W] integrated with vector_field
        """

        dt = t_max / n_step
        for t in torch.linspace(0, t_max, n_step):
            grad_I = tb.spatialGradient(image, dx_convention=self.dx_convention)
            grad_I_scalar_v = (grad_I[0] * tb.grid2im(vector_field)).sum(dim=1)
            image = image - grad_I_scalar_v * dt
        return image

    def _compute_vectorField_(self, momentum, grad_image):
        r""" operate the equation $K \star (z_t \cdot \nabla I_t)$

        :param momentum: (tensor array) of shape [H,W] or [D,H,W]
        :param grad_image: (tensor array) of shape [B,C,2,H,W] or [B,C,3,D,H,W]
        :return: (tensor array) of shape [B,H,W,2]
        """

        # if isinstance(momentum, Momenta):
        #     momentum = _primary_tensor(momentum)

        # C = residuals.shape[1]
        field_momentum = (grad_image * momentum.momentum_I.unsqueeze(2)).sum(dim=1)
        field =  self.kernelOperator(field_momentum)
        norm_v = None
        if self.flag_hamiltonian_integration:
            norm_v = .5 * self.rho * (field_momentum.clone() * field.clone()).sum()

        return -tb.im2grid(field), norm_v

    def _compute_vectorField_multimodal_(self, momentum, grad_image):
        r""" operate the equation $K \star (z_t \cdot \nabla I_t)$

        :param momentum: (tensor array) of shape [B,C,H,W] or [B,C,D,H,W]
        :param grad_image: (tensor array) of shape [B,C,2,H,W] or [B,C,3,D,H,W]
        :return: (tensor array) of shape [B,H,W,2]
        """

        if isinstance(momentum, Momenta):
            momentum = _primary_tensor(momentum)
        wheigths = self.channel_weight.to(momentum.device)
        W = wheigths.sum()
        # ic(residuals.shape,self.channel_weight.shape)
        return tb.im2grid(
            self.kernelOperator(
                (
                    -((wheigths * momentum).unsqueeze(2) * grad_image).sum(dim=1)
                    # / W
                )
            )
        )  # PAS OUF SI BATCH

    def _update_field_multimodal_(self):
        grad_image = tb.spatialGradient(self.image, dx_convention=self.dx_convention)
        self.field = self._compute_vectorField_multimodal_(self.momenta, grad_image)
        self.field *= self._field_cst_mult()

    # Done
    def _field_cst_mult(self):
        warnings.warn(
            "The method _field_cst_mult should not be used anymore,"
            "You might have to check the integrator steps equations."
        )
        rho = self._get_rho_()
        if rho == 1:
            return 1
        return rho / (1 - rho)

    # Done
    def _update_field_(self, momentum, image):
        grad_image = tb.spatialGradient(
            image,
            dx_convention=self.dx_convention,
            boundary=getattr(self, "gradient_boundary", "replicate"),
        )
        # ic(grad_image.min().item(), grad_image.max().item(),self.dx_convention)
        field, self.norm_v_i = self._compute_vectorField_(momentum, grad_image)
        # self.field *= self._field_cst_mult()
        # self.field *= sqrt(self.rho)

        return field

    # Done
    def _update_momentum_Eulerian_(self, momentum):
        momentum_dt = - tb.Field_divergence(dx_convention=self.dx_convention)(
            momentum[0, 0][None, :, :, None] * self.field,
        )

        return momentum + sqrt(self.rho) * momentum_dt / self.n_step

    # Done
    def _update_momentum_semiLagrangian_(self, deformation):
        warnings.warn("ANTON ! You should not use this function but "
                      "_compute_div_momentum_semiLagrangian_() instead !"
        )
        div_v_times_z = (
                self.momenta
                * tb.Field_divergence(dx_convention=self.dx_convention)(self.field)[0, 0]
        )
        self.momenta = (
                tb.imgDeform(
                    self.momenta, # TODO: check si c'est bien ça ...
                    deformation,
                    dx_convention=self.dx_convention,
                    clamp=False
                )
                - div_v_times_z / self.n_step
        )

    def _compute_div_momentum_semiLagrangian_(self,
                                              deformation,
                                              momentum,
                                              cst,
                                              field
                                              ):
        r"""
        Semi-Lagrangian momentum update for a divergence term.

        This method applies an explicit discretization of
        :math:`c\,\nabla \cdot (p\,v)` where :math:`p` is a scalar momentum and
        :math:`v` is a vector field:

        .. math::
            c\,\nabla\cdot(pv) = c\,v\cdot\nabla p + c\,p\,\nabla\cdot v

        In practice, the transport term is handled by warping ``momentum`` with
        ``deformation`` and the divergence term is evaluated on ``field``:

        .. math::
            p_{k+1} = \mathrm{warp}(p_k, \phi_k) -
            \frac{c}{n_{\text{step}}}\,p_k\,\nabla\cdot v_k

        Parameters
        ----------
        deformation : torch.Tensor
            Sampling grid used to advect the momentum (typically
            ``id_grid - cst * field / n_step``). Shape
            ``[B,H,W,2]`` in 2D or ``[B,D,H,W,3]`` in 3D.
        momentum : torch.Tensor
            Scalar momentum map to update. Shape ``[B,1,H,W]`` in 2D or
            ``[B,1,D,H,W]`` in 3D.
        cst : float or torch.Tensor
            Multiplicative coefficient ``c`` in front of the divergence term.
            Can be a scalar or a tensor broadcastable to ``momentum``.
        field : torch.Tensor
            Vector field ``v`` used to compute ``div(v)``. Shape
            ``[B,H,W,2]`` in 2D or ``[B,D,H,W,3]`` in 3D.

        Returns
        -------
        torch.Tensor
            Updated momentum with the same shape as ``momentum``.
        """

        div_v_times_p = cst * (
            momentum.momentum_I
            * tb.Field_divergence(
                dx_convention=self.dx_convention,
                boundary=getattr(self, "divergence_boundary", "reflect"),
            )(field)[0, 0]
        )
        momentum_I = (
            tb.imgDeform(
                momentum.momentum_I,
                deformation,
                dx_convention=self.dx_convention,
                clamp=False,
                boundary=getattr(self, "deformation_boundary", "zeros"),
            )
            - div_v_times_p / self.n_step
        )
        return momentum_I

    def _compute_sharp_intermediary_residuals_(self):
        base = _primary_tensor(self.momenta)
        device = base.device
        resi_cumul = torch.zeros(base.shape, device=device)
        # for k,phi in enumerate(self._phis[self._i][:]):
        for k, phi in enumerate(self._phis[self._i][1:]):
            resi_cumul += tb.imgDeform(_primary_tensor(self.momentum_stock[k])[None].to(device),
                                       phi,
                                       dx_convention=self.dx_convention,
                                       clamp=False)
        resi_cumul = resi_cumul + base
        return resi_cumul
        # Non sharp but working residual
        # if self._i >0:
        #     for k,z in enumerate(self._resi_deform):
        #         self._resi_deform[k] = tb.imgDeform(z[None,None].to(device),
        #                                             self._phis[self._i][self._i],
        #                                             self.dx_convention)[0,0]
        #     self._phis[self._i - 1] = None
        # self._resi_deform.append(self.residuals.clone())

    # Done
    def _update_image_Eulerian_(self):
        # Warning, in classical metamorphosis, the momentum (p) is proportional to the residual (z)
        # with the relation z = (1 - rho) * p. Here we use the momentum as the residual
        self.image = self._image_Eulerian_integrator_(
            self.image, self.field, 1 / self.n_step, 1
        )
        # z = sqrt(1 - rho) * p and I = v gradI + sqrt(1-rho) * z
        residuals = (1 - self.rho) * self.momentum
        self.image = (sqrt(self.rho) * self.image + residuals) / self.n_step

    # Done
    def _update_image_semiLagrangian_(self, momentum, image, deformation, residuals=None, sharp=False):
        if residuals is None:
            # z = sqrt(1 - rho) * p and I = v gradI + sqrt(1-rho) * z
            residuals = (1 - self.rho) * momentum.momentum_I
        self.norm_z_i = None
        if self.flag_hamiltonian_integration:
            self.norm_z_i = .5 * residuals.pow(2).sum()
        # if self.rho > 0:
        image_def = tb.imgDeform(
            image,
            deformation,
            dx_convention=self.dx_convention,
            boundary=getattr(self, "deformation_boundary", "zeros"),
        )

        if self._get_rho_() < 1:
            image_def += residuals / self.n_step
        return image_def

    def _update_sharp_intermediary_field_(self):
        # print('update phi ',self._i,self._phis[self._i])
        self._phis[self._i][self._i] = self.id_grid - self.field / self.n_step
        if self._i > 0:
            for k, phi in enumerate(self._phis[self._i - 1]):
                self._phis[self._i][k] = phi + tb.compose_fields(
                    -self.field / self.n_step, phi, self.dx_convention
                ).to(self.field.device)
                # self._phis[self._i][k] = tb.compose_fields(
                #     phi,
                #     self._phis[self._i][self._i],
                #     # self.field/self.n_step,
                #     'pixel'
                # ).to(self.field.device)

    def _update_momentum_weighted_semiLagrangian_(self, deformation):
        sqm = torch.sqrt(self.residual_norm[self._i])
        fz_times_div_v = (
            sqm
            * self.momentum
            * tb.Field_divergence(dx_convention=self.dx_convention)(self.field)[0, 0]
        )
        div_fzv = (
            -tb.imgDeform(
                sqm * self.momentum,
                deformation,
                dx_convention=self.dx_convention,
                clamp=False,
            )[0, 0]
            + fz_times_div_v / self.n_step
        )
        z_time_dtF = self.momentum * self.rf.dt_F(self._i)
        self.momentum = -(div_fzv + z_time_dtF)

    def _update_image_weighted_semiLagrangian_(
        self, momentum, image, deformation, residuals=None, sharp=False
    ):
        if residuals is None:
            residuals = momentum
        image = self.source if sharp else image
        image = tb.imgDeform(image, deformation, dx_convention=self.dx_convention)
        image += residuals / self.n_step

        return image


    def _update_field_oriented_weighted_(self, momentum, image):
        grad_image = tb.spatialGradient(image, dx_convention=self.dx_convention)
        free_field = tb.im2grid(
            (momentum * grad_image[0]) * torch.sqrt(self.residual_mask[self._i])
        )
        oriented_field = 0
        if self.flag_O:

            oriented_field = (self.orienting_field[self._i][None]
                                    * self.orienting_mask[self._i][..., None])

        field =  -tb.im2grid(
            self.kernelOperator(tb.grid2im(free_field + oriented_field))
        )
        return field

    def to_device(self, device):
        # TODO: completer ça
        self.device = device
        try:
            self.image = self.image.to(device)
            self.id_grid = self.id_grid.to(device)
            self.field = self.field.to(device)
            self.residuals = self.residuals.to(device)
            if isinstance(self.momenta, Momenta):
                self.momenta = _momenta_to_device(self.momenta, device)

            # self.

        except AttributeError:
            pass

    def get_deformation(self, from_t=0, to_t=None, save=False):
        r"""Returns the deformation use it for showing results
        $\Phi = \int_s^t v_t dt$ with $s < t$ and $t \in [0,n_{step}-1]$

        :params: from_t : (int) the starting time step (default 0)
        :params: to_t : (int) the ending time step (default n_step)
        :params: save : (bool) option to save the integration intermediary steps. If true, the return value will have its shape with T>1
        :return: deformation [T,H,W,2] or [T,H,W,D,3]
        """

        # if n_step == 0:
        #     return self.id_grid.detach().cpu() + self.field_stock[0][None].detach().cpu()/self.n_step
        # temporal_integrator = vff.FieldIntegrator(method='temporal',save=save)
        # if n_step is None:
        #     return temporal_integrator(self.field_stock/self.n_step,forward=True)
        # else:
        #     return temporal_integrator(self.field_stock[:n_step]/self.n_step,forward=True)
        #
        temporal_integrator = vff.FieldIntegrator(
            method="temporal",
            save=save,
            dx_convention=self.dx_convention,
            boundary=getattr(self, "field_integration_boundary", "border"),
        )
        if from_t is None and to_t is None:
            print("Je suis passé par là")
            return temporal_integrator(self.field_stock / self.n_step, forward=False)
        # if from_t is None: from_t = 0
        if to_t is None:
            to_t = self.n_step
        if from_t < 0 and from_t >= to_t:
            raise ValueError(
                f"from_t must be in [0,n_step-1], got from_t ={from_t} and n_step = {self.n_step}"
            )
        if to_t > self.n_step or to_t <= from_t:
            raise ValueError(
                f"to_t must be in [from_t+1,n_step], got to_t ={to_t} and n_step = {self.n_step}"
            )
        if to_t == 1:
            return (
                self.id_grid.detach().cpu()
                + self.field_stock[0][None].detach().cpu() / self.n_step
            )
        # ic(from_t,to_t,self.field_stock[from_t:to_t].shape)
        return temporal_integrator(
            self.field_stock[from_t:to_t] / self.n_step, forward=True
        )

    def get_deformator(self, from_t=0, to_t=None, save=False):
        r"""Returns the inverse deformation use it for deforming images
        $(\Phi_{s,t})^{-1}$ with $s < t$ and $t \in [0,n_{step}-1]$

        :params: from_t : (int) the starting time step (default 0)
        :params: to_t : (int) the ending time step (default n_step)
        :params: save : (bool) option to save the integration intermediary steps. If true, the return value will have its shape with T>1
        :return: deformation [T,H,W,2] or [T,H,W,D,3]
        """

        temporal_integrator = vff.FieldIntegrator(
            method="temporal",
            save=save,
            dx_convention=self.dx_convention,
            boundary=getattr(self, "field_integration_boundary", "border"),
        )
        if from_t is None and to_t is None:
            return temporal_integrator(self.field_stock / self.n_step, forward=False)
        # if from_t is None: from_t = 0
        if to_t is None:
            to_t = self.n_step
        if from_t < 0 and from_t >= to_t:
            raise ValueError(
                f"from_t must be in [0,n_step-1], got from_t ={from_t} and n_step = {self.n_step}"
            )
        if to_t > self.n_step or to_t <= from_t:
            raise ValueError(
                f"to_t must be in [from_t+1,n_step], got to_t ={to_t} and n_step = {self.n_step}"
            )
        if to_t == 1:
            return (
                self.id_grid.detach().cpu()
                - self.field_stock[0][None].detach().cpu() / self.n_step
            )
        # ic(from_t,to_t,self.field_stock[from_t:to_t].shape)
        return temporal_integrator(
            self.field_stock[from_t:to_t] / self.n_step, forward=False
        )

    # ==================================================================
    #                       PLOTS
    # ==================================================================

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
        base_momentum = _primary_tensor(self.momenta)
        v_abs_max = torch.quantile(base_momentum.abs(), 0.99)
        kw_residuals_args = dict(
            cmap="RdYlBu_r",
            extent=[-1, 1, -1, 1],
            origin="lower",
            vmin=-v_abs_max,
            vmax=v_abs_max,
        )
        size_fig = 5
        C = 1 # C = self.momentum_stock.shape[1]
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

            for j in range(C):
                r_s = ax[i, j + 1].imshow(
                    self.momentum_stock[t].detach().momentum_I[0,0].numpy(),
                    **kw_residuals_args
                )
                ax[i, j + 1].axis("off")

            fig.colorbar(r_s, ax=ax[i, -2], fraction=0.046, pad=0.04)

            tb.gridDef_plot_2d(
                self.get_deformation(to_t = t+1),
                add_grid=False,
                ax=ax[i, -1],
                step=int(min(self.field_stock.shape[2:-1]) / 30),
                check_diffeo=True,
                dx_convention=self.dx_convention,
            )

        return fig, ax

    def plot_deform(self, target, temporal_nfig=0):

        if self.save == False:
            raise TypeError(
                "metamophosis_path.forward attribute 'save' has to be True to use self.plot_deform"
            )

        temporal = temporal_nfig > 0
        # temporal integration over v_t
        temporal_integrator = vff.FieldIntegrator(
            method="temporal",
            save=temporal,
            dx_convention=self.dx_convention,
            boundary=getattr(self, "field_integration_boundary", "border"),
        )

        # field_stock_toplot = tb.pixel2square_convention(self.field_stock)
        # tb.gridDef_plot(field_stock_toplot[-1][None],dx_convention='2square')
        if temporal:
            full_deformation_t = temporal_integrator(
                self.field_stock / self.n_step, forward=True
            )
            full_deformator_t = temporal_integrator(
                self.field_stock / self.n_step, forward=False
            )
            full_deformation = full_deformation_t[-1].unsqueeze(0)
            full_deformator = full_deformator_t[-1].unsqueeze(0)
        else:
            full_deformation = temporal_integrator(
                self.field_stock / self.n_step, forward=True
            )
            full_deformator = temporal_integrator(
                self.field_stock / self.n_step, forward=False
            )

        fig, axes = plt.subplots(3, 2, constrained_layout=True, figsize=(20, 30))
        # show resulting deformation

        tb.gridDef_plot_2d(
            full_deformation,
            step=int(max(self.image.shape) / 30),
            ax=axes[0, 0],
            check_diffeo=True,
            dx_convention=self.dx_convention,
        )
        tb.quiver_plot(
            full_deformation - self.id_grid.cpu(),
            step=int(max(self.image.shape) / 30),
            ax=axes[0, 1],
            check_diffeo=False,
            dx_convention=self.dx_convention,
        )

        # show S deformed by full_deformation
        S_deformed = tb.imgDeform(
            self.source.cpu(),
            full_deformator,
            dx_convention=self.dx_convention,
            boundary=getattr(self, "deformation_boundary", "zeros"),
        )
        # axes[1,0].imshow(self.source[0,0,:,:].cpu().permute(1,2,0),cmap='gray',origin='lower',vmin=0,vmax=1)
        # axes[1,1].imshow(target[0].cpu().permute(1,2,0),cmap='gray',origin='lower',vmin=0,vmax=1)
        # axes[2,0].imshow(S_deformed[0,0,:,:].permute(1,2,0),cmap='gray',origin='lower',vmin=0,vmax=1)
        # axes[2,1].imshow(tb.imCmp(target,S_deformed),origin='lower',vmin=0,vmax=1)

        axes[1, 0].imshow(
            self.source[0, 0, :, :].cpu(), cmap="gray", origin="lower", vmin=0, vmax=1
        )
        axes[1, 1].imshow(
            target[0, 0].cpu(), cmap="gray", origin="lower", vmin=0, vmax=1
        )
        axes[2, 0].imshow(
            S_deformed[0, 0, :, :], cmap="gray", origin="lower", vmin=0, vmax=1
        )
        axes[2, 1].imshow(
            tb.imCmp(target[:, 0][None], S_deformed[:, 0][None], method="compose")[0],
            origin="lower",
            vmin=0,
            vmax=1,
        )

        set_ticks_off(axes)
        if temporal:
            t_max = full_deformator_t.shape[0]
            plot_id = (
                torch.quantile(
                    torch.arange(t_max, dtype=torch.float),
                    torch.linspace(0, 1, temporal_nfig),
                )
                .round()
                .int()
            )
            size_fig = 5
            plt.rcParams["figure.figsize"] = [size_fig, temporal_nfig * size_fig]
            fig, ax = plt.subplots(temporal_nfig)

            for i, t in enumerate(plot_id):
                tb.quiver_plot(
                    full_deformation_t[i].unsqueeze(0) - self.id_grid, step=10, ax=ax[i]
                )
                tb.gridDef_plot(
                    full_deformation_t[i].unsqueeze(0),
                    add_grid=False,
                    step=10,
                    ax=ax[i],
                    color="green",
                )

                tb.quiver_plot(
                    self.field_stock[i].unsqueeze(0), step=10, ax=ax[i], color="red"
                )


class Optimize_geodesicShooting(torch.nn.Module, ABC):
    """Abstract method for geodesic shooting optimisation. It needs to be provided with an object
    inheriting from Geodesic_integrator
    """

    @abstractmethod
    def __init__(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        geodesic: Geodesic_integrator,
        cost_cst,
        data_term=None,
        optimizer_method: str = 'LBFGS_torch',
        lbfgs_max_iter: int = 20,
        lbfgs_history_size: int = 100,
        hamiltonian_integration=False,
        debug=False,
        **kwargs
    ):
        """

        Important note to potential forks : all children of this method
        must have the same __init__ method for proper loading.
        :param source:
        :param target:
        :param geodesic:
        :param cost_cst:
        :param optimizer_method:
        """

        super().__init__()
        self.mp = geodesic
        self.dx_convention = self.mp.dx_convention
        self.source = source
        self.target = target
        self.lbfgs_max_iter = lbfgs_max_iter
        self.lbfgs_history_size = lbfgs_history_size
        self.debug = debug
        # Adam extras (harmless for other optimizers)
        self._adam_scheduler_type = kwargs.pop('adam_scheduler', None)
        self._adam_grad_clip = kwargs.pop('adam_grad_clip', None)
        if self.debug:
            self.mp.debug = self.debug

        self.flag_hamiltonian_integration = hamiltonian_integration
        self.mp.kernelOperator.init_kernel(source)
        try:
            self.dx = self.mp.kernelOperator.dx
        except AttributeError:
            if self.dx_convention == "pixel":
                self.dx = (1,) * len(source.shape[2:])
            elif self.dx_convention == "square":
                self.dx = tuple([1 / (h - 1) for h in source.shape[2:]])
            elif self.dx_convention == "2square":
                self.dx = tuple([2 / (h - 1) for h in source.shape[2:]])
            else:
                raise ValueError("dx_convention must be in ['pixel','square']")

        self.cost_cst = cost_cst
        # optimize on the cost as defined in the 2021 paper.
        self._cost_saving_ = self._default_cost_saving_

        self.optimizer_method_name = optimizer_method  # for __repr__
        # forward function choice among developed optimizers
        if optimizer_method == "grad_descent":
            self._initialize_optimizer_ = self._initialize_grad_descent_
            self._step_optimizer_ = self._step_grad_descent_
        elif optimizer_method == "LBFGS_torch":
            self._initialize_optimizer_ = self._initialize_LBFGS_
            self._step_optimizer_ = self._step_LBFGS_
        elif optimizer_method == "adadelta":
            self._initialize_optimizer_ = self._initialize_adadelta_
            self._step_optimizer_ = self._step_adadelta_
        elif optimizer_method == "Adam":
            self._initialize_optimizer_ = self._initialize_Adam_with_scheduler_
            self._step_optimizer_ = self._step_Adam_
        else:
            raise ValueError(
                "\noptimizer_method is "
                + optimizer_method
                + "You have to specify the optimizer_method used among"
                "{'grad_descent', 'LBFGS_torch','adadelta'}"
            )
        self._iter = 0  # optimisation iteration counter
        self.data_term = dt.Ssd(self.target) if data_term is None else data_term
        if isinstance(self.data_term, type):
            raise ValueError(
                f"You provided {self.data_term} as data_term."
                f"It seems that you did not initialize it."
            )
        self.data_term.set_optimizer(self)

        # self.temporal_integrator = vff.FieldIntegrator(method='temporal',save=False)
        self.is_DICE_cmp = False  # Is dice already computed ?
        self._plot_forward_ = self._plot_forward_dlt_

        # # Default parameters to save (write to file)
        # self.field_to_save = FIELD_TO_SAVE
        self.optimized_momenta = None
        self.loss_stock = None
        self.integration_diverged = False

    @property
    def to_analyse(self):
        warnings.warn(
            "to_analyse is deprecated. Use optimized_momenta and loss_stock instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self.optimized_momenta is None and self.loss_stock is None:
            return None
        return self.optimized_momenta, self.loss_stock

    @to_analyse.setter
    def to_analyse(self, value):
        warnings.warn(
            "to_analyse is deprecated. Use optimized_momenta and loss_stock instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if isinstance(value, (tuple, list)) and len(value) == 2:
            self.optimized_momenta, self.loss_stock = value
        else:
            self.optimized_momenta, self.loss_stock = None, value

    # @abstractmethod
    # def _compute_V_norm_(self,*args):
    #     pass
    def _compute_V_norm_(self, momentum, image):
        """

         _compute_V_norm_(momentum, image)
            :momentum: torch Tensor of shape [1,C,H,W] or [1,C,D,H,W]
            :image: torch Tensor of shape [1,C,H,W] or [1,C,D,H,W]
        :return: float
        """

        # Computes only
        grad_source = tb.spatialGradient(
            image,
            dx_convention=self.dx_convention,
            boundary=getattr(self.mp, "gradient_boundary", "replicate"),
        )
        field_momentum = (grad_source * momentum.momentum_I.unsqueeze(2)).sum(dim=1)  # / C
        field = self.mp.kernelOperator(field_momentum)

        norm_v = (field_momentum * field).sum()
        if norm_v < 0:
            warnings.warn(f"norm_v is negative : {norm_v}, increasing"
                          f" kernel_reach in kernelOperator might help")
        return norm_v

    def _timed_observation_images(self, target_steps):
        images = []
        for step in target_steps:
            if step == 0:
                images.append(self.source)
            else:
                images.append(self.mp.image_stock[step - 1][None])
        return torch.cat(images, dim=0)

    @abstractmethod
    def cost(self, **kwargs):
        pass

    # @abstractmethod
    # def _get_rho_(self):
    #     pass

    @abstractmethod
    def get_all_arguments(self):
        return {
            "n_step": self.mp.n_step,
            "cost_cst": self.cost_cst,
            "kernelOperator": self.mp.kernelOperator.get_all_arguments(),
            "hamiltonian_integration": self.flag_hamiltonian_integration,
            "dx_convention": self.dx_convention,
        }

    def get_geodesic_distance(self, only_zero=False):
        if only_zero:
            if self.optimized_momenta is None:
                raise ValueError("No optimized momenta available to compute distance.")
            return float(self._compute_V_norm_(_primary_tensor(self.optimized_momenta), self.source))
        else:
            dist = float(
                self._compute_V_norm_(_primary_tensor(self.mp.momentum_stock[0])[None],
                                      self.mp.source)
            )
            for t in range(len(self.mp.momentum_stock) - 1):
                dist += float(
                    self._compute_V_norm_(
                        _primary_tensor(self.mp.momentum_stock[t + 1])[None],
                        self.mp.image_stock[t][None],
                    )
                )
            return dist

    def __repr__(self) -> str:
        return self.__class__.__name__ + \
            '(cost_parameters : {' + \
            ', \n\t\trho =' + str(self._get_rho_()) + \
            ', \n\t\tlambda =' + str(self.cost_cst) + '\n\t},' + \
            f'\n\tgeodesic integrator : ' + self.mp.__repr__() + \
            f'\n\tintegration method : ' + self.mp.step.__name__ + \
            f'\n\toptimisation method : ' + self.optimizer_method_name + \
            f'\n\t# geodesic steps =' + str(self.mp.n_step) + '\n)'

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #   Implemented OPTIMIZERS
    def _prepare_optimization_parameter_(self, parameter):
        """Map a physical shooting parameter to optimizer coordinates."""
        return parameter

    def _parameter_for_cost_(self, parameter):
        """Map optimizer coordinates back to physical shooting variables."""
        return parameter

    def _optimization_cost_(self, parameter):
        return self.cost(self._parameter_for_cost_(parameter))

    def _finalize_integrator_(self, final_parameter):
        self.mp.forward(
            self.source.clone(),
            final_parameter,
            save=True,
            plot=0,
        )

    # GRADIENT DESCENT
    def _initialize_grad_descent_(self, dt_step, max_iter=20):
        self.optimizer = GradientDescent(
            self._optimization_cost_, self._optimization_parameter, lr=dt_step
        )

    def _step_grad_descent_(self):
        self.optimizer.step(verbose=False)

    def _dict_or_torch_parameter_(self):
        parameter = self._optimization_parameter
        if isinstance(parameter, TorchDataClass):
            parameters = list(parameter.as_dict().values())
        elif isinstance(parameter, torch.Tensor):
            parameters = [parameter]
        else:
            raise TypeError(f"unsupported optimizer parameter {type(parameter)}")
        if len({id(parameter) for parameter in parameters}) != len(parameters):
            raise ValueError("optimizer parameter tensors must be distinct")
        if any(not parameter.is_leaf or not parameter.requires_grad for parameter in parameters):
            raise ValueError("optimizer parameters must be grad-enabled leaf tensors")
        return parameters

    # LBFGS
    def _initialize_LBFGS_(self, dt_step):

        self.optimizer = torch.optim.LBFGS(
                self._dict_or_torch_parameter_(),
                max_iter=self.lbfgs_max_iter,
                history_size=self.lbfgs_history_size,
               lr=dt_step,
               line_search_fn='strong_wolfe'
            )

        def closure():
            self.optimizer.zero_grad()
            L = self._optimization_cost_(self._optimization_parameter)
            # save best cms
            # if(self._it_count >1 and L < self._loss_stock[:self._it_count].min()):
            #     cms_tosave.data = self.cms_ini.detach().data
            # L.backward()
            L.backward(retain_graph=False)
            return L

        self.closure = closure

    def _step_LBFGS_(self):
        self.optimizer.step(self.closure)

    # Adam
    def _initialize_Adam_(self, dt_step):
        """
        Initialize Adam optimizer for this model.
        dt_step : float
            Learning rate for Adam.
        """
        self.optimizer = torch.optim.Adam(
            self._dict_or_torch_parameter_(),
            lr=dt_step,
            betas=(0.9, 0.999),   # default
            eps=1e-8,             # default
            weight_decay=0        # default
        )

    def _initialize_Adam_with_scheduler_(self, dt_step):
        """Dispatch wrapper: calls _initialize_Adam_ (possibly overridden in children)
        then attaches the optional LR scheduler. Using a wrapper avoids touching
        subclass overrides of _initialize_Adam_."""
        self._initialize_Adam_(dt_step)
        self._setup_adam_scheduler_(dt_step)

    def _setup_adam_scheduler_(self, base_lr):
        """Create and attach an LR scheduler to self.optimizer after Adam is set up.

        Controlled by ``self._adam_scheduler_type``:
        - ``None``                 — no scheduler (default).
        - ``'reduce_on_plateau'``  — halves LR after 10 stagnant outer steps.
          Best for oscillating losses: reacts to the actual loss curve.
        - ``'cosine'``             — cosine annealing over ``n_iter`` steps down to
          1 % of initial LR.  Smooth, predictable decay.
        - ``'exponential'``        — multiplies LR by 0.99 each outer step.
          Gentle continuous decay.
        """
        self._lr_scheduler = None
        if self._adam_scheduler_type is None:
            return
        t = self._adam_scheduler_type
        if t in ('reduce_on_plateau', 'rop'):
            self._lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=10,
                min_lr=base_lr * 1e-3,
            )
        elif t == 'cosine':
            self._lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.n_iter,
                eta_min=base_lr * 1e-2,
            )
        elif t == 'exponential':
            self._lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=0.99,
            )
        else:
            raise ValueError(
                f"Unknown adam_scheduler={t!r}. "
                "Choose from: 'reduce_on_plateau', 'cosine', 'exponential'."
            )

    def _step_Adam_(self):
        """
        Perform one optimization step with Adam, with optional gradient clipping
        and LR scheduling.
        """
        self.optimizer.zero_grad()
        L = self._optimization_cost_(self._optimization_parameter)
        L.backward(retain_graph=False)
        if self._adam_grad_clip is not None:
            all_params = [p for group in self.optimizer.param_groups
                          for p in group['params']]
            torch.nn.utils.clip_grad_norm_(all_params, self._adam_grad_clip)
        self.optimizer.step()
        if getattr(self, '_lr_scheduler', None) is not None:
            if isinstance(self._lr_scheduler,
                          torch.optim.lr_scheduler.ReduceLROnPlateau):
                self._lr_scheduler.step(L.detach())
            else:
                self._lr_scheduler.step()
        return L

    def _initialize_adadelta_(self, dt_step, max_iter=None):
        self.optimizer = torch.optim.Adadelta(self._dict_or_torch_parameter_(),
                                              lr=dt_step,
                                              rho=0.9,
                                              weight_decay=0)

        def closure():
            self.optimizer.zero_grad()
            L = self._optimization_cost_(self._optimization_parameter)
            # save best cms
            # if(self._it_count >1 and L < self._loss_stock[:self._it_count].min()):
            #     cms_tosave.data = self.cms_ini.detach().data
            L.backward(retain_graph=True)
            return L

        self.closure = closure

    def _step_adadelta_(self):
        self.optimizer.step(self.closure)

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def _default_cost_saving_(self, i, loss_stock):
        """

        :param i: index for saving the according values
                !!! if `loss_stock` is None, `loss_stock` will be initialized, and
                `i` must have the value of the number of iterations.
        :param loss_stock:
        :return: updated `loss_stock`
        """

        # initialise loss_stock
        if loss_stock is None:
            d = 3
            return torch.zeros((i, d))

        loss_stock[i, 0] = self.data_loss.detach()
        loss_stock[i, 1] = self.norm_v_2.detach()
        loss_stock[i, 2] = self.norm_l2_on_z.detach()
        return loss_stock

    def _plot_forward_dlt_(self):
        plt.figure()
        plt.imshow(self.mp.image[0, 0].detach().cpu(), **DLT_KW_IMAGE)
        plt.show()

    def _build_parameter_dict_(self, momentum_ini):
        if isinstance(momentum_ini, TorchDataClass):
            return momentum_ini
        raise ValueError(
            "In Optimize_geodesicShooting forward, momentum_ini must be a "
            "TorchDataClass "
            f"instance. Got : {type(momentum_ini)}"
        )




    @time_it
    def forward(self,
                momenta_ini,
                n_iter=10,
                grad_coef=1e-3,
                verbose=True,
                plot=False,
                sharp=None,
                convergence_tol=None,
                convergence_patience=3,
                ):
        r""" The function is and perform the optimisation with the desired method.
        The result is stored in two attributes. `self.optimized_momenta` holds the optimized
        initial residual ($z_O$ in the article) used for the shooting.
        `self.loss_stock` stores the values of the loss norms over time. The function
        plot_cost() is designed to show them automatically.

        :param momenta_ini: initial momentum. It is the variable on which we optimize.
        `require_grad` must be set to True.
        :param n_iter: (int) number of optimizer iterations
        :param verbose: (bool) display advancement
        :param convergence_tol: (float or None) If set, the optimisation stops early when the
        relative change in data loss between two consecutive outer iterations falls below this
        threshold for `convergence_patience` consecutive steps.
        Relative change is defined as ``|L_i - L_{i-1}| / (|L_{i-1}| + 1e-12)``.
        Set to ``None`` (default) to disable early stopping and always run ``n_iter`` iterations.
        :param convergence_patience: (int) Number of consecutive iterations whose relative loss
        change is below ``convergence_tol`` before the optimisation is declared converged and
        stopped. Default is 3.

        """
        def _detach(p):
            if isinstance(p, TorchDataClass):
                return _momenta_detach(p)
            raise TypeError(f"parameter must be a TorchDataClass, got {type(p)}")

        if not isinstance(momenta_ini, TorchDataClass):
            raise TypeError(
                "momenta_ini must be a TorchDataClass instance, "
                f"got {type(momenta_ini)}"
            )
        device = _get_device_from_momenta(momenta_ini)
        self.integration_diverged = False

        self.source = self.source.to(device)
        # self.target = self.target.to(z_0.device)
        # self.mp.kernelOperator.kernel = self.mp.kernelOperator.kernel.to(z_0.device)
        self.data_term.to_device(device)

        self.parameter = _momenta_to_device(momenta_ini, device)
        self._optimization_parameter = self._prepare_optimization_parameter_(
            self.parameter
        )
        self.n_iter = n_iter  # must precede _initialize_optimizer_ (scheduler may read it)
        self._initialize_optimizer_(grad_coef)
        # self.n_iter = n_iter

        self.id_grid = tb.make_regular_grid(self.source.shape[2:],
                                            dx_convention=self.dx_convention,
                                            device=device)
        # self.to_device(momenta_ini.device)
        if self.id_grid is None:
            raise ValueError(
                f"The initial momentum provided might have the wrong shape, got :{momenta_ini.shape}"
            )

        self._iter_ = 0
        self._optimization_cost_(_detach(self._optimization_parameter))

        loss_stock = self._cost_saving_(n_iter, None)  # initialisation
        loss_stock = self._cost_saving_(0, loss_stock)

        # convergence tracking
        _prev_loss = float(self.data_loss.detach())
        _patience_count = 0
        _last_iter = 0  # tracks the last completed iteration index

        for i in range(1, n_iter):
            self._iter_ = i
            self._step_optimizer_()
            self._optimization_cost_(_detach(self._optimization_parameter))
            loss_stock = self._cost_saving_(i, loss_stock)

            if verbose:
                loss_val = loss_stock["data_loss"][i] if isinstance(loss_stock, dict) else loss_stock[i, 0]
                update_progress(
                    (i + 1) / n_iter,
                    message=(
                        f"{self.data_term.__class__.__name__} :",
                        loss_val,
                    ),
                )
            if plot and i in [n_iter // 4, n_iter // 2, 3 * n_iter // 4]:
                self._plot_forward_()

            # --- early stopping on convergence ---
            if convergence_tol is not None:
                _curr_loss = float(self.data_loss.detach())
                rel_change = abs(_curr_loss - _prev_loss) / (abs(_prev_loss) + 1e-12)
                _prev_loss = _curr_loss
                if rel_change < convergence_tol:
                    _patience_count += 1
                else:
                    _patience_count = 0
                # print(f"abstract.py:1490 in forward()\n"
                #       f"\t_curr_loss:{_curr_loss}\n"
                #       f"\t_rel_change:{rel_change}\n"
                #       f"\tconvergence_tol:{convergence_tol}\n"
                #       f"\t_patience_count:{_patience_count}")
                # ic(i,_curr_loss, rel_change, convergence_tol,_patience_count, )
                if _patience_count >= convergence_patience:
                    _last_iter = i
                    if verbose:
                        print(
                            f"\nConverged at iteration {i}/{n_iter} "
                            f"(rel_change={rel_change:.2e} < tol={convergence_tol:.2e} "
                            f"for {convergence_patience} consecutive steps)"
                        )
                    break
            # -------------------------------------

            _last_iter = i

        # trim loss_stock to the iterations actually performed so that
        # plot_cost() does not show a spurious flat tail of zeros
        if convergence_tol is not None and _last_iter < n_iter - 1:
            actual = _last_iter + 1
            if isinstance(loss_stock, dict):
                loss_stock = {k: v[:actual] for k, v in loss_stock.items()}
            else:
                loss_stock = loss_stock[:actual]

        # Prepare the finalized integration data used by plots and diagnostics.
        final_parameter = _detach(
            self._parameter_for_cost_(self._optimization_parameter)
        )
        self._finalize_integrator_(final_parameter)

        self.parameter = final_parameter
        self.optimized_momenta = final_parameter
        self._optimization_parameter = None
        self.optimizer = None
        self.closure = None
        self._lr_scheduler = None
        self.loss_stock = loss_stock
        self.to_device('cpu')

    def to_device(self, device):
        # self.mp.kernelOperator.kernel = self.mp.kernelOperator.kernel.to(device)
        self.mp.to_device(device)
        self.source = self.source.to(device)
        self.target = self.target.to(device)
        if isinstance(self.parameter, TorchDataClass):
            self.parameter = _momenta_to_device(self.parameter, device)
        self.id_grid = self.id_grid.to(device)
        self.data_term.to_device(device)
        if isinstance(self.optimized_momenta, TorchDataClass):
            self.optimized_momenta = _momenta_to_device(self.optimized_momenta, device)
        def _loss_to_device(loss):
            if loss is None or isinstance(loss, str):
                return loss
            if torch.is_tensor(loss):
                return loss.to(device)
            if isinstance(loss, dict):
                return {
                    key: (val.to(device) if hasattr(val, "to") else val)
                    for key, val in loss.items()
                }
            return loss
        self.loss_stock = _loss_to_device(self.loss_stock)

    def forward_safe_mode(self,
                          z_0,
                          n_iter=10,
                          grad_coef=1e-3,
                          verbose=True,
                          mode=None,
                          convergence_tol=None,
                          convergence_patience=3,
                          ):
        """ Same as Optimize_geodesicShooting.forward(...) but
        does not stop the program when the integration diverges.
        If mode is not None, it tries to change the parameter
        until convergence as described in ```mode```

        :param z_0: initial residual. It is the variable on which we optimize.
        `require_grad` must be set to True.
        :param n_iter: (int) number of optimizer iterations
        :param verbose: (bool) display advancement
        :param mode:
            `'grad_coef'` this mode will decrease the grad_coef by
            dividing it by 10.
        :param convergence_tol: passed to ``forward`` — see its docstring.
        :param convergence_patience: passed to ``forward`` — see its docstring.
        :return:
        """
        try:
            self.forward(z_0, n_iter, grad_coef, verbose=verbose,
                         convergence_tol=convergence_tol,
                         convergence_patience=convergence_patience)
        except OverflowError:
            if mode is None:
                print("Integration diverged : Stop.\n\n")
                self.integration_diverged = True
                self.optimized_momenta = None
                self.loss_stock = None
            elif mode == "grad_coef":
                print(f"Integration diverged :" f" set grad_coef to {grad_coef*0.1}")
                self.forward_safe_mode(z_0, n_iter, grad_coef * 0.1, verbose, mode=mode,
                                       convergence_tol=convergence_tol,
                                       convergence_patience=convergence_patience)

    def compute_landmark_dist(
        self,
            source_landmark,
            target_landmark=None,
            forward=True,
            verbose=True,
            round = False
    ):
        # from scipy.interpolate import interpn
        # import numpy as np
        # compute deformed landmarks
        if forward:
            deformation = self.mp.get_deformation()
        else:
            deformation = self.mp.get_deformator()
        if self.dx_convention== "square":
            deformation = tb.square_to_pixel_convention(deformation,is_grid=True)
        elif self.dx_convention == "2square":
            deformation = tb.square2_to_pixel_convention(deformation,is_grid=True)
        deform_landmark = []
        for l in source_landmark:
            idx = (0,) + tuple([int(j) for j in l.flip(0)])
            deform_landmark.append(deformation[idx].tolist())



        self.source_landmark = source_landmark
        self.target_landmark = target_landmark
        self.deform_landmark = torch.Tensor(deform_landmark)
        if target_landmark is None:
            return self.deform_landmark
        self.landmark_dist = tb.landmark_distance(target_landmark, self.deform_landmark, round)
        dist_source_target = tb.landmark_distance(target_landmark, source_landmark, round)
        if verbose:
            print(
                f"Landmarks:\n\tBefore : {dist_source_target}\n\tAfter : {self.landmark_dist}"
            )
        return self.deform_landmark, self.landmark_dist, dist_source_target

    def get_landmark_dist(self):
        try:
            return float(self.landmark_dist)
        except AttributeError:
            return "not computed"

    def compute_DICE(
        self, source_segmentation, target_segmentation, plot=False, forward=True, verbose=True
    ):
        """Compute the DICE score of a regristration. Given the segmentations of
        a structure  (ex: ventricules) that should be present in both source and target image.
        it gives a score close to one if the segmentations are well matching after transformation.


        :param source_segmentation: Tensor of source size?
        :param target_segmentation:
        :return: (float) DICE score.
        """

        self.is_DICE_cmp = True
        deformator = self.mp.get_deformator() if forward else self.mp.get_deformation()
        device = source_segmentation.device
        if len(source_segmentation.shape) == 2 or (len(source_segmentation.shape)) == 3:
            source_segmentation = source_segmentation[None, None]
        self.source_seg_deformed = tb.imgDeform(
            source_segmentation, deformator.to(device),
            dx_convention=self.dx_convention,
            mode = 'nearest'
        )

        self.source_segmentation = source_segmentation
        self.target_segmentation = target_segmentation


        # source_deformed[source_deformed>1e-2] =1
        # prod_seg = source_deformed * target_segmentation
        # sum_seg = source_deformed + target_segmentation
        #
        # self.dice = 2*prod_seg.sum() / sum_seg.sum()
        # self.dice = tb.dice(source_deformed, target_segmentation)
        self.dice = tb.average_dice(self.source_seg_deformed, target_segmentation, verbose=verbose)
        if plot:
            fig, ax = plt.subplots()
            ax.imshow(tb.imCmp(target_segmentation, self.source_seg_deformed))
            plt.show()
        return self.dice, self.source_seg_deformed

    def get_DICE(self):
        # if self.is_DICE_cmp :
        #     return self.DICE
        # else:
        try:
            return self.dice
        except AttributeError:
            return "not computed"

    def get_ssd_def(self):
        image_def = tb.imgDeform(
            self.source, self.mp.get_deformator(), dx_convention=self.dx_convention
        )
        return float(cf.SumSquaredDifference(self.target)(image_def))

    def save(
        self,
        file_name,
        save_path = None,
        light_save=False,
        message=None,
        destination=None,
        file_csv=None,
        add_location_to_file = True
    ):
        """Save an optimisation to be later loaded and write all sort of info
        in a csv file


        Parameters:
        ---------------
        file_name  : str
            will appear in the file name
        save_path : str
            Path where to save the optimisation. by default the saving location is given by the
            constant : `OPTIM_SAVE_DIR`. You can change it in your environment (file .env)
        light_save : bool
            if True, only the initial momentum is saved.
            If False all data, integration, source and target are saved. Setting it to True
            save a lot of space on the disk, but you might not be able to get the whole
            registration back if the source image is different or the code used for
            computing it changed for any reason.
        message : str
            will appear in the csv storing all data
        destination : str
            path of the folder to store the csvfile overview
        file_csv : str
            name of the csv file to store the overview of the saved optimisation
            default is 'saved_optim/saves_overview.csv'
        add_location_to_file : bool
            add home name in saved file to track on which computer it has been computed. (default True)

        .. note ::
            Demeter allows you to save registration results as metamorphosis
            objects to be able to reuse, restart, visualize, or analyze the results later.
            By default, we store them in the `~/.local/share/Demeter_metamorphosis/`
            folder on linux (may wary on other platforms). You can change
            the default location by setting the `DEMETER_OPTIM_SAVE_DIR` environment variable.
            in the .env file. To locate it you can use the following commands: (in a python file
            or in a ipython terminal)

            `demeter.display_env_help()`

        Returns:
        ---------------
        file_save,
            name of the file saved
        path
            path of the file saved
        """

        if self.integration_diverged:
            print("Can't save optimisation that didn't converged")
            return 0
        self.to_device("cpu")
        if save_path is None:
            path = OPTIM_SAVE_DIR
        else:
            path = save_path

        ic(path)
        date_time = datetime.now()

        if len(self.mp.image.shape) == 4:
            n_dim = "2D"
        elif len(self.mp.image.shape) == 5:
            n_dim = "3D"
        else:
            raise ValueError(
                "Image dimension not understood, "
                f"got self.image.shape :{self.image.shape}"
            )

        id_num = 0


        # build file name
        location = os.getenv("HOME").split("/")[-1]
        location = f"_{location}" if add_location_to_file else ''
        def file_name_maker_(id_num):
            return (
                n_dim
                + date_time.strftime("_%Y%m%d_")
                + file_name
                + location
                + "_{:03d}".format(id_num)
                + ".pk1"
            )

        file_save = file_name_maker_(id_num)
        while file_save in os.listdir(path):
            id_num += 1
            file_save = file_name_maker_(id_num)

        state_dict = fill_saves_overview._optim_to_state_dict_(
            self,
            file_save,
            dict(
                time=date_time.strftime("%d/%m/%Y %H:%M:%S"),
                saved_file_name="",  # Petit hack pour me simplifier la vie.
                n_dim=n_dim,
            ),
            message=message,
        )
        fill_saves_overview._write_dict_to_csv(
            state_dict, path=destination, csv_file=file_csv
        )

        # =================
        # save the data
        # copy and clean dictionary containing all values
        dict_copy = {}
        dict_copy["format_version"] = 2
        dict_copy["optimizer_class"] = self.__class__.__name__
        dict_copy["light_save"] = light_save
        dict_copy["__repr__"] = self.__repr__()
        for k in FIELD_TO_SAVE:
            if k in ("mp", "data_term"):
                dict_copy[k] = None
            else:
                dict_copy[k] = getattr(self, k, None)
            if torch.is_tensor(dict_copy[k]):
                dict_copy[k] = dict_copy[k].cpu().detach()

        dict_copy["args"] = self.get_all_arguments()
        dict_copy["args"].update(
            {
                "lbfgs_max_iter": self.lbfgs_max_iter,
                "lbfgs_history_size": self.lbfgs_history_size,
                "adam_scheduler": self._adam_scheduler_type,
                "adam_grad_clip": self._adam_grad_clip,
            }
        )
        if not light_save:
            dict_copy["mp"] = self.mp  # For some reason 'mp' wasn't showing in __dict__

        if not isinstance(self.data_term, (dt.Ssd, dt.TimedSsd)):
            print(
                "\nBUG WARNING : An other data term than Ssd was detected"
                "For now our method can't save it, it is ok to visualise"
                "the optimisation, but be careful loading the optimisation.\n"
            )
        # save landmarks if they exist
        try:
            dict_copy["landmarks"] = (
                self.source_landmark,
                self.target_landmark,
                self.deform_landmark,
            )
        except AttributeError:
            # print('No landmark detected')
            pass
        try:
            dict_copy["segmentations"] = (
                self.source_segmentation,
                self.target_segmentation,
            #     self.source_seg_deformed
            )
            # if "Rigid" in self.__class__.__name__:
            #     dict_copy["segmentations"] += (self.source_seg_rotated,)
        except AttributeError:
            pass

        with open(os.path.join(path, file_save), "wb") as f:
            pickle.dump(dict_copy, f, pickle.HIGHEST_PROTOCOL)
        print(f"Optimisation saved in { os.path.join(path, file_save)} \n")

        return file_save, path

    def save_to_gif(self, object, file_name, folder=None, delay=40, clean=True):
        """
        Save a gif of the optimisation. The object must be a string containing at least
        one of the following : `image`,`residual`,`deformation`.

        :param object: str
            can be a string containing at least one of the following : `image`,`residual`,`deformation`. or a combination of them.
        :param file_name: str
            name of the file to save
        :param folder: str
            path of the folder to save the gif
        :param delay: int
            delay between each frame in the gif
        :param clean: bool
            if True, the images used to create the gif are deleted.
        """

        # prepare list of object
        if "image" in object and "deformation" in object:
            # source image
            fig, ax = plt.subplots()
            ax.imshow(self.source[0,0].cpu().numpy(), **DLT_KW_IMAGE)
            tb.gridDef_plot_2d(self.mp.id_grid,ax=ax,step=10,color="#E5BB5F",linewidth=3)

            image_list_for_gif = [fig_to_image(fig, ax)]
            image_kw = dict()
            for n in range(self.mp.n_step):
                deformation = self.mp.get_deformation(to_t=n+1).cpu()
                img = self.mp.image_stock[n, 0].cpu().numpy()
                fig, ax = plt.subplots()
                ax.imshow(img, **DLT_KW_IMAGE)
                tb.gridDef_plot_2d(
                    deformation,
                    ax=ax,
                    step=10,
                    # color='#FFC759',
                    color="#E5BB5F",
                    linewidth=3,
                )
                image_list_for_gif.append(fig_to_image(fig, ax))
            plt.close(fig)

        elif ("image" in object or "I" in object) and "quiver" in object:
            image_list_for_gif = []
            for n in range(self.n_step):
                deformation = self.mp.get_deformation(to_t = n+1).cpu()
                if n != 0:
                    deformation -= self.id_grid.cpu()
                img = self.mp.image_stock[n, 0].cpu().numpy()
                fig, ax = plt.subplots()
                ax.imshow(img, **DLT_KW_IMAGE)
                tb.quiver_plot(
                    deformation,
                    ax=ax,
                    step=10,
                    color="#E5BB5F",
                )
                image_list_for_gif.append(fig_to_image(fig, ax))
            image_kw = dict()
            plt.close(fig)

        elif 'image' in object and 'cmp' in object:
            method="segw"
            image_list_for_gif = [
                tb.imCmp(self.source, self.target, method=method)
            ]
            ic(image_list_for_gif[0].shape,image_list_for_gif[0].min(),image_list_for_gif[0].max())
            ic(self.target.max(),self.source.max())
            for n in range(self.mp.n_step):
                img = self.mp.image_stock[n,None]
                image_list_for_gif.append(
                    tb.imCmp(img, self.target, method=method)
                )

                ic(image_list_for_gif[0].shape,image_list_for_gif[0].min(),image_list_for_gif[0].max(),
                   img.max())

            image_kw = DLT_KW_IMAGE

        elif "image" in object or "I" in object:
            image_list_for_gif = [self.source[0,0].cpu().numpy()]
            tmp_list= [img[0].numpy() for img in self.mp.image_stock]
            image_list_for_gif += tmp_list
            image_kw = DLT_KW_IMAGE
        elif "residual" in object or "z" in object:
            image_list_for_gif = [z[0].numpy() for z in self.mp.residuals_stock]
            # image_kw = DLT_KW_RESIDUALS
            image_kw = dict(
                cmap="RdYlBu_r",
                origin="lower",
                vmin=self.mp.residuals_stock.min(),
                vmax=self.mp.residuals_stock.max(),
            )
        elif "deformation" in object:
            image_list_for_gif = []
            for n in range(self.mp.n_step):
                deformation = self.mp.get_deformation(to_t = n+1).cpu()
                if n == 0:
                    deformation += self.mp.id_grid.cpu()
                fig, ax = plt.subplots()
                tb.gridDef_plot_2d(
                    deformation,
                    ax=ax,
                    step=10,
                    color="black",
                    # color='#E5BB5F',
                    linewidth=5,
                )
                image_list_for_gif.append(fig_to_image(fig, ax))
            image_kw = dict()
            plt.close(fig)
        elif "quiver" in object:
            image_list_for_gif = []
            for n in range(self.mp.n_step):
                deformation = self.mp.get_deformation(to_t = n+1).cpu()
                if n != 0:
                    deformation -= self.id_grid.cpu()
                fig, ax = plt.subplots()
                tb.quiver_plot(
                    deformation,
                    ax=ax,
                    step=10,
                    color="black",
                )
                image_list_for_gif.append(fig_to_image(fig, ax))
            image_kw = dict()
            plt.close(fig)
        else:
            raise ValueError(
                "object must be a string containing at least"
                "one of the following : `image`,`residual`,`deformation`."
            )

        path, im = save_gif_with_plt(
            image_list_for_gif,
            file_name,
            folder,
            duplicate=True,
            image_args=image_kw,
            verbose=True,
            delay=delay,
            clean=clean,
        )
        return path, im

    def _get_loss_components(self):
        if self.loss_stock is None:
            raise ValueError("Loss history is not available.")
        loss_stock = self.loss_stock
        if isinstance(loss_stock, dict):
            data_loss = loss_stock.get("data_loss")
            norm_v_2 = loss_stock.get("norm_v_2")
            norm_l2_on_z = loss_stock.get("norm_l2_on_z")
        else:
            data_loss = loss_stock[:, 0]
            norm_v_2 = loss_stock[:, 1]
            norm_l2_on_z = loss_stock[:, 2] if loss_stock.shape[1] > 2 else None
        return data_loss, norm_v_2, norm_l2_on_z

    @staticmethod
    def _to_numpy_array(value):
        if torch.is_tensor(value):
            return value.detach().cpu().numpy()
        return value

    def _loss_history_length(self):
        if self.loss_stock is None or isinstance(self.loss_stock, str):
            return 0
        if isinstance(self.loss_stock, dict):
            try:
                return len(next(iter(self.loss_stock.values())))
            except StopIteration:
                return 0
        return len(self.loss_stock)

    # ==================================================================
    #                 PLOTS
    # ==================================================================

    def get_total_cost(self):
        data_loss, norm_v_2, norm_l2_on_z = self._get_loss_components()
        total_cost = data_loss + self.cost_cst * norm_v_2
        if norm_l2_on_z is not None:
            total_cost += self.cost_cst * norm_l2_on_z
        return total_cost

    def plot_cost(self, y_log=False):
        """To display the evolution of cost during the optimisation."""

        fig1, ax1 = plt.subplots(1, 2, figsize=(10, 5))
        if y_log:
            ax1[0].set_yscale("log")
            ax1[1].set_yscale("log")

        data_loss, norm_v_2, norm_l2_on_z = self._get_loss_components()
        ssd_plot = self._to_numpy_array(data_loss)
        ax1[0].plot(
            ssd_plot, "--", color="blue", label=self.data_term.__class__.__name__
        )
        ax1[1].plot(
            ssd_plot, "--", color="blue", label=self.data_term.__class__.__name__
        )

        nbpix = prod(self.source.shape[2:])
        normv_plot = self.cost_cst * self._to_numpy_array(norm_v_2)
        ax1[0].plot(normv_plot, "--", color="green", label="normv")
        ax1[1].plot(
            self._to_numpy_array(norm_v_2),
            "--",
            color="green",
            label="normv",
        )
        total_cost = ssd_plot + normv_plot
        if norm_l2_on_z is not None:
            norm_l2_on_z_val = self.cost_cst * self._to_numpy_array(norm_l2_on_z)
            total_cost += norm_l2_on_z_val
            ax1[0].plot(norm_l2_on_z_val, "--", color="orange", label="norm_l2_on_z")
            ax1[1].plot(
                self._to_numpy_array(norm_l2_on_z),
                "--",
                color="orange",
                label="norm_l2_on_z",
            )

        ax1[0].plot(total_cost, color="black", label=r"$\Sigma$")
        ax1[0].legend()
        ax1[1].legend()
        ax1[0].set_title(
            "Lambda = " + str(self.cost_cst) + " rho = " + str(self._get_rho_())
        )
        return fig1, ax1

    def plot_imgCmp(self, origin= "lower", cmp_method = 'compose'):
        r"""Display and compare the deformed image $I_1$ with the target$"""

        fig, ax = plt.subplots(2, 2, figsize=(7, 7), constrained_layout=True)
        image_kw = dict(cmap="gray", origin=origin, vmin=0, vmax=1)
        set_ticks_off(ax)
        ax[0, 0].imshow(self.source[0, 0, :, :].detach().cpu().numpy(), **image_kw)
        ax[0, 0].set_title("source", fontsize=25)
        ax[0, 1].imshow(self.target[0, 0, :, :].detach().cpu().numpy(), **image_kw)
        ax[0, 1].set_title("target", fontsize=25)

        ax[1, 1].imshow(
            tb.imCmp(self.target, self.mp.image.detach().cpu(), method=cmp_method)[0],
            **image_kw,
        )
        ax[1, 1].set_title("comparaison deformed image with target", fontsize=25)
        ax[1, 0].imshow(self.mp.image[0, 0].detach().cpu().numpy(), **image_kw)
        ax[1, 0].set_title("Integrated source image", fontsize=25)
        tb.quiver_plot(
            self.mp.get_deformation().detach().cpu() - self.mp.id_grid,
            ax=ax[1, 1],
            step=15,
            color=GRIDDEF_YELLOW,
            dx_convention=self.dx_convention,
        )

        try:
            text_param = f"rho = {self.mp._get_rho_()},"
        except AttributeError:
            text_param = ""
        try:
            text_param += f" gamma = {self.mp._get_gamma_()}"
        except AttributeError:
            pass
        ax[1, 1].text(10, self.source.shape[2] - 10, text_param, c="white", size=25)

        text_score = ""
        if type(self.get_DICE()) is float:
            text_score += f"dice : {self.get_DICE():.2f},"

        if type(self.get_landmark_dist()) is float:
            ax[1, 1].plot(
                self.source_landmark[:, 0], self.source_landmark[:, 1], **source_ldmk_kw
            )
            ax[1, 1].plot(
                self.target_landmark[:, 0], self.target_landmark[:, 1], **target_ldmk_kw
            )
            ax[1, 1].plot(
                self.deform_landmark[:, 0], self.deform_landmark[:, 1], **deform_ldmk_kw
            )
            ax[1, 1].quiver(
                self.source_landmark[:, 0],
                self.source_landmark[:, 1],
                self.deform_landmark[:, 0] - self.source_landmark[:, 0],
                self.deform_landmark[:, 1] - self.source_landmark[:, 1],
                color="#2E8DFA",
            )
            ax[1, 1].legend()
            text_score += f"landmark : {self.get_landmark_dist():.2f},"
        ax[1, 1].text(10, 10, text_score, c="white", size=25)

        return fig, ax

    def plot_deform(self, temporal_nfigs=0):
        r"""Display the deformation of the source image to the target image and the source image deformed
        by the deformation field.
        """

        if self.optimized_momenta is None:
            raise ValueError("No optimized momenta available, run forward() first.")
        residuals = self.optimized_momenta
        # print(residuals.device,self.source.device)
        self.mp.forward(self.source.clone(), residuals, save=True, plot=0)
        self.mp.plot_deform(self.target, temporal_nfigs)

    def plot(self, y_log=False):
        fig_c, ax_c = self.plot_cost()
        fig_i, ax_i = self.plot_imgCmp()
        return (fig_c, ax_c), (fig_i, ax_i)
