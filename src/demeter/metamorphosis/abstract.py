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
import os, sys, csv  # , time
from icecream import ic

from datetime import datetime
from abc import ABC, abstractmethod

from ..utils.optim import GradientDescent
from demeter.constants import *
from ..utils import torchbox as tb
from ..utils import vector_field_to_flow as vff
from ..utils.toolbox import (
    update_progress,
    fig_to_image,
    save_gif_with_plt,
)
from ..utils.decorators import time_it
from ..utils import cost_functions as cf
from ..utils import fill_saves_overview as fill_saves_overview

from ..metamorphosis import data_cost as dt

# TO DRAW THE BACKWARD GRAPH
# from torchviz import make_dot

# =========================================================================
#
#            Abstract classes
#
# =========================================================================
# See them as a toolkit


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


    """

    @abstractmethod
    def __init__(self, kernelOperator, n_step, dx_convention="pixel",**kwargs):
        super().__init__()
        self._force_save = False
        self._detach_image = True
        self.dx_convention = dx_convention

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
    def step(self):
        pass

    def forward(
        self,
        image,
        momentum_ini,
        save=True,
        plot=0,
        t_max=1,
        verbose=False,
        sharp=None,
        debug=False,
        hamiltonian_integration=False,
    ):
        r"""This method is doing the temporal loop using the good method `_step_`

        Parameters
        ----------
        image : tensor array of shape [1,1,H,W]
            Source image ($I_0$)
        momentum_ini : tensor array of shape [1,1,H,W]
            Momentum ($p_0$) or residual ($z_0$)
        save : bool, optional
            Option to save the integration intermediary steps, by default True
            it saves the image, field and momentum at each step in the attributes
            `image_stock`, `field_stock` and `momentum_stock`.
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

        """

        if len(momentum_ini.shape) not in [4, 5]:
            raise ValueError(
                f"residual_ini must be of shape [B,C,H,W] or [B,C,D,H,W] got {momentum_ini.shape}"
            )
        device = momentum_ini.device
        # print(f'sharp = {sharp} flag_sharp : {self.flag_sharp},{self._phis}')
        self._init_sharp_(sharp)
        self.source = image.detach()
        self.image = image.clone().to(device)
        self.momentum = momentum_ini
        self.debug = debug
        self.flag_hamiltonian_integration = hamiltonian_integration
        try:
            self.save = True if self._force_save else save
        except AttributeError:
            self.save = save

        self.id_grid = tb.make_regular_grid(
            momentum_ini.shape[2:], dx_convention=self.dx_convention, device=device
        )
        assert self.id_grid != None

        # field initialization to a regular grid
        self.field = self.id_grid.clone().to(device)

        if plot > 0:
            self.save = True

        if self.save:
            self.image_stock = torch.zeros((t_max * self.n_step,) + image.shape[1:])
            self.field_stock = torch.zeros(
                (t_max * self.n_step,) + self.field.shape[1:]
            )
            self.momentum_stock = torch.zeros(
                (t_max * self.n_step,) + momentum_ini.shape[1:]
            )

        if self.flag_hamiltonian_integration:
            self.norm_v = 0
            self.norm_z = 0

        for i, t in enumerate(torch.linspace(0, t_max, t_max * self.n_step)):
            self._i = i

            _, field_to_stock, residuals_dt = self.step()

            if self.flag_hamiltonian_integration:
                self.norm_v += self.norm_v_i / self.n_step
                self.norm_z += self.norm_z_i / self.n_step
                # self.ham_integration += self.ham_value / self.n_step
            # ic(self._i,self.field.min().item(),self.field.max().item(),
            #    self.momentum.min().item(),self.momentum.max().item(),
            #     self.image.min().item(),self.image.max().item())

            if self.image.isnan().any() or self.momentum.isnan().any():
                raise OverflowError(
                    "Some nan where produced ! the integration diverged",
                    "changing the parameters is needed. "
                    "You can try:"
                    "\n- increasing n_step (deformation more complex"
                    "\n- decreasing grad_coef (convergence slower but more stable)"
                    "\n- increasing sigma_v (catching less details)",
                )

            if self.save:
                if self._detach_image:
                    self.image_stock[i] = self.image[0].detach().to("cpu")
                else:
                    self.image_stock[i] = self.image[0]
                self.field_stock[i] = field_to_stock[0].detach().to("cpu")
                self.momentum_stock[i] = residuals_dt.detach().to("cpu")

            if verbose:
                update_progress(i / (t_max * self.n_step))
                if self.flag_hamilt_integration:
                    print('ham :', self.ham_value.detach().cpu().item(),
                      self.norm_v.detach().cpu().item(),
                      self.norm_z.detach().cpu().item())

        # try:
        #     _d_ = device if self._force_save else 'cpu'
        #     self.field_stock = self.field_stock.to(device)
        # except AttributeError: pass

        if plot > 0:
            self.plot(n_figs=plot)

    def _image_Eulerian_integrator_(self, image, vector_field, t_max, n_step):
        """image integrator using an Eulerian scheme

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
        r"""operate the equation $K \star (z_t \cdot \nabla I_t)$

        :param momentum: (tensor array) of shape [H,W] or [D,H,W]
        :param grad_image: (tensor array) of shape [B,C,2,H,W] or [B,C,3,D,H,W]
        :return: (tensor array) of shape [B,H,W,2]
        """

        # C = residuals.shape[1]
        field_momentum = -(momentum.unsqueeze(2) * grad_image).sum(dim=1)
        field =  self.kernelOperator(field_momentum)

        norm_v = None
        if self.flag_hamiltonian_integration:
            norm_v = .5 * self.rho * (field_momentum.clone() * field.clone()).sum()

        return tb.im2grid(field), norm_v

    def _compute_vectorField_multimodal_(self, momentum, grad_image):
        r"""operate the equation $K \star (z_t \cdot \nabla I_t)$

        :param momentum: (tensor array) of shape [B,C,H,W] or [B,C,D,H,W]
        :param grad_image: (tensor array) of shape [B,C,2,H,W] or [B,C,3,D,H,W]
        :return: (tensor array) of shape [B,H,W,2]
        """

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
        self.field = self._compute_vectorField_multimodal_(self.momentum, grad_image)
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
    def _update_field_(self):
        grad_image = tb.spatialGradient(self.image, dx_convention=self.dx_convention)
        # ic(grad_image.min().item(), grad_image.max().item(),self.dx_convention)
        self.field,self.norm_v_i = self._compute_vectorField_(self.momentum, grad_image)
        # self.field *= self._field_cst_mult()
        # self.field *= sqrt(self.rho)

    # Done
    def _update_momentum_Eulerian_(self):
        momentum_dt = -tb.Field_divergence(dx_convention=self.dx_convention)(
            self.momentum[0, 0][None, :, :, None] * self.field,
        )

        self.momentum = self.momentum + sqrt(self.rho) * momentum_dt / self.n_step

    # Done
    def _update_momentum_semiLagrangian_(self, deformation):
        warnings.warn(
            "ANTON ! You should not use this function but "
            "_compute_div_momentum_semiLagrangian_() instead !"
        )
        div_v_times_z = (
            self.momentum
            * tb.Field_divergence(dx_convention=self.dx_convention)(self.field)[0, 0]
        )
        self.momentum = (
            tb.imgDeform(
                self.momentum,
                deformation,
                dx_convention=self.dx_convention,
                clamp=False,
            )
            - div_v_times_z / self.n_step
        )

    def _compute_div_momentum_semiLagrangian_(self,
                                              deformation,
                                              momentum,
                                              cst,
                                              field = None
                                              ):
        r"""
        Compute the divergence of the momentum in the semiLagrangian scheme
        meaning
        $$ c \times \nabla \cdot (a v) = cv \cdot \nabla a + c a \nabla \cdot v$$
        with $cv \cdot \nabla a$ being the transport of a by the deformation $x + cv$
        with $a : \Omega \to \mathbb{R}$ and $v : \Omega \to \mathbb{R}^d$

        Parameters
        ----------

        deformation (tensor array)
            tensor of shape [1,H,W,2] or [1,D,H,W,3]
        momentum (tensor array)
            tensor of shape [1,1,H,W] or [1,1,D,H,W]
        cst (float | tensor array)
            the constant $c$ in the above equation, if tensor array,
             it must have the same shape as momentum

        Returns
        -------
        tensor array of shape [1,1,H,W] or [1,1,D,H,W]
        """

        if field is None:
            field = self.field
        div_v_times_p = cst * (
            momentum
            * tb.Field_divergence(dx_convention=self.dx_convention)(field)[0, 0]
        )
        momentum = (
            tb.imgDeform(
                momentum, deformation, dx_convention=self.dx_convention, clamp=False
            )
            - div_v_times_p / self.n_step
        )
        return momentum

    def _compute_sharp_intermediary_residuals_(self):
        device = self.momentum.device
        resi_cumul = torch.zeros(self.momentum.shape, device=device)
        # for k,phi in enumerate(self._phis[self._i][:]):
        for k, phi in enumerate(self._phis[self._i][1:]):
            resi_cumul += tb.imgDeform(
                self.momentum_stock[k][None].to(device),
                phi,
                dx_convention=self.dx_convention,
                clamp=False,
            )
        resi_cumul = resi_cumul + self.momentum
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
    def _update_image_semiLagrangian_(self, deformation, residuals=None, sharp=False):
        if residuals is None:
            # z = sqrt(1 - rho) * p and I = v gradI + sqrt(1-rho) * z
            residuals = (1 - self.rho) * self.momentum
        self.norm_z_i = None
        if self.flag_hamiltonian_integration:
            self.norm_z_i = .5 * residuals.pow(2).sum()
        image = self.source if sharp else self.image
        # if self.rho > 0:
        self.image = tb.imgDeform(image, deformation, dx_convention=self.dx_convention)

        if self._get_rho_() < 1:
            self.image += residuals / self.n_step

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
        self, deformation, residuals=None, sharp=False
    ):
        if residuals is None:
            residuals = self.momentum
        image = self.source if sharp else self.image
        self.image = tb.imgDeform(image, deformation, dx_convention=self.dx_convention)
        self.image += residuals / self.n_step

        # (self.rf.mu * self.rf.mask[self._i] * residuals) / self.n_step

        # ga = (self.rf.mask[self._i] * self.residuals) / self.n_step
        # plt.figure()
        # p = plt.imshow(ga[0])
        # plt.colorbar(p)
        # plt.show()

    def _update_field_oriented_weighted_(self):
        grad_image = tb.spatialGradient(self.image, dx_convention=self.dx_convention)
        free_field = tb.im2grid(
            (self.momentum * grad_image[0]) * torch.sqrt(self.residual_mask[self._i])
        )
        oriented_field = 0
        if self.flag_O:

            oriented_field = (self.orienting_field[self._i][None]
                                    * self.orienting_mask[self._i][..., None])

        self.field = -tb.im2grid(
            self.kernelOperator(tb.grid2im(free_field + oriented_field))
        )

    def to_device(self, device):
        # TODO: completer ça
        try:
            self.image = self.image.to(device)
            self.id_grid = self.id_grid.to(device)
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
            method="temporal", save=save, dx_convention=self.dx_convention
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
            method="temporal", save=save, dx_convention=self.dx_convention
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
        v_abs_max = torch.quantile(self.momentum.abs(), 0.99)
        kw_residuals_args = dict(
            cmap="RdYlBu_r",
            extent=[-1, 1, -1, 1],
            origin="lower",
            vmin=-v_abs_max,
            vmax=v_abs_max,
        )
        size_fig = 5
        C = self.momentum_stock.shape[1]
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
                    self.momentum_stock[t, j].detach().numpy(), **kw_residuals_args
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
            method="temporal", save=temporal, dx_convention=self.dx_convention
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
            self.source.cpu(), full_deformator, dx_convention=self.dx_convention
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
            tb.imCmp(target[:, 0][None], S_deformed[:, 0][None], method="compose"),
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
        optimizer_method: str = "grad_descent",
        hamiltonian_integration=False,
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
        grad_source = tb.spatialGradient(image, dx_convention=self.dx_convention)
        field_momentum = (grad_source * momentum.unsqueeze(2)).sum(dim=1)  # / C
        field = self.mp.kernelOperator(field_momentum)

        norm_v = (field_momentum * field).sum()
        if norm_v < 0:
            warnings.warn(f"norm_v is negative : {norm_v}, increasing"
                          f" kernel_reach in kernelOperator might help")
        return norm_v

    @abstractmethod
    def cost(self, residual_ini):
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
            return float(self._compute_V_norm_(self.to_analyse[0], self.source))
        else:
            dist = float(
                self._compute_V_norm_(self.mp.momentum_stock[0][None], self.mp.source)
            )
            for t in range(self.mp.momentum_stock.shape[0] - 1):
                dist += float(
                    self._compute_V_norm_(
                        self.mp.momentum_stock[t + 1][None],
                        self.mp.image_stock[t][None],
                    )
                )
            return dist

    def __repr__(self) -> str:
        return (
            self.__class__.__name__
            + "(cost_parameters : {"
            + ", \n\t\trho ="
            + str(self._get_rho_())
            + ", \n\t\tlambda ="
            + str(self.cost_cst)
            + "\n\t},"
            + f"\n\tgeodesic integrator : "
            + self.mp.__repr__()
            + f"\n\tintegration method : "
            + self.mp.step.__name__
            + f"\n\toptimisation method : "
            + self.optimizer_method_name
            + f"\n\t# geodesic steps ="
            + str(self.mp.n_step)
            + "\n)"
        )

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #   Implemented OPTIMIZERS
    # GRADIENT DESCENT
    def _initialize_grad_descent_(self, dt_step, max_iter=20):
        self.optimizer = GradientDescent(self.cost, self.parameter, lr=dt_step)

    def _step_grad_descent_(self):
        self.optimizer.step(verbose=False)

    # LBFGS
    def _initialize_LBFGS_(self, dt_step, max_iter=20):
        self.optimizer = torch.optim.LBFGS(
            [self.parameter],
            max_eval=30,
            max_iter=max_iter,
            lr=dt_step,
            line_search_fn="strong_wolfe",
        )

        def closure():
            self.optimizer.zero_grad()
            L = self.cost(self.parameter)
            # save best cms
            # if(self._it_count >1 and L < self._loss_stock[:self._it_count].min()):
            #     cms_tosave.data = self.cms_ini.detach().data
            L.backward()
            return L

        self.closure = closure

    def _step_LBFGS_(self):
        self.optimizer.step(self.closure)

    def _initialize_adadelta_(self, dt_step, max_iter=None):
        self.optimizer = torch.optim.Adadelta(
            [self.parameter], lr=dt_step, rho=0.9, weight_decay=0
        )

        def closure():
            self.optimizer.zero_grad()
            L = self.cost(self.parameter)
            assert not torch.isnan(
                L
            ).any(), f"Loss is NaN at iteration {self._it_count}"
            # save best cms
            # if(self._it_count >1 and L < self._loss_stock[:self._it_count].min()):
            #     cms_tosave.data = self.cms_ini.detach().data
            L.backward()
            ic("adadelta", L, self.parameter.grad)
            # dot = make_dot(L, params=dict(x=self.parameter))
            # dot.render("torch_backward_graph", format="png")
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
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(self.mp.image[0, 0].detach().cpu(), **DLT_KW_IMAGE)
        a = ax[1].imshow(
            self.mp.momentum[0, 0].detach().cpu(), cmap="RdYlBu_r", origin="lower"
        )
        fig.colorbar(a, ax=ax[1], fraction=0.046, pad=0.04)
        plt.show()

    @time_it
    def forward(
        self, z_0, n_iter=10, grad_coef=1e-3, verbose=True, plot=False, sharp=None,
            debug=False
    ):
        r"""The function is and perform the optimisation with the desired method.
        The result is stored in the tuple self.to_analyse with two elements. First element is the optimized
        initial residual ($z_O$ in the article) used for the shooting.
        The second is a tensor with the values of the loss norms over time. The function
        plot_cost() is designed to show them automatically.

        :param z_0: initial residual. It is the variable on which we optimize.
        `require_grad` must be set to True.
        :param n_iter: (int) number of optimizer iterations
        :param verbose: (bool) display advancement

        """

        # self.source = self.source.to(z_0.device)
        # self.target = self.target.to(z_0.device)
        # self.mp.kernelOperator.kernel = self.mp.kernelOperator.kernel.to(z_0.device)
        # self.data_term.to_device(z_0.device)

        self.parameter = z_0  # optimized variable
        self._initialize_optimizer_(grad_coef, max_iter=n_iter)
        self.n_iter = n_iter

        self.id_grid = tb.make_regular_grid(
            z_0.shape[2:], dx_convention=self.dx_convention, device=z_0.device
        )
        self.to_device(z_0.device)
        if self.id_grid is None:
            raise ValueError(
                f"The initial momentum provided might have the wrong shape, got :{z_0.shape}"
            )

        self.cost(self.parameter)

        loss_stock = self._cost_saving_(n_iter, None)  # initialisation
        loss_stock = self._cost_saving_(0, loss_stock)

        for i in range(1, n_iter):
            self._iter = i

            self._step_optimizer_()
            loss_stock = self._cost_saving_(i, loss_stock)

            if verbose:
                update_progress(
                    (i + 1) / n_iter,
                    message=(
                        f"{self.data_term.__class__.__name__} :",
                        loss_stock[i, 0],
                    ),
                )
            if plot and i in [n_iter // 4, n_iter // 2, 3 * n_iter // 4]:
                self._plot_forward_()

        # for future plots compute shooting with save = True
        self.mp.forward(
            self.source.clone(), self.parameter.detach().clone(), save=True, plot=0,debug=debug
        )

        self.to_analyse = (self.parameter.detach(), loss_stock)
        self.to_device("cpu")

    def to_device(self, device):
        # self.mp.kernelOperator.kernel = self.mp.kernelOperator.kernel.to(device)
        self.mp.to_device(device)
        self.source = self.source.to(device)
        self.target = self.target.to(device)
        self.data_term.to_device(device)
        # self.target = self.target.to(device)
        self.parameter = self.parameter.to(device)
        self.id_grid = self.id_grid.to(device)
        self.data_term.to_device(device)
        try:
            # To analyse might not have been initialized yet.
            self.to_analyse = (
                self.to_analyse[0].to(device),
                self.to_analyse[1].to(device),
            )
        except AttributeError:
            pass

    def forward_safe_mode(
        self, z_0, n_iter=10, grad_coef=1e-3, verbose=True, mode=None
    ):
        """Same as Optimize_geodesicShooting.forward(...) but
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
        :return:
        """

        try:
            self.forward(z_0, n_iter, grad_coef, verbose=verbose)
        except OverflowError:
            if mode is None:
                print("Integration diverged : Stop.\n\n")
                self.to_analyse = "Integration diverged"
            elif mode == "grad_coef":
                print(f"Integration diverged :" f" set grad_coef to {grad_coef*0.1}")
                self.forward_safe_mode(z_0, n_iter, grad_coef * 0.1, verbose, mode=mode)

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

        def land_dist(land_1, land_2):
            # print(f"land type : {land_1.dtype}, {land_2.dtype}")
            print(f"round {round}")
            if  not round or land_2.dtype == torch.int :
                return (land_1 - land_2).abs().mean()
            else:
                return (land_1 - land_2.round()).abs().mean()

        self.source_landmark = source_landmark
        self.target_landmark = target_landmark
        self.deform_landmark = torch.Tensor(deform_landmark)
        if target_landmark is None:
            return self.deform_landmark
        self.landmark_dist = land_dist(target_landmark, self.deform_landmark)
        dist_source_target = land_dist(target_landmark, source_landmark)
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
        self, source_segmentation, target_segmentation, plot=False, forward=True
    ):
        """Compute the DICE score of a regristration. Given the segmentations of
        a structure  (ex: ventricules) that should be present in both source and target image.
        it gives a score close to one if the segmentations are well matching after transformation.


        :param source_segmentation: Tensor of source size?
        :param target_segmentation:
        :return: (float) DICE score.
        """

        # TODO : make the computation
        self.is_DICE_cmp = True
        deformator = self.mp.get_deformator() if forward else self.mp.get_deformation()
        device = source_segmentation.device
        if len(source_segmentation.shape) == 2 or (len(source_segmentation.shape)) == 3:
            source_segmentation = source_segmentation[None, None]
        source_deformed = tb.imgDeform(
            source_segmentation, deformator.to(device), dx_convention=self.dx_convention
        )
        # source_deformed[source_deformed>1e-2] =1
        # prod_seg = source_deformed * target_segmentation
        # sum_seg = source_deformed + target_segmentation
        #
        # self.dice = 2*prod_seg.sum() / sum_seg.sum()
        self.dice = tb.dice(source_deformed, target_segmentation)
        if plot:
            fig, ax = plt.subplots()
            ax.imshow(tb.imCmp(target_segmentation, source_deformed))
            plt.show()
        return self.dice, source_deformed

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

        if self.to_analyse == "Integration diverged":
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
                "got self.image.shape :{self.image.shape}"
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
        dict_copy["light_save"] = light_save
        dict_copy["__repr__"] = self.__repr__()
        for k in FIELD_TO_SAVE:
            dict_copy[k] = self.__dict__.get(k)
            if torch.is_tensor(dict_copy[k]):
                dict_copy[k] = dict_copy[k].cpu().detach()

        dict_copy["args"] = self.get_all_arguments()
        if not light_save:
            dict_copy["mp"] = self.mp  # For some reason 'mp' wasn't showing in __dict__

        if type(self.data_term) != dt.Ssd:
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
            image_list_for_gif = [z[0].numpy() for z in self.mp.momentum_stock]
            # image_kw = DLT_KW_RESIDUALS
            image_kw = dict(
                cmap="RdYlBu_r",
                origin="lower",
                vmin=self.mp.momentum_stock.min(),
                vmax=self.mp.momentum_stock.max(),
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



    # ==================================================================
    #                 PLOTS
    # ==================================================================

    def get_total_cost(self):
        total_cost = self.to_analyse[1][:, 0] + self.cost_cst * self.to_analyse[1][:, 1]
        if self._get_rho_() < 1:
            if type(self._get_rho_()) == float:
                total_cost += (
                    self.cost_cst * (self._get_rho_()) * self.to_analyse[1][:, 2]
                )
            elif type(self._get_rho_()) == tuple:
                total_cost += (
                    self.cost_cst * (self._get_rho_()[0]) * self.to_analyse[1][:, 2]
                )
        return total_cost

    def plot_cost(self, y_log=False):
        """To display the evolution of cost during the optimisation."""

        fig1, ax1 = plt.subplots(1, 2, figsize=(10, 5))
        if y_log:
            ax1[0].set_yscale("log")
            ax1[1].set_yscale("log")

        ssd_plot = self.to_analyse[1][:, 0].numpy()
        ax1[0].plot(
            ssd_plot, "--", color="blue", label=self.data_term.__class__.__name__
        )
        ax1[1].plot(
            ssd_plot, "--", color="blue", label=self.data_term.__class__.__name__
        )

        nbpix = prod(self.source.shape[2:])
        normv_plot = self.cost_cst * self.to_analyse[1][:, 1].detach().numpy()
        ax1[0].plot(normv_plot, "--", color="green", label="normv")
        ax1[1].plot(
            self.to_analyse[1][:, 1].detach().numpy(),
            "--",
            color="green",
            label="normv",
        )
        total_cost = ssd_plot + normv_plot
        if self._get_rho_() < 1:
            norm_l2_on_z = self.cost_cst * self.to_analyse[1][:, 2].numpy()
            total_cost += norm_l2_on_z
            ax1[0].plot(norm_l2_on_z, "--", color="orange", label="norm_l2_on_z")
            ax1[1].plot(
                self.to_analyse[1][:, 2].numpy(),
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

    def plot_imgCmp(self):
        r"""Display and compare the deformed image $I_1$ with the target$"""

        fig, ax = plt.subplots(2, 2, figsize=(20, 20), constrained_layout=True)
        image_kw = dict(cmap="gray", origin="lower", vmin=0, vmax=1)
        set_ticks_off(ax)
        ax[0, 0].imshow(self.source[0, 0, :, :].detach().cpu().numpy(), **image_kw)
        ax[0, 0].set_title("source", fontsize=25)
        ax[0, 1].imshow(self.target[0, 0, :, :].detach().cpu().numpy(), **image_kw)
        ax[0, 1].set_title("target", fontsize=25)

        ax[1, 1].imshow(
            tb.imCmp(self.target, self.mp.image.detach().cpu(), method="compose"),
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

        residuals = self.to_analyse[0]
        # print(residuals.device,self.source.device)
        self.mp.forward(self.source.clone(), residuals, save=True, plot=0)
        self.mp.plot_deform(self.target, temporal_nfigs)

    def plot(self, y_log=False):
        fig_c, ax_c = self.plot_cost()
        fig_i, ax_i = self.plot_imgCmp()
        return (fig_c, ax_c), (fig_i, ax_i)
