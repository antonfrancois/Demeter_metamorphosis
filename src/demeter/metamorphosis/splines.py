"""Shooting integration for balanced metamorphosis splines in 2D.

The evolving state is ``(I, p, a, r)`` from system (43) of
``metamorphosis_splines_optimal_control.pdf``. The force ``u`` is algebraic:
``u = A_I^-1 a``. Control values are absolute right limits of the jerk, not
increments.

The appendix's image action and its adjoint are expanded into their scalar and
density transport equations. This keeps the semi-Lagrangian step explicit and
avoids conflating that action with the vector-field Lie bracket.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import isclose, isfinite, sqrt

import torch
from torch import Tensor
from torch.utils._pytree import register_pytree_node

from .abstract import Geodesic_integrator, Optimize_geodesicShooting
from .data_cost import TimedSsd
from .var_classes import TorchDataClass
from ..utils import torchbox as tb
from ..utils.cometric_inversion import CometricOperator
from ..utils.reproducing_kernels import SobolevFluidOperator
from ..utils.spline_data import validate_timed_observations
from ..utils.temporal_preconditioning import (
    TemporalTransform,
    cometric_temporal_metric,
)


@dataclass
class SplinesVariables(TorchDataClass):
    """Optimized shooting variables for a 2D metamorphosis spline.

    All initial fields have shape ``[1, 1, H, W]``. ``control_jerks`` has
    time-major shape ``[C, 1, 1, H, W]``, where entry ``c`` is the absolute
    right-limit value ``r(tau_c^+)``.
    """

    initial_momentum: Tensor
    initial_acceleration: Tensor
    initial_jerk: Tensor
    control_jerks: Tensor

    def __post_init__(self) -> None:
        super().__post_init__()
        reference = self.initial_momentum
        if (
            reference.ndim != 4
            or reference.shape[:2] != (1, 1)
            or min(reference.shape[-2:]) < 2
        ):
            raise ValueError(
                "initial_momentum must have shape [1, 1, H, W] with H,W >= 2, "
                f"got {tuple(reference.shape)}"
            )
        for name, tensor in (
            ("initial_acceleration", self.initial_acceleration),
            ("initial_jerk", self.initial_jerk),
        ):
            if tensor.shape != reference.shape:
                raise ValueError(
                    f"{name} must have shape {tuple(reference.shape)}, "
                    f"got {tuple(tensor.shape)}"
                )

        if self.control_jerks.ndim != 5:
            raise ValueError(
                "control_jerks must have shape [C, 1, 1, H, W], "
                f"got {tuple(self.control_jerks.shape)}"
            )
        expected_control_shape = (self.control_jerks.shape[0],) + tuple(
            reference.shape
        )
        if tuple(self.control_jerks.shape) != expected_control_shape:
            raise ValueError(
                "control_jerks must have shape [C, 1, 1, H, W], "
                f"got {tuple(self.control_jerks.shape)}"
            )

        for name, tensor in self:
            if not torch.is_floating_point(tensor):
                raise TypeError(f"{name} must be floating point, got {tensor.dtype}")
            if tensor.device != reference.device or tensor.dtype != reference.dtype:
                raise ValueError(
                    "all spline variables must share one device and dtype"
                )

    @property
    def n_controls(self) -> int:
        return self.control_jerks.shape[0]

    @classmethod
    def zeros(
        cls,
        image: Tensor,
        n_controls: int = 0,
        *,
        requires_grad: bool = True,
    ) -> "SplinesVariables":
        """Create zero shooting variables matching a ``[1, 1, H, W]`` image."""
        if n_controls < 0:
            raise ValueError("n_controls must be non-negative")
        variables = cls(
            initial_momentum=torch.zeros_like(image),
            initial_acceleration=torch.zeros_like(image),
            initial_jerk=torch.zeros_like(image),
            control_jerks=image.new_zeros((n_controls,) + tuple(image.shape)),
        )
        variables.requires_grad_(requires_grad)
        return variables

def _splines_variables_flatten(variables: SplinesVariables):
    return tuple(tensor for _, tensor in variables), None


def _splines_variables_unflatten(children, _context):
    return SplinesVariables(*children)


register_pytree_node(
    SplinesVariables,
    _splines_variables_flatten,
    _splines_variables_unflatten,
)


class MetamorphosisSplineIntegrator(Geodesic_integrator):
    r"""Integrate system (43) on the pixel torus.

    The physical deformation velocity is

    .. math::
        v=-\sqrt{\rho}\,K(p\nabla I),

    while ``field`` stores the effective transport field
    :math:`b=\sqrt{\rho}v=-\rho K(p\nabla I)` expected by Demeter's flow
    reconstruction. Control times must lie on the uniform integration mesh so
    no numerical step crosses a jerk discontinuity.
    """

    _DIAGNOSTIC_STOCK_NAMES = (
        "time_stock",
        "image_stock",
        "momentum_stock",
        "acceleration_stock",
        "jerk_stock",
        "field_stock",
        "residuals_stock",
        "force_stock",
        "velocity_stock",
    )

    def __init__(
        self,
        rho: float,
        control_times: Sequence[float] = (),
        *,
        kernelOperator,
        n_step: int,
        cg_eps: float = 1e-6,
        dx_convention: str = "pixel",
        save_gpu_memory: bool = False,
        debug: bool = False,
    ) -> None:
        rho = float(rho)
        cg_eps = float(cg_eps)
        if not isfinite(rho) or not 0 <= rho < 1:
            raise ValueError("rho must be finite and satisfy 0 <= rho < 1")
        if not isfinite(cg_eps) or cg_eps <= 0:
            raise ValueError("cg_eps must be finite and strictly positive")
        if not isinstance(n_step, int) or isinstance(n_step, bool) or n_step < 1:
            raise ValueError("n_step must be a strictly positive integer")
        if dx_convention != "pixel":
            raise ValueError(
                "the 2D spline integrator currently requires dx_convention='pixel'"
            )
        if save_gpu_memory:
            raise NotImplementedError(
                "checkpointed spline integration is not implemented"
            )
        if not isinstance(kernelOperator, SobolevFluidOperator):
            raise TypeError(
                "kernelOperator must be a SobolevFluidOperator with matched L/K"
            )
        controls = tuple(float(time) for time in control_times)
        if any(not isfinite(time) or not 0 < time < 1 for time in controls):
            raise ValueError("control times must be finite and lie strictly in (0, 1)")
        if any(right <= left for left, right in zip(controls, controls[1:])):
            raise ValueError("control times must be strictly increasing")

        control_steps = []
        for time in controls:
            exact_step = time * n_step
            step = round(exact_step)
            if not isclose(exact_step, step, rel_tol=0, abs_tol=1e-6):
                raise ValueError(
                    f"control time {time} is not on the {n_step}-step temporal mesh"
                )
            if not 1 <= step < n_step - 1:
                raise ValueError(
                    "control times must map before the final interior mesh node"
                )
            control_steps.append(step)
        if any(
            right <= left
            for left, right in zip(control_steps, control_steps[1:])
        ):
            raise ValueError("control times must map to distinct mesh nodes")

        super().__init__(
            kernelOperator=kernelOperator,
            n_step=n_step,
            dx_convention=dx_convention,
            save_gpu_memory=save_gpu_memory,
            debug=debug,
        )
        self.rho = rho
        self.cg_eps = cg_eps
        self.control_times = controls
        self.control_steps = tuple(control_steps)
        self._control_by_step = {
            step: index for index, step in enumerate(self.control_steps)
        }
        self.dt = 1 / n_step
        self.field_integration_boundary = "periodic"

    def _get_rho_(self) -> float:
        return self.rho

    @staticmethod
    def _dot(vector: Tensor, gradient: Tensor) -> Tensor:
        return (vector * gradient).sum(dim=1, keepdim=True)

    @staticmethod
    def _gradient(scalar: Tensor) -> Tensor:
        return tb.spatialGradient(
            scalar, dx_convention="pixel", boundary="periodic"
        )[:, 0]

    def _divergence(self, vector: Tensor) -> Tensor:
        gradient = tb.spatialGradient(
            vector,
            dx_convention=self.dx_convention,
            boundary="periodic",
        )
        return gradient[:, 0, 0, None] + gradient[:, 1, 1, None]

    def _advect(self, value: Tensor, source: Tensor, departure: Tensor) -> Tensor:
        return tb.imgDeform(
            value + self.dt * source,
            departure,
            dx_convention=self.dx_convention,
            clamp=False,
            boundary="periodic",
        )

    def _cometric(self, image: Tensor) -> CometricOperator:
        cometric = CometricOperator(
            image,
            self.rho,
            self.kernelOperator,
            dx_convention=self.dx_convention,
        )
        return cometric

    def _validate_inputs(self, image: Tensor, variables) -> SplinesVariables:
        if not isinstance(variables, SplinesVariables):
            raise TypeError(
                f"variables must be SplinesVariables, got {type(variables)}"
            )
        if (
            image.ndim != 4
            or image.shape[:2] != (1, 1)
            or min(image.shape[-2:]) < 2
        ):
            raise ValueError(
                "image must have shape [1, 1, H, W] with H,W >= 2, "
                f"got {tuple(image.shape)}"
            )
        if not torch.is_floating_point(image):
            raise TypeError("image must be floating point")
        if image.dtype not in (torch.float32, torch.float64):
            raise TypeError("image and spline variables must use float32 or float64")
        if variables.initial_momentum.shape != image.shape:
            raise ValueError("spline variable and image shapes must match")
        if variables.n_controls != len(self.control_times):
            raise ValueError(
                f"expected {len(self.control_times)} control jerks, "
                f"got {variables.n_controls}"
            )
        if (
            variables.initial_momentum.device != image.device
            or variables.initial_momentum.dtype != image.dtype
        ):
            raise ValueError("image and spline variables must share device and dtype")
        finite = torch.stack(
            [torch.isfinite(image).all()]
            + [torch.isfinite(value).all() for _, value in variables]
        )
        if not finite.all():
            raise ValueError("image and spline variables must contain only finite values")
        return variables

    @staticmethod
    def _check_finite(**state: Tensor) -> None:
        items = tuple(state.items())
        finite = torch.stack([torch.isfinite(tensor).all() for _, tensor in items])
        if finite.all():
            return
        for (name, _tensor), is_finite in zip(items, finite):
            if not is_finite:
                raise OverflowError(f"non-finite values in spline state '{name}'")

    def step(  # type: ignore[override]
        self,
        image: Tensor,
        momentum,
        *,
        force_x_0: Tensor | None = None,
    ):
        """Integrate the interior equations over one control-free interval."""
        momentum, acceleration, jerk = momentum
        gradient_image = None
        if self.rho == 0:
            force = acceleration
            kernel_momentum = image.new_zeros((1, 2) + image.shape[-2:])
            kernel_force = torch.zeros_like(kernel_momentum)
            kernel_linearization = torch.zeros_like(kernel_momentum)
        else:
            cometric = self._cometric(image)
            force = cometric.inverse(
                acceleration,
                eps=self.cg_eps,
                x_0=force_x_0,
            )
            gradient_image = cometric.image_gradient[:, 0]
            gradient_acceleration = self._gradient(acceleration)
            kernel_momentum, kernel_force, kernel_linearization = self.kernelOperator(
                torch.cat(
                    (
                        momentum * gradient_image,
                        force * gradient_image,
                        jerk * gradient_image + momentum * gradient_acceleration,
                    ),
                    dim=0,
                )
            ).split(1, dim=0)

        physical_velocity = -sqrt(self.rho) * kernel_momentum
        transport = sqrt(self.rho) * physical_velocity
        field = tb.im2grid(transport)
        divergence_transport = self._divergence(transport)

        if self.rho == 0:
            acceleration_coupling = torch.zeros_like(image)
            jerk_flux_divergence = torch.zeros_like(image)
        else:
            assert gradient_image is not None
            acceleration_coupling = self.rho * self._dot(
                kernel_linearization, gradient_image
            )
            jerk_flux_divergence = self.rho * self._divergence(
                force * kernel_force + momentum * kernel_linearization
            )

        image_source = (1 - self.rho) * momentum
        momentum_source = force - momentum * divergence_transport
        acceleration_source = (1 - self.rho) * jerk + acceleration_coupling
        jerk_source = jerk_flux_divergence - jerk * divergence_transport
        departure = self.id_grid - self.dt * field

        next_momentum, next_acceleration, next_jerk, next_image = self._advect(
            torch.cat((momentum, acceleration, jerk, image), dim=1),
            torch.cat(
                (
                    momentum_source,
                    acceleration_source,
                    jerk_source,
                    image_source,
                ),
                dim=1,
            ),
            departure,
        ).split(1, dim=1)
        next_state = (
            next_momentum,
            next_acceleration,
            next_jerk,
        )
        return (
            next_state,
            next_image,
            field,
            image_source,
            force,
            physical_velocity,
        )

    def forward(
        self,
        image: Tensor,
        momenta,
        save: bool = True,
        plot: int = 0,
        t_max: int = 1,
        verbose: bool = False,
        sharp=None,
        hamiltonian_integration: bool = False,
        progress_callback: Callable[[int, int], None] | None = None,
        retain_diagnostics: bool = False,
    ) -> None:
        """Integrate from 0 to 1 and optionally retain detached diagnostics.

        Saved node states include both endpoints. At a control node, ``jerk_stock``
        stores the right-limit value after the reset; the other three states are
        continuous. ``retain_diagnostics`` keeps one unstacked final record for
        later materialization without duplicating it into public stocks.
        """
        if plot:
            raise NotImplementedError("spline trajectory plotting is not implemented")
        if t_max != 1:
            raise ValueError("spline integration is defined on t in [0, 1]")
        if sharp:
            raise NotImplementedError("sharp spline integration is not implemented")
        if hamiltonian_integration:
            raise NotImplementedError(
                "use acceleration_energy for the spline regularization integral"
            )
        if progress_callback is not None and not callable(progress_callback):
            raise TypeError("progress_callback must be callable")
        if not isinstance(retain_diagnostics, bool):
            raise TypeError("retain_diagnostics must be a boolean")
        if save and retain_diagnostics:
            raise ValueError("save and retain_diagnostics are mutually exclusive")
        variables = self._validate_inputs(image, momenta)
        if hasattr(self.kernelOperator, "init_kernel"):
            self.kernelOperator.init_kernel(image)

        for name in (
            "trajectory",
            "_diagnostic_intervals",
            *self._DIAGNOSTIC_STOCK_NAMES,
        ):
            if hasattr(self, name):
                delattr(self, name)

        self.source = image.detach()
        self.initial_variables = variables
        self.image = image.clone()
        self.momentum = variables.initial_momentum.clone()
        self.acceleration = variables.initial_acceleration.clone()
        self.jerk = variables.initial_jerk.clone()
        id_grid = getattr(self, "id_grid", None)
        expected_grid_shape = (1, *image.shape[-2:], 2)
        if (
            not isinstance(id_grid, Tensor)
            or tuple(id_grid.shape) != expected_grid_shape
            or id_grid.device != image.device
            or id_grid.dtype != image.dtype
        ):
            self.id_grid = tb.make_regular_grid(
                image.shape[-2:],
                dx_convention=self.dx_convention,
                device=image.device,
            ).to(dtype=image.dtype)
        self.save = bool(save)
        self.acceleration_energy = image.new_zeros(())
        trajectory = None if self.save else [
            (self.image, self.momentum, self.acceleration, self.jerk)
        ]
        diagnostic_intervals = []

        images = []
        momenta = []
        accelerations = []
        jerks = []
        fields = []
        residuals = []
        forces = []
        velocities = []
        if self.save:
            images.append(self.image[0].detach().cpu())
            momenta.append(self.momentum[0].detach().cpu())
            accelerations.append(self.acceleration[0].detach().cpu())
            jerks.append(self.jerk[0].detach().cpu())

        force_x_0 = None
        for index in range(self.n_step):
            self._i = index
            old_acceleration = self.acceleration

            (
                next_state,
                self.image,
                self.field,
                self.residuals,
                self.force,
                self.velocity,
            ) = self.step(
                self.image,
                (self.momentum, self.acceleration, self.jerk),
                force_x_0=force_x_0,
            )
            self.momentum, self.acceleration, self.jerk = next_state
            self.acceleration_energy = self.acceleration_energy + (
                0.5 * self.dt * (self.force * old_acceleration).sum()
            )

            control_index = self._control_by_step.get(index + 1)
            if control_index is not None:
                self.jerk = variables.control_jerks[control_index]

            self._check_finite(
                image=self.image,
                momentum=self.momentum,
                acceleration=self.acceleration,
                jerk=self.jerk,
                acceleration_energy=self.acceleration_energy,
            )
            if self.rho > 0:
                force_x_0 = self.force.detach()
            if trajectory is not None:
                trajectory.append(
                    (self.image, self.momentum, self.acceleration, self.jerk)
                )
            if self.save:
                images.append(self.image[0].detach().cpu())
                momenta.append(self.momentum[0].detach().cpu())
                accelerations.append(self.acceleration[0].detach().cpu())
                jerks.append(self.jerk[0].detach().cpu())
                fields.append(self.field[0].detach().cpu())
                residuals.append(self.residuals[0].detach().cpu())
                forces.append(self.force[0].detach().cpu())
                velocities.append(self.velocity[0].detach().cpu())
            elif retain_diagnostics:
                diagnostic_intervals.append(
                    (self.field, self.residuals, self.force, self.velocity)
                )
            if progress_callback is not None:
                progress_callback(index + 1, self.n_step)
            if verbose:
                print(f"\rSpline integration {index + 1}/{self.n_step}", end="")

        if verbose:
            print()

        if self.save:
            self.time_stock = torch.linspace(0, 1, self.n_step + 1)
            self.image_stock = torch.stack(images)
            self.momentum_stock = torch.stack(momenta)
            self.acceleration_stock = torch.stack(accelerations)
            self.jerk_stock = torch.stack(jerks)
            self.field_stock = torch.stack(fields)
            self.residuals_stock = torch.stack(residuals)
            self.force_stock = torch.stack(forces)
            self.velocity_stock = torch.stack(velocities)
        else:
            assert trajectory is not None
            self.trajectory = tuple(trajectory)
            if retain_diagnostics:
                self._diagnostic_intervals = tuple(diagnostic_intervals)
            for name in self._DIAGNOSTIC_STOCK_NAMES:
                if hasattr(self, name):
                    delattr(self, name)

    def materialize_diagnostic_stocks(self) -> None:
        """Build detached CPU diagnostics and release their node trajectory."""
        if hasattr(self, "image_stock"):
            return
        trajectory = getattr(self, "trajectory", None)
        intervals = getattr(self, "_diagnostic_intervals", None)
        if trajectory is None or intervals is None:
            raise RuntimeError("no retained spline diagnostics are available")
        if len(trajectory) != self.n_step + 1 or len(intervals) != self.n_step:
            raise RuntimeError("retained spline diagnostics are incomplete")

        node_stocks = tuple(
            torch.stack(
                [state[index][0].detach().cpu() for state in trajectory]
            )
            for index in range(4)
        )
        interval_stocks = tuple(
            torch.stack(
                [state[index][0].detach().cpu() for state in intervals]
            )
            for index in range(4)
        )
        (
            self.image_stock,
            self.momentum_stock,
            self.acceleration_stock,
            self.jerk_stock,
        ) = node_stocks
        (
            self.field_stock,
            self.residuals_stock,
            self.force_stock,
            self.velocity_stock,
        ) = interval_stocks
        self.time_stock = torch.linspace(0, 1, self.n_step + 1)
        del self.trajectory
        del self._diagnostic_intervals
        self.save = True

    def to_device(self, device) -> None:
        super().to_device(device)
        self.kernelOperator.to(device)
        for name in (
            "source",
            "momentum",
            "acceleration",
            "jerk",
            "force",
            "velocity",
            "acceleration_energy",
        ):
            value = getattr(self, name, None)
            if isinstance(value, Tensor):
                setattr(self, name, value.to(device))
        initial_variables = getattr(self, "initial_variables", None)
        if isinstance(initial_variables, SplinesVariables):
            self.initial_variables = initial_variables.to(device)
        if hasattr(self, "trajectory"):
            self.trajectory = tuple(
                tuple(value.to(device) for value in state)
                for state in self.trajectory
            )
        if hasattr(self, "_diagnostic_intervals"):
            self._diagnostic_intervals = tuple(
                tuple(value.to(device) for value in state)
                for state in self._diagnostic_intervals
            )

    def plot(self, n_figs=5):
        raise NotImplementedError("spline trajectory plotting is not implemented")

    def get_all_arguments(self) -> dict:
        return {
            "rho": self.rho,
            "control_times": self.control_times,
            "kernelOperator": self.kernelOperator,
            "n_step": self.n_step,
            "cg_eps": self.cg_eps,
            "dx_convention": self.dx_convention,
        }


class MetamorphosisSplineOptimizer(Optimize_geodesicShooting):
    r"""Optimize a spline trajectory against images observed at fixed times.

    The implemented discretization of equation (37) is

    .. math::
        E(\theta) = \frac12\sum_i \|I(t_i)-J_i\|_2^2
          + \lambda E_{\mathrm{acc}},

    where ``theta`` is a :class:`SplinesVariables` instance and
    ``E_acc`` is accumulated by :class:`MetamorphosisSplineIntegrator`.
    This is equation (37) multiplied by ``sigma_I**2``, so ``cost_cst``
    corresponds to ``sigma_I**2``.
    """

    _OPTIMIZATION_CG_EPS = 1e-4

    @staticmethod
    def _validate_observations(
        source: Tensor,
        target: Tensor,
        target_times: Sequence[float] | Tensor,
        n_step: int,
    ) -> tuple[tuple[float, ...], tuple[int, ...]]:
        return validate_timed_observations(
            source, target, target_times, n_step
        )

    def __init__(
        self,
        source: Tensor,
        target: Tensor,
        target_times: Sequence[float] | Tensor,
        geodesic: MetamorphosisSplineIntegrator,
        cost_cst: float,
        optimizer_method: str = "LBFGS_torch",
        lbfgs_max_iter: int = 20,
        lbfgs_history_size: int = 100,
        hamiltonian_integration: bool = False,
        debug: bool = False,
    ) -> None:
        if not isinstance(geodesic, MetamorphosisSplineIntegrator):
            raise TypeError(
                "geodesic must be a MetamorphosisSplineIntegrator, "
                f"got {type(geodesic)}"
            )
        if optimizer_method != "LBFGS_torch":
            raise ValueError("spline optimization currently supports LBFGS_torch only")
        if hamiltonian_integration:
            raise NotImplementedError(
                "spline regularization uses the integrated acceleration energy"
            )
        try:
            cost_cst = float(cost_cst)
        except (TypeError, ValueError) as error:
            raise TypeError("cost_cst must be a finite non-negative scalar") from error
        if not isfinite(cost_cst) or cost_cst < 0:
            raise ValueError("cost_cst must be finite and non-negative")

        self.target_times, self.target_steps = self._validate_observations(
            source, target, target_times, geodesic.n_step
        )
        self.temporal_parameter_metric: Tensor | None = None
        self._temporal_transform: TemporalTransform | None = None
        source = source.detach()
        target = target.detach()
        super().__init__(
            source=source,
            target=target,
            geodesic=geodesic,
            cost_cst=cost_cst,
            data_term=TimedSsd(target, self.target_steps),
            optimizer_method=optimizer_method,
            lbfgs_max_iter=lbfgs_max_iter,
            lbfgs_history_size=lbfgs_history_size,
            hamiltonian_integration=False,
            debug=debug,
        )
        self._cost_saving_ = self._spline_cost_saving_

    @staticmethod
    def _stack_temporal_parameters(parameter: SplinesVariables) -> Tensor:
        return torch.cat(
            (
                parameter.initial_momentum.unsqueeze(0),
                parameter.initial_acceleration.unsqueeze(0),
                parameter.initial_jerk.unsqueeze(0),
                parameter.control_jerks,
            ),
            dim=0,
        )

    @staticmethod
    def _unstack_temporal_parameters(values: Tensor) -> SplinesVariables:
        return SplinesVariables(values[0], values[1], values[2], values[3:])

    def _apply_temporal_transform(
        self,
        parameter: SplinesVariables,
        *,
        inverse: bool = False,
    ) -> SplinesVariables:
        if self._temporal_transform is None:
            raise RuntimeError("the temporal block transform is not initialized")
        values = self._temporal_transform.apply(
            self._stack_temporal_parameters(parameter),
            inverse=inverse,
        )
        return self._unstack_temporal_parameters(values)

    def _prepare_optimization_parameter_(self, parameter):
        physical = super()._prepare_optimization_parameter_(parameter)
        if not isinstance(physical, SplinesVariables):
            raise TypeError("spline optimizer parameters must be SplinesVariables")
        active_indices = tuple(
            index
            for index, value in enumerate(
                (
                    physical.initial_momentum,
                    physical.initial_acceleration,
                    physical.initial_jerk,
                )
            )
            if value.requires_grad
        )
        if physical.control_jerks.requires_grad:
            active_indices += tuple(range(3, 3 + physical.n_controls))
        if not active_indices:
            return physical
        if self.temporal_parameter_metric is None:
            self.temporal_parameter_metric = cometric_temporal_metric(
                self.source,
                self.mp.rho,
                self.mp.kernelOperator,
                self.mp.cg_eps,
                self.mp.n_step,
                self.mp.control_steps,
                self.target_steps,
                self.cost_cst,
            )
        self._temporal_transform = TemporalTransform.from_metric(
            self.temporal_parameter_metric,
            active_indices,
        ).to(physical.initial_momentum)
        scaled = self._apply_temporal_transform(physical)
        return SplinesVariables(
            *(
                value.detach().clone().requires_grad_(
                    getattr(physical, name).requires_grad
                )
                for name, value in scaled
            )
        )

    def _parameter_for_cost_(self, parameter):
        if not isinstance(parameter, SplinesVariables):
            raise TypeError("spline optimizer parameters must be SplinesVariables")
        return self._apply_temporal_transform(parameter, inverse=True)

    def _optimization_cost_(self, parameter):
        physical = self._parameter_for_cost_(parameter)
        final_cg_eps = self.mp.cg_eps
        final_iteration = self._iter_ == self.n_iter - 1
        active_cg_eps = (
            final_cg_eps
            if final_iteration
            else max(self._OPTIMIZATION_CG_EPS, final_cg_eps)
        )
        retain_diagnostics = final_iteration and not any(
            value.requires_grad for _, value in physical if value.numel()
        )
        self.mp.cg_eps = active_cg_eps
        try:
            return self.cost(
                physical,
                retain_diagnostics=retain_diagnostics,
            )
        finally:
            self.mp.cg_eps = final_cg_eps

    def _finalize_integrator_(self, final_parameter):
        if not hasattr(self.mp, "_diagnostic_intervals"):
            self.mp.forward(
                self.source.clone(),
                final_parameter,
                save=False,
                plot=0,
                retain_diagnostics=True,
            )

    def _get_rho_(self) -> float:
        return float(self.mp.rho)

    def _dict_or_torch_parameter_(self) -> list[Tensor]:
        parameters = [
            value
            for _, value in self._optimization_parameter
            if value.requires_grad and value.numel()
        ]
        if len({id(parameter) for parameter in parameters}) != len(parameters):
            raise ValueError("optimizer parameter tensors must be distinct")
        if any(not parameter.is_leaf for parameter in parameters):
            raise ValueError("optimizer parameters must be leaf tensors")
        return parameters

    def to_device(self, device) -> None:
        super().to_device(device)
        if self._temporal_transform is not None:
            self._temporal_transform = self._temporal_transform.to(self.source)

    def _timed_observation_images(self, target_steps) -> Tensor:
        if tuple(target_steps) != self.target_steps:
            raise ValueError("timed SSD steps do not match spline observations")
        if hasattr(self.mp, "trajectory"):
            return torch.cat(
                [self.mp.trajectory[step][0] for step in target_steps],
                dim=0,
            )
        return self.mp.image_stock[list(target_steps)]

    def cost(
        self,
        variables: SplinesVariables,
        *,
        retain_diagnostics: bool = False,
        **kwargs,
    ) -> Tensor:
        for name in ("data_loss", "acceleration_energy", "total_cost"):
            if hasattr(self, name):
                delattr(self, name)
        self.mp.forward(
            self.source,
            variables,
            save=False,
            plot=0,
            retain_diagnostics=retain_diagnostics,
        )
        self.data_loss = self.data_term()
        self.acceleration_energy = self.mp.acceleration_energy
        self.total_cost = (
            self.data_loss + self.cost_cst * self.acceleration_energy
        )
        return self.total_cost

    def _spline_cost_saving_(self, index: int, loss_stock):
        if loss_stock is None:
            return {
                "data_loss": torch.zeros(index),
                "acceleration_energy": torch.zeros(index),
                "total_cost": torch.zeros(index),
            }
        self.data_loss = self.data_loss.detach()
        self.acceleration_energy = self.acceleration_energy.detach()
        self.total_cost = self.total_cost.detach()
        components = torch.stack(
            (self.data_loss, self.acceleration_energy, self.total_cost)
        ).cpu()
        loss_stock["data_loss"][index] = components[0]
        loss_stock["acceleration_energy"][index] = components[1]
        loss_stock["total_cost"][index] = components[2]
        return loss_stock

    def _get_loss_components(self):
        if not isinstance(self.loss_stock, dict):
            raise ValueError("Loss history is not available.")
        return (
            self.loss_stock["data_loss"],
            self.loss_stock["acceleration_energy"],
            None,
        )

    @property
    def optimized_variables(self) -> SplinesVariables | None:
        if isinstance(self.optimized_momenta, SplinesVariables):
            return self.optimized_momenta
        return None

    def get_geodesic_distance(self, only_zero=False):
        raise NotImplementedError(
            "geodesic distance is not defined for acceleration-regularized splines"
        )

    def get_all_arguments(self) -> dict:
        return {
            **super().get_all_arguments(),
            "rho": self._get_rho_(),
            "target_times": self.target_times,
            "control_times": self.mp.control_times,
            "cg_eps": self.mp.cg_eps,
            "lbfgs_max_iter": self.lbfgs_max_iter,
            "lbfgs_history_size": self.lbfgs_history_size,
        }
