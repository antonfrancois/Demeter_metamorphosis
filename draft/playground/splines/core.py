"""Numerical and persistence layer for the spline playground.

Version: July 23, 2026.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field as dataclass_field
from math import isfinite, sqrt
from operator import le, lt
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
import torch.nn.functional as F

from demeter.metamorphosis.classic import Metamorphosis_integrator
from demeter.metamorphosis.splines import (
    MetamorphosisSplineIntegrator,
    SplinesVariables,
)
from demeter.metamorphosis.var_classes import Momenta
from demeter.utils import torchbox as tb
from demeter.utils.cometric_inversion import CometricOperator
from demeter.utils.reproducing_kernels import GaussianRKHS, SobolevFluidOperator
from demeter.utils.spline_data import TimedImageBatch
from ..field_playground_core import (
    coerce_field,
    coerce_image,
    load_field_file,
    resize_field,
)


TRAJECTORY_FIELDS = (
    "momentum",
    "force",
    "acceleration",
    "jerk",
    "velocity",
    "vector_momentum",
)
OPTIMIZABLE_INITIAL_FIELDS = (
    "initial_momentum",
    "initial_acceleration",
    "initial_jerk",
)


def minimum_mesh_steps(
    times: tuple[float, ...],
    *,
    max_steps: int,
) -> int:
    """Return the smallest temporal mesh containing every normalized time."""
    for n_steps in range(1, max_steps + 1):
        if all(abs(time * n_steps - round(time * n_steps)) <= 1e-6 for time in times):
            return n_steps
    raise ValueError(
        f"no temporal mesh with at most {max_steps} steps contains all times"
    )


def resolve_device(device: str | torch.device | None = "auto") -> torch.device:
    """Resolve ``auto`` to CUDA when available and CPU otherwise."""
    if device is None or str(device).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _validate_progress_callback(callback: Callable[[int, int], None] | None) -> None:
    if callback is not None and not callable(callback):
        raise TypeError("progress_callback must be callable")


def _synchronize_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _require_finite_input(value: Any, name: str) -> None:
    tensor = torch.as_tensor(value)
    if (
        torch.is_floating_point(tensor) or torch.is_complex(tensor)
    ) and not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must contain only finite values")


def _finite_float(value: Any, name: str) -> float:
    value = float(value)
    if not isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _positive_float(value: Any, name: str) -> float:
    value = float(value)
    if not isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and strictly positive")
    return value


def _positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a strictly positive integer")
    return value


def _choice(value: Any, name: str, choices: tuple[str, str]) -> str:
    value = str(value).lower()
    if value not in choices:
        raise ValueError(f"{name} must be {choices[0]!r} or {choices[1]!r}")
    return value


def _strictly_increasing(values: tuple[int, ...] | tuple[float, ...]) -> bool:
    return all(left < right for left, right in zip(values, values[1:]))


def _control_nodes(times: tuple[float, ...], steps: int) -> tuple[int, ...]:
    nodes = tuple(round(time * steps) for time in times)
    if any(not 1 <= node < steps - 1 for node in nodes):
        raise ValueError("control times must be before the final interior mesh node")
    if not _strictly_increasing(nodes):
        raise ValueError("control times must map to distinct ordered mesh nodes")
    return nodes


def _control_times(values: tuple[float, ...], steps: int) -> tuple[float, ...]:
    times = tuple(float(time) for time in values)
    if any(not isfinite(time) or not 0 < time < 1 for time in times):
        raise ValueError("control_times must be finite and lie strictly in (0, 1)")
    if not _strictly_increasing(times):
        raise ValueError("control_times must be strictly increasing")
    _control_nodes(times, steps)
    return times


def _optimized_fields(values: tuple[str, ...]) -> tuple[str, ...]:
    selected = set(values)
    if not selected <= set(OPTIMIZABLE_INITIAL_FIELDS):
        raise ValueError("optimized_fields contains an unknown field")
    return tuple(name for name in OPTIMIZABLE_INITIAL_FIELDS if name in selected)


def _validate_model(parameters: "SplineParameters") -> None:
    if min(parameters.alpha, parameters.beta) < 0 or parameters.gamma <= 0:
        raise ValueError(
            "alpha and beta must be non-negative, and gamma must be positive"
        )
    compare, symbol = {"classic": (le, "<="), "splines": (lt, "<")}[parameters.model]
    if parameters.rho < 0 or not compare(parameters.rho, 1):
        raise ValueError(f"rho must satisfy 0 <= rho {symbol} 1")
    kernels = {"classic": ("sobolev", "gaussian"), "splines": ("sobolev",)}
    if parameters.kernel not in kernels[parameters.model]:
        raise ValueError("spline runs require the Sobolev operator")


@dataclass(frozen=True)
class SolverSettings:
    """One optimization mesh and its LBFGS controls."""

    cost: float = 0.01
    steps: int = 16
    iterations: int = 10
    learning_rate: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "cost", _positive_float(self.cost, "cost"))
        object.__setattr__(self, "steps", _positive_int(self.steps, "steps"))
        object.__setattr__(
            self, "iterations", _positive_int(self.iterations, "iterations")
        )
        object.__setattr__(
            self,
            "learning_rate",
            _positive_float(self.learning_rate, "learning_rate"),
        )


@dataclass(frozen=True)
class SplineParameters:
    """Physical model, control times, and optimization settings."""

    alpha: float = 0.2
    beta: float = 0.2
    gamma: float = 0.001
    rho: float = 0.5
    cg_tolerance: float = 1e-5
    kernel: str = "sobolev"
    sigma: float = 3.0
    model: str = "splines"
    control_times: tuple[float, ...] = ()
    optimized_fields: tuple[str, ...] = OPTIMIZABLE_INITIAL_FIELDS
    initialization: str = "cold"
    spline: SolverSettings = dataclass_field(default_factory=SolverSettings)
    regression: SolverSettings = dataclass_field(default_factory=SolverSettings)

    def __post_init__(self) -> None:
        for name in ("alpha", "beta", "gamma", "rho"):
            object.__setattr__(self, name, _finite_float(getattr(self, name), name))
        normalized = {
            "cg_tolerance": _positive_float(self.cg_tolerance, "cg_tolerance"),
            "sigma": _positive_float(self.sigma, "sigma"),
            "model": _choice(self.model, "model", ("classic", "splines")),
            "kernel": _choice(self.kernel, "kernel", ("sobolev", "gaussian")),
            "initialization": _choice(
                self.initialization, "initialization", ("cold", "warm")
            ),
            "control_times": _control_times(self.control_times, self.spline.steps),
            "optimized_fields": _optimized_fields(self.optimized_fields),
        }
        for name, value in normalized.items():
            object.__setattr__(self, name, value)
        _validate_model(self)

    @property
    def control_nodes(self) -> tuple[int, ...]:
        return _control_nodes(self.control_times, self.spline.steps)

    @property
    def projected_control_times(self) -> tuple[float, ...]:
        return tuple(node / self.spline.steps for node in self.control_nodes)


def _scalar_field(
    value: Any,
    size: tuple[int, int],
    *,
    dtype: torch.dtype,
    name: str,
) -> torch.Tensor:
    field, mode = coerce_field(value)
    if mode != "scalar" or field.ndim != 4 or field.shape[:2] != (1, 1):
        raise ValueError(f"{name} must have shape [1, 1, H, W]")
    field = resize_field(
        field,
        size,
        scale_vector_displacement=False,
    )
    field = field.to(dtype=dtype).contiguous().clone()
    if not torch.isfinite(field).all():
        raise ValueError(f"{name} must contain only finite values")
    return field


def _validate_target_mesh(
    times: tuple[float, ...],
    n_steps: int,
    name: str,
) -> None:
    target_steps = tuple(round(time * n_steps) for time in times)
    if any(not 1 <= step <= n_steps for step in target_steps):
        raise ValueError(f"target times must map to nonzero {name} temporal nodes")
    if any(
        abs(time * n_steps - step) > 1e-6 for time, step in zip(times, target_steps)
    ):
        raise ValueError(f"target times must lie on the {name} mesh")
    if not _strictly_increasing(target_steps):
        raise ValueError(f"target times must map to distinct {name} temporal nodes")


@dataclass
class SplineSetup:
    """Canonical images, shooting variables, and numerical parameters."""

    images: TimedImageBatch
    variables: SplinesVariables
    parameters: SplineParameters

    def __post_init__(self) -> None:
        if self.variables.initial_momentum.shape != self.images.source.shape:
            raise ValueError("shooting fields must match the source image")
        if self.variables.n_controls != len(self.parameters.control_times):
            raise ValueError("control fields and control times must have equal length")
        _validate_target_mesh(
            self.images.target_times,
            self.parameters.spline.steps,
            "spline",
        )
        if self.parameters.initialization == "warm":
            _validate_target_mesh(
                self.images.target_times,
                self.parameters.regression.steps,
                "regression",
            )

    @property
    def size(self) -> tuple[int, int]:
        return tuple(self.images.source.shape[-2:])

    @property
    def n_controls(self) -> int:
        return self.variables.n_controls

    @property
    def target_steps(self) -> tuple[int, ...]:
        return tuple(
            round(time * self.parameters.spline.steps)
            for time in self.images.target_times
        )


def zero_setup(
    source: Any,
    target: Any | None = None,
    parameters: SplineParameters | None = None,
    *,
    source_path: str | Path | None = None,
    target_times: tuple[float, ...] = (1.0,),
    target_paths: tuple[str | Path, ...] = (),
) -> SplineSetup:
    """Create a zero-field setup matching the source image."""
    parameters = parameters or SplineParameters()
    _require_finite_input(source, "source")
    source_tensor = coerce_image(source).detach().cpu().contiguous().clone()
    if target is None:
        target = torch.zeros_like(source_tensor)
    target_tensor = coerce_image(target).detach().cpu().clone()
    if tuple(target_tensor.shape[-2:]) != tuple(source_tensor.shape[-2:]):
        target_tensor = F.interpolate(
            target_tensor,
            size=source_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    paths = tuple(str(path) for path in target_paths) or ("",) * len(target_tensor)
    return SplineSetup(
        images=TimedImageBatch(
            source_tensor,
            target_tensor.to(source_tensor).contiguous(),
            target_times,
            str(source_path or ""),
            paths,
        ),
        variables=SplinesVariables.zeros(
            source_tensor,
            len(parameters.control_times),
            requires_grad=False,
        ),
        parameters=parameters,
    )


def save_setup(setup: SplineSetup, path: str | Path) -> Path:
    path = Path(path).expanduser()
    if not path.suffix:
        path = path.with_suffix(".pt")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(setup, path)
    return path


def load_setup(path: str | Path) -> SplineSetup:
    path = Path(path).expanduser()
    setup = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(setup, SplineSetup):
        raise ValueError(f"{path} is not a spline playground setup")
    return setup


def load_scalar_field(
    path: str | Path,
    size: tuple[int, int],
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Load one scalar field using the formats accepted by the field playground."""
    loaded = load_field_file(path)
    return _scalar_field(
        loaded.field,
        size,
        dtype=dtype,
        name="loaded field",
    )


def cometric_squared_norm(
    image: torch.Tensor,
    covector: torch.Tensor,
    parameters: SplineParameters,
) -> float:
    """Return ``<covector, A_image covector>`` on the tensor's device."""
    covector = covector.to(device=image.device, dtype=image.dtype)
    kernel = _kernel_operator(parameters)
    with torch.no_grad():
        acceleration = CometricOperator(
            image,
            parameters.rho,
            kernel,
            dx_convention="pixel",
        )(covector)
    return float((covector * acceleration).sum().detach().cpu())


def metric_squared_norm(
    image: torch.Tensor,
    vector: torch.Tensor,
    parameters: SplineParameters,
) -> float:
    """Return ``<vector, A_image^-1 vector>`` on the tensor's device."""
    if parameters.rho == 1:
        return float("nan")
    vector = vector.to(device=image.device, dtype=image.dtype)
    kernel = _kernel_operator(parameters)
    with torch.no_grad():
        covector = CometricOperator(
            image,
            parameters.rho,
            kernel,
            dx_convention="pixel",
        ).inverse(vector, eps=parameters.cg_tolerance)
    return float((vector * covector).sum().detach().cpu())


def _kernel_operator(parameters: SplineParameters):
    if parameters.kernel == "gaussian":
        return GaussianRKHS(
            (parameters.sigma, parameters.sigma),
            border_type="circular",
            normalized=False,
            kernel_reach=3,
        )
    return SobolevFluidOperator(
        alpha=parameters.alpha,
        beta=parameters.beta,
        gamma=parameters.gamma,
        boundary="periodic",
    )


@dataclass(frozen=True)
class SplineTrajectory:
    """Detached, CPU-resident, node-aligned trajectory for interactive viewing."""

    images: torch.Tensor
    deformed_source: torch.Tensor
    photometric_only: torch.Tensor
    momentum: torch.Tensor
    force: torch.Tensor
    acceleration: torch.Tensor
    jerk: torch.Tensor
    velocity: torch.Tensor
    vector_momentum: torch.Tensor
    field_energies: dict[str, torch.Tensor]
    target_mse: torch.Tensor
    elapsed_seconds: float

    def field(self, name: str) -> torch.Tensor:
        if name not in TRAJECTORY_FIELDS:
            raise KeyError(f"unknown trajectory field {name!r}")
        return getattr(self, name)

    def field_energy(self, name: str, index: int) -> float:
        return float(self.field_energies[name][index])


def _decompose_image_nodes(
    source: torch.Tensor,
    fields: torch.Tensor,
    residuals: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Replay deformation-only and photometric-only counterfactual images."""
    n_steps = fields.shape[0]
    identity = tb.make_regular_grid(
        source.shape[-2:],
        dx_convention="pixel",
        device=source.device,
    ).to(dtype=source.dtype)
    deformed_source = source.clone()
    photometric_only = source.clone()
    deformed_nodes = [deformed_source[0].clone()]
    photometric_nodes = [photometric_only[0].clone()]
    dt = 1 / n_steps
    for field, residual in zip(fields, residuals):
        departure = identity - dt * field[None]
        deformed_source = tb.imgDeform(
            deformed_source,
            departure,
            dx_convention="pixel",
            clamp=False,
            boundary="periodic",
        )
        photometric_only = photometric_only + dt * residual[None]
        deformed_nodes.append(deformed_source[0].clone())
        photometric_nodes.append(photometric_only[0].clone())
    return (
        torch.stack(deformed_nodes).contiguous(),
        torch.stack(photometric_nodes).contiguous(),
    )


def target_mse(images: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        [(images - target).square().mean(dim=(1, 2, 3)) for target in targets]
    )


def _scalar_field_energies(
    images: torch.Tensor,
    momentum: torch.Tensor,
    force: torch.Tensor,
    acceleration: torch.Tensor,
    jerk: torch.Tensor,
    parameters: SplineParameters,
) -> dict[str, torch.Tensor]:
    kernel = _kernel_operator(parameters)
    energies = {name: [] for name in ("momentum", "force", "acceleration", "jerk")}
    with torch.no_grad():
        for image, p, u, a, r in zip(images, momentum, force, acceleration, jerk):
            cometric = CometricOperator(
                image[None],
                parameters.rho,
                kernel,
                dx_convention="pixel",
            )
            p_energy = (p * cometric(p[None])[0]).sum()
            if parameters.rho == 1:
                a_energy = image.new_tensor(float("nan"))
                u_energy = (u * cometric(u[None])[0]).sum()
            else:
                u_energy = a_energy = (u * a).sum()
            r_energy = (r * cometric(r[None])[0]).sum()
            for name, value in (
                ("momentum", p_energy),
                ("force", u_energy),
                ("acceleration", a_energy),
                ("jerk", r_energy),
            ):
                energies[name].append(value)
    return {name: torch.stack(values) for name, values in energies.items()}


def _endpoint_fields(
    integrator: MetamorphosisSplineIntegrator,
    kernel: SobolevFluidOperator,
    parameters: SplineParameters,
) -> tuple[torch.Tensor, torch.Tensor]:
    image = integrator.image
    acceleration = integrator.acceleration
    if parameters.rho == 0:
        force = acceleration
        velocity = image.new_zeros((1, 2) + image.shape[-2:])
        return force, velocity

    force = CometricOperator(
        image,
        parameters.rho,
        kernel,
        dx_convention="pixel",
    ).inverse(acceleration, eps=parameters.cg_tolerance)
    gradient = tb.spatialGradient(
        image,
        dx_convention="pixel",
        boundary="periodic",
    )[:, 0]
    velocity = -sqrt(parameters.rho) * kernel(integrator.momentum * gradient)
    return force, velocity


def run_spline(
    setup: SplineSetup,
    *,
    device: str | torch.device = "auto",
    progress_callback: Callable[[int, int], None] | None = None,
) -> SplineTrajectory:
    """Run the forward spline and return only detached node-aligned data."""
    _validate_progress_callback(progress_callback)
    parameters = setup.parameters
    if parameters.kernel != "sobolev":
        raise ValueError(
            "spline integration requires the Sobolev operator; Gaussian "
            "cometric inversion is not defined"
        )
    run_device = resolve_device(device)
    source = setup.images.source.to(run_device)
    kernel = SobolevFluidOperator(
        alpha=parameters.alpha,
        beta=parameters.beta,
        gamma=parameters.gamma,
        boundary="periodic",
    )
    integrator_progress: Callable[[int, int], None] | None = None
    if progress_callback is not None:

        def report_integration_progress(completed: int, total: int) -> None:
            if completed < total:
                progress_callback(completed, total)

        integrator_progress = report_integration_progress

    _synchronize_cuda(run_device)
    start = perf_counter()
    with torch.no_grad():
        variables = setup.variables.clone().to(run_device)
        integrator = MetamorphosisSplineIntegrator(
            parameters.rho,
            control_times=parameters.projected_control_times,
            kernelOperator=kernel,
            n_step=parameters.spline.steps,
            cg_eps=parameters.cg_tolerance,
            dx_convention="pixel",
        )
        integrator(
            source,
            variables,
            save=True,
            progress_callback=integrator_progress,
        )
    return _trajectory_from_final_spline_integration(
        integrator,
        parameters,
        setup.images.target,
        progress_callback=progress_callback,
        started_at=start,
    )


def _trajectory_from_final_spline_integration(
    integrator: MetamorphosisSplineIntegrator,
    parameters: SplineParameters,
    targets: torch.Tensor,
    progress_callback: Callable[[int, int], None] | None = None,
    *,
    started_at: float | None = None,
) -> SplineTrajectory:
    """Build playground data from an optimizer's retained final integration."""
    start = perf_counter() if started_at is None else started_at
    integrator.materialize_diagnostic_stocks()
    kernel = integrator.kernelOperator
    device = integrator.source.device
    with torch.no_grad():
        endpoint_force, endpoint_velocity = _endpoint_fields(
            integrator,
            kernel,
            parameters,
        )
        deformed_source, photometric_only = _decompose_image_nodes(
            integrator.source,
            integrator.field_stock.to(device),
            integrator.residuals_stock.to(device),
        )
    _synchronize_cuda(device)

    images = integrator.image_stock.detach().cpu().contiguous()
    force = torch.cat(
        (integrator.force_stock, endpoint_force.detach().cpu()),
        dim=0,
    ).contiguous()
    velocity = torch.cat(
        (integrator.velocity_stock, endpoint_velocity.detach().cpu()),
        dim=0,
    ).contiguous()
    momentum = integrator.momentum_stock.detach().cpu().contiguous()
    acceleration = integrator.acceleration_stock.detach().cpu().contiguous()
    jerk = integrator.jerk_stock.detach().cpu().contiguous()
    vector_momentum = kernel.apply_operator(velocity).contiguous()
    velocity_energy = (velocity * vector_momentum).sum(dim=(1, 2, 3))
    field_energies = _scalar_field_energies(
        images,
        momentum,
        force,
        acceleration,
        jerk,
        parameters,
    ) | {
        "velocity": velocity_energy,
        "vector_momentum": velocity_energy,
    }
    target_errors = target_mse(images, targets.to(dtype=images.dtype))
    if progress_callback is not None:
        progress_callback(parameters.spline.steps, parameters.spline.steps)
    return SplineTrajectory(
        images=images,
        deformed_source=deformed_source.detach().cpu().contiguous(),
        photometric_only=photometric_only.detach().cpu().contiguous(),
        momentum=momentum,
        force=force,
        acceleration=acceleration,
        jerk=jerk,
        velocity=velocity,
        vector_momentum=vector_momentum,
        field_energies=field_energies,
        target_mse=target_errors,
        elapsed_seconds=perf_counter() - start,
    )


def run_classic(
    setup: SplineSetup,
    *,
    device: str | torch.device = "auto",
    progress_callback: Callable[[int, int], None] | None = None,
) -> SplineTrajectory:
    """Run classic shooting from a setup containing only initial momentum."""
    _validate_progress_callback(progress_callback)
    unsupported = [
        name
        for name, field in (
            ("initial acceleration", setup.variables.initial_acceleration),
            ("initial jerk", setup.variables.initial_jerk),
            ("control jerk", setup.variables.control_jerks),
        )
        if bool(torch.count_nonzero(field))
    ]
    if unsupported:
        raise ValueError(
            "classic metamorphosis accepts only initial momentum; clear "
            + ", ".join(unsupported)
        )

    parameters = setup.parameters
    run_device = resolve_device(device)
    source = setup.images.source.to(run_device)
    initial_momentum = setup.variables.initial_momentum.to(run_device)
    kernel = _kernel_operator(parameters)
    integrator = Metamorphosis_integrator(
        method="semiLagrangian",
        rho=parameters.rho,
        kernelOperator=kernel,
        n_step=parameters.spline.steps,
        dx_convention="pixel",
        boundary="periodic",
    )

    _synchronize_cuda(run_device)
    start = perf_counter()
    with torch.no_grad():
        integrator(
            source,
            Momenta(momentum_I=initial_momentum),
            save=True,
            progress_callback=progress_callback,
        )
        images = torch.cat(
            (source.detach().cpu(), integrator.image_stock),
            dim=0,
        ).contiguous()
        momentum = torch.cat(
            (
                initial_momentum.detach().cpu(),
                torch.cat(
                    [state.momentum_I for state in integrator.momentum_stock],
                    dim=0,
                ),
            ),
            dim=0,
        ).contiguous()
        deformed_source_device, photometric_only_device = _decompose_image_nodes(
            source,
            integrator.field_stock.to(run_device),
            ((1 - parameters.rho) * momentum[:-1]).to(run_device),
        )
        if parameters.rho == 0:
            velocity_device = source.new_zeros(
                (parameters.spline.steps + 1, 2) + tuple(source.shape[-2:])
            )
            vector_momentum_device = torch.zeros_like(velocity_device)
        else:
            image_nodes = images.to(run_device)
            momentum_nodes = momentum.to(run_device)
            gradient = tb.spatialGradient(
                image_nodes,
                dx_convention="pixel",
                boundary="periodic",
            )[:, 0]
            vector_momentum_device = -sqrt(parameters.rho) * (momentum_nodes * gradient)
            velocity_device = kernel(vector_momentum_device)
    _synchronize_cuda(run_device)

    deformed_source = deformed_source_device.detach().cpu().contiguous()
    photometric_only = photometric_only_device.detach().cpu().contiguous()
    velocity = velocity_device.detach().cpu().contiguous()
    vector_momentum = vector_momentum_device.detach().cpu().contiguous()
    zero = torch.zeros_like(momentum)
    velocity_energy = (velocity * vector_momentum).sum(dim=(1, 2, 3))
    field_energies = _scalar_field_energies(
        images,
        momentum,
        zero,
        zero,
        zero,
        parameters,
    ) | {
        "velocity": velocity_energy,
        "vector_momentum": velocity_energy,
    }
    targets = setup.images.target.to(dtype=images.dtype)
    target_errors = target_mse(images, targets)
    elapsed = perf_counter() - start
    return SplineTrajectory(
        images=images,
        deformed_source=deformed_source,
        photometric_only=photometric_only,
        momentum=momentum,
        force=zero,
        acceleration=zero.clone(),
        jerk=zero.clone(),
        velocity=velocity,
        vector_momentum=vector_momentum,
        field_energies=field_energies,
        target_mse=target_errors,
        elapsed_seconds=elapsed,
    )
