"""Numerical and persistence layer for the spline playground.

Version: July 23, 2026.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite, sqrt
import os
from pathlib import Path
import tempfile
from time import perf_counter
from typing import Any

import torch
import torch.nn.functional as F

from demeter.metamorphosis.splines import (
    MetamorphosisSplineIntegrator,
    SplinesVariables,
)
from demeter.utils import torchbox as tb
from demeter.utils.cometric_inversion import CometricOperator
from demeter.utils.reproducing_kernels import SobolevFluidOperator
from ..field_playground_core import (
    coerce_field,
    coerce_image,
    load_field_file,
    resize_field,
)


FORMAT_VERSION = 1
SETUP_KIND = "demeter_spline_playground"
TRAJECTORY_FIELDS = (
    "momentum",
    "force",
    "acceleration",
    "jerk",
    "velocity",
    "vector_momentum",
)


def resolve_device(device: str | torch.device | None = "auto") -> torch.device:
    """Resolve ``auto`` to CUDA when available and CPU otherwise."""
    if device is None or str(device).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _require_finite_input(value: Any, name: str) -> None:
    tensor = torch.as_tensor(value)
    if (torch.is_floating_point(tensor) or torch.is_complex(tensor)) and not torch.isfinite(
        tensor
    ).all():
        raise ValueError(f"{name} must contain only finite values")


@dataclass(frozen=True)
class SplineParameters:
    """Numerical parameters and control-node topology for one spline run."""

    alpha: float = 0.2
    beta: float = 0.2
    gamma: float = 0.001
    rho: float = 0.5
    cg_eps: float = 1e-5
    n_steps: int = 16
    control_steps: tuple[int, ...] = ()
    control_times: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        for name in ("alpha", "beta", "gamma", "rho", "cg_eps"):
            value = float(getattr(self, name))
            if not isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        if self.alpha < 0 or self.beta < 0 or self.gamma <= 0:
            raise ValueError(
                "alpha and beta must be non-negative, and gamma must be positive"
            )
        if not 0 <= self.rho < 1:
            raise ValueError("rho must satisfy 0 <= rho < 1")
        if self.cg_eps <= 0:
            raise ValueError("cg_eps must be strictly positive")
        if (
            not isinstance(self.n_steps, int)
            or isinstance(self.n_steps, bool)
            or self.n_steps < 1
        ):
            raise ValueError("n_steps must be a strictly positive integer")

        if not self.control_times and self.control_steps:
            controls = tuple(self.control_steps)
            if any(
                not isinstance(step, int) or isinstance(step, bool)
                for step in controls
            ):
                raise TypeError("control_steps must contain integers")
            if any(not 1 <= step < self.n_steps for step in controls):
                raise ValueError("control_steps must be interior temporal mesh nodes")
            if any(right <= left for left, right in zip(controls, controls[1:])):
                raise ValueError("control_steps must be strictly increasing")
            times = tuple(step / self.n_steps for step in controls)
        else:
            times = tuple(float(time) for time in self.control_times)
            if any(not isfinite(time) or not 0 < time < 1 for time in times):
                raise ValueError(
                    "control_times must be finite and lie strictly in (0, 1)"
                )
            if any(right <= left for left, right in zip(times, times[1:])):
                raise ValueError("control_times must be strictly increasing")
            controls = self.project_control_times(times, self.n_steps)
        object.__setattr__(self, "control_steps", controls)
        object.__setattr__(self, "control_times", times)

    @staticmethod
    def project_control_times(
        control_times: tuple[float, ...],
        n_steps: int,
    ) -> tuple[int, ...]:
        controls = tuple(round(time * n_steps) for time in control_times)
        if any(not 1 <= step < n_steps for step in controls):
            raise ValueError(
                "control times must project to interior temporal mesh nodes"
            )
        if any(right <= left for left, right in zip(controls, controls[1:])):
            raise ValueError(
                "control times must project to distinct temporal mesh nodes"
            )
        return controls

    @property
    def mesh_control_times(self) -> tuple[float, ...]:
        return tuple(step / self.n_steps for step in self.control_steps)

    def as_dict(self) -> dict[str, Any]:
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
            "rho": self.rho,
            "cg_eps": self.cg_eps,
            "n_steps": self.n_steps,
            "control_steps": self.control_steps,
            "control_times": self.control_times,
        }

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "SplineParameters":
        return cls(
            alpha=values.get("alpha", 0.2),
            beta=values.get("beta", 0.2),
            gamma=values.get("gamma", 0.001),
            rho=values.get("rho", 0.5),
            cg_eps=values.get("cg_eps", 1e-5),
            n_steps=values.get("n_steps", 16),
            control_steps=tuple(values.get("control_steps", ())),
            control_times=(
                tuple(values["control_times"])
                if "control_times" in values
                else ()
            ),
        )


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


@dataclass
class SplineSetup:
    """Editable source, target, and shooting fields stored on CPU."""

    source: torch.Tensor
    target: torch.Tensor
    initial_momentum: torch.Tensor
    initial_force: torch.Tensor
    initial_jerk: torch.Tensor
    control_jerks: torch.Tensor
    parameters: SplineParameters
    source_path: str = ""
    target_path: str = ""

    def __post_init__(self) -> None:
        _require_finite_input(self.source, "source")
        _require_finite_input(self.target, "target")
        source = coerce_image(self.source)
        if (
            source.ndim != 4
            or source.shape[:2] != (1, 1)
            or min(source.shape[-2:]) < 2
        ):
            raise ValueError(
                "source must have shape [1, 1, H, W] with H,W >= 2"
            )
        source = source.contiguous().clone()
        size = tuple(source.shape[-2:])
        dtype = source.dtype

        target = coerce_image(self.target)
        if target.ndim != 4 or target.shape[:2] != (1, 1):
            raise ValueError("target must have shape [1, 1, H, W]")
        if tuple(target.shape[-2:]) != size:
            target = F.interpolate(
                target,
                size=size,
                mode="bilinear",
                align_corners=False,
            )

        self.source = source
        self.target = target.to(dtype=dtype).contiguous().clone()
        self.initial_momentum = _scalar_field(
            self.initial_momentum,
            size,
            dtype=dtype,
            name="initial_momentum",
        )
        self.initial_force = _scalar_field(
            self.initial_force,
            size,
            dtype=dtype,
            name="initial_force",
        )
        self.initial_jerk = _scalar_field(
            self.initial_jerk,
            size,
            dtype=dtype,
            name="initial_jerk",
        )

        controls = torch.as_tensor(self.control_jerks).detach().cpu()
        if controls.dtype != torch.float64:
            controls = controls.float()
        expected = (len(self.parameters.control_steps), 1, 1) + size
        if tuple(controls.shape) != expected:
            raise ValueError(
                f"control_jerks must have shape {expected}, got {tuple(controls.shape)}"
            )
        self.control_jerks = controls.to(dtype=dtype).contiguous().clone()
        for name, tensor in (
            ("source", self.source),
            ("target", self.target),
            ("control_jerks", self.control_jerks),
        ):
            if not torch.isfinite(tensor).all():
                raise ValueError(f"{name} must contain only finite values")
        self.source_path = str(self.source_path)
        self.target_path = str(self.target_path)

    @property
    def size(self) -> tuple[int, int]:
        return tuple(self.source.shape[-2:])

    @property
    def n_controls(self) -> int:
        return len(self.parameters.control_steps)

    def payload(self) -> dict[str, Any]:
        return {
            "format_version": FORMAT_VERSION,
            "kind": SETUP_KIND,
            "source": self.source.detach().cpu().clone(),
            "target": self.target.detach().cpu().clone(),
            "initial_momentum": self.initial_momentum.detach().cpu().clone(),
            "initial_force": self.initial_force.detach().cpu().clone(),
            "initial_jerk": self.initial_jerk.detach().cpu().clone(),
            "control_jerks": self.control_jerks.detach().cpu().clone(),
            "parameters": self.parameters.as_dict(),
            "source_path": self.source_path,
            "target_path": self.target_path,
        }


def zero_setup(
    source: Any,
    target: Any | None = None,
    parameters: SplineParameters | None = None,
    *,
    source_path: str | Path | None = None,
    target_path: str | Path | None = None,
) -> SplineSetup:
    """Create a zero-field setup matching the source image."""
    parameters = parameters or SplineParameters()
    _require_finite_input(source, "source")
    source_tensor = coerce_image(source)
    if target is None:
        target = torch.zeros_like(source_tensor)
    zero = torch.zeros_like(source_tensor)
    controls = source_tensor.new_zeros(
        (len(parameters.control_steps),) + tuple(source_tensor.shape)
    )
    return SplineSetup(
        source_tensor,
        target,
        zero,
        zero.clone(),
        zero.clone(),
        controls,
        parameters,
        str(source_path or ""),
        str(target_path or ""),
    )


def save_setup(setup: SplineSetup, path: str | Path) -> Path:
    path = Path(path).expanduser()
    if not path.suffix:
        path = path.with_suffix(".pt")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        torch.save(setup.payload(), temporary_path)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return path


def load_setup(path: str | Path) -> SplineSetup:
    path = Path(path).expanduser()
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("kind") != SETUP_KIND:
        raise ValueError(f"{path} is not a spline playground setup")
    version = payload.get("format_version")
    if version != FORMAT_VERSION:
        raise ValueError(
            f"unsupported spline setup format {version!r}; expected {FORMAT_VERSION}"
        )
    required_parameters = {
        "alpha",
        "beta",
        "gamma",
        "rho",
        "cg_eps",
        "n_steps",
        "control_steps",
    }
    missing = required_parameters.difference(payload.get("parameters", {}))
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"spline setup is missing parameters: {names}")
    parameter_values = payload["parameters"]
    parameters = SplineParameters.from_dict(parameter_values)
    if (
        "control_times" in parameter_values
        and (
            tuple(parameter_values["control_times"]) != parameters.control_times
            or tuple(parameter_values["control_steps"]) != parameters.control_steps
        )
    ):
        raise ValueError("saved control_steps do not match control_times")
    return SplineSetup(
        source=payload["source"],
        target=payload["target"],
        initial_momentum=payload["initial_momentum"],
        initial_force=payload["initial_force"],
        initial_jerk=payload["initial_jerk"],
        control_jerks=payload["control_jerks"],
        parameters=parameters,
        source_path=payload.get("source_path", ""),
        target_path=payload.get("target_path", ""),
    )


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
    kernel = SobolevFluidOperator(
        alpha=parameters.alpha,
        beta=parameters.beta,
        gamma=parameters.gamma,
        boundary="periodic",
    )
    with torch.no_grad():
        acceleration = CometricOperator(
            image,
            parameters.rho,
            kernel,
            dx_convention="pixel",
        )(covector)
    return float((covector * acceleration).sum().detach().cpu())


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
    ).inverse(acceleration, eps=parameters.cg_eps)
    gradient = tb.spatialGradient(
        image,
        dx_convention="pixel",
        boundary="periodic",
    )[:, 0]
    velocity = -sqrt(parameters.rho) * kernel(
        integrator.momentum * gradient
    )
    return force, velocity


def run_spline(
    setup: SplineSetup,
    *,
    device: str | torch.device = "auto",
    progress_callback: Callable[[int, int], None] | None = None,
) -> SplineTrajectory:
    """Run the forward spline and return only detached node-aligned data."""
    if progress_callback is not None and not callable(progress_callback):
        raise TypeError("progress_callback must be callable")
    parameters = setup.parameters
    run_device = resolve_device(device)
    source = setup.source.to(run_device)
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

    if run_device.type == "cuda":
        torch.cuda.synchronize(run_device)
    start = perf_counter()
    with torch.no_grad():
        initial_force = setup.initial_force.to(run_device)
        initial_acceleration = CometricOperator(
            source,
            parameters.rho,
            kernel,
            dx_convention="pixel",
        )(initial_force)
        variables = SplinesVariables(
            initial_momentum=setup.initial_momentum.to(run_device),
            initial_acceleration=initial_acceleration,
            initial_jerk=setup.initial_jerk.to(run_device),
            control_jerks=setup.control_jerks.to(run_device),
        )
        integrator = MetamorphosisSplineIntegrator(
            parameters.rho,
            control_times=parameters.mesh_control_times,
            kernelOperator=kernel,
            n_step=parameters.n_steps,
            cg_eps=parameters.cg_eps,
            dx_convention="pixel",
        )
        integrator(
            source,
            variables,
            save=True,
            progress_callback=integrator_progress,
        )
        endpoint_force, endpoint_velocity = _endpoint_fields(
            integrator,
            kernel,
            parameters,
        )
        deformed_source_device, photometric_only_device = _decompose_image_nodes(
            source,
            integrator.field_stock.to(run_device),
            integrator.residuals_stock.to(run_device),
        )
    if run_device.type == "cuda":
        torch.cuda.synchronize(run_device)

    images = integrator.image_stock.detach().cpu().contiguous()
    deformed_source = deformed_source_device.detach().cpu().contiguous()
    photometric_only = photometric_only_device.detach().cpu().contiguous()
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
    momentum_energy = (
        (1 - parameters.rho) * momentum.square().sum(dim=(1, 2, 3))
        + velocity_energy
    )
    force_acceleration_energy = (force * acceleration).sum(dim=(1, 2, 3))
    field_energies = {
        "momentum": momentum_energy,
        "force": force_acceleration_energy,
        "acceleration": force_acceleration_energy,
        "jerk": jerk.square().sum(dim=(1, 2, 3)),
        "velocity": velocity_energy,
        "vector_momentum": velocity_energy,
    }
    target = setup.target[0].to(dtype=images.dtype)
    target_mse = (images - target).square().mean(dim=(1, 2, 3))
    if progress_callback is not None:
        progress_callback(parameters.n_steps, parameters.n_steps)
    elapsed = perf_counter() - start

    return SplineTrajectory(
        images=images,
        deformed_source=deformed_source,
        photometric_only=photometric_only,
        momentum=momentum,
        force=force,
        acceleration=acceleration,
        jerk=jerk,
        velocity=velocity,
        vector_momentum=vector_momentum,
        field_energies=field_energies,
        target_mse=target_mse,
        elapsed_seconds=elapsed,
    )
