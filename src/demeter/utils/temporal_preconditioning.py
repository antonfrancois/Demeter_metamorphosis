"""Cometric-aware temporal coordinates for metamorphosis spline shooting."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .cometric_inversion import CometricOperator

_RELATIVE_EIGENVALUE_FLOOR = 1e-6


def _temporal_basis_matrices(
    n_step: int,
    control_steps: tuple[int, ...],
    target_steps: tuple[int, ...],
) -> tuple[Tensor, Tensor]:
    n_parameters = 3 + len(control_steps)
    image = torch.zeros(n_parameters, dtype=torch.float64)
    momentum = torch.zeros_like(image)
    acceleration = torch.zeros_like(image)
    jerk = torch.zeros_like(image)
    momentum[0] = 1
    acceleration[1] = 1
    jerk[2] = 1
    controls = {step: 3 + index for index, step in enumerate(control_steps)}
    dt = 1 / n_step
    images = [image.clone()]
    acceleration_intervals = []
    for step in range(1, n_step + 1):
        acceleration_intervals.append(acceleration.clone())
        image = image + dt * momentum
        momentum = momentum + dt * acceleration
        acceleration = acceleration + dt * jerk
        control_index = controls.get(step)
        if control_index is not None:
            jerk = torch.zeros_like(jerk)
            jerk[control_index] = 1
        images.append(image.clone())
    return (
        torch.stack(images)[list(target_steps)],
        torch.stack(acceleration_intervals),
    )


def source_cometric_moments(
    source: Tensor,
    rho: float,
    kernel_operator,
    cg_eps: float,
) -> Tensor:
    """Estimate normalized traces of ``A^-1``, ``A``, and ``A^2``."""
    if rho == 0:
        return torch.ones(3, dtype=torch.float64)

    generator = torch.Generator().manual_seed(0)
    probe = torch.empty(source.shape, dtype=torch.float64).bernoulli_(
        0.5,
        generator=generator,
    )
    probe.mul_(2).sub_(1)
    probe = probe.to(source)
    operator = CometricOperator(
        source.detach(),
        rho,
        kernel_operator,
        dx_convention="pixel",
    )
    with torch.no_grad():
        applied = operator(probe)
        inverse = operator.inverse(probe, eps=cg_eps)
        moments = torch.stack(
            (
                (probe * inverse).sum(),
                (probe * applied).sum(),
                applied.square().sum(),
            )
        )
    return moments.to(device="cpu", dtype=torch.float64) / probe.numel()


def cometric_temporal_metric(
    source: Tensor,
    rho: float,
    kernel_operator,
    cg_eps: float,
    n_step: int,
    control_steps: tuple[int, ...],
    target_steps: tuple[int, ...],
    cost_cst: float,
) -> Tensor:
    """Return the zero-trajectory block-trace Gauss-Newton metric."""
    observation, acceleration = _temporal_basis_matrices(
        n_step,
        control_steps,
        target_steps,
    )
    inverse_trace, trace, square_trace = source_cometric_moments(
        source,
        rho,
        kernel_operator,
        cg_eps,
    )
    operator_powers = torch.ones(observation.shape[1], dtype=torch.long)
    operator_powers[1] = 0
    pair_powers = operator_powers[:, None] + operator_powers[None, :]
    data_factors = torch.stack(
        (torch.ones((), dtype=torch.float64), trace, square_trace)
    )[pair_powers]
    regularizer_factors = torch.stack(
        (inverse_trace, torch.ones((), dtype=torch.float64), trace)
    )[pair_powers]
    metric = (observation.T @ observation) * data_factors
    metric += (cost_cst / n_step) * (
        acceleration.T @ acceleration
    ) * regularizer_factors
    return 0.5 * (metric + metric.T)


@dataclass(frozen=True)
class TemporalTransform:
    """Damped square-root coordinate transform on active temporal blocks."""

    active_indices: tuple[int, ...]
    factor: Tensor
    inverse_factor: Tensor

    @classmethod
    def from_metric(
        cls,
        metric: Tensor,
        active_indices: tuple[int, ...],
    ) -> "TemporalTransform":
        if not active_indices:
            raise ValueError("at least one temporal block must be active")
        indices = torch.tensor(active_indices, dtype=torch.long)
        active_metric = metric.index_select(0, indices).index_select(1, indices)
        eigenvalues, eigenvectors = torch.linalg.eigh(active_metric.double())
        largest = float(eigenvalues[-1])
        if largest <= 0:
            factor = torch.eye(len(active_indices), dtype=torch.float64)
            return cls(active_indices, factor, factor)
        eigenvalues = eigenvalues.clamp_min(
            largest * _RELATIVE_EIGENVALUE_FLOOR
        )
        factor = eigenvectors @ torch.diag(eigenvalues.sqrt()) @ eigenvectors.T
        inverse = (
            eigenvectors @ torch.diag(eigenvalues.rsqrt()) @ eigenvectors.T
        )
        return cls(active_indices, factor, inverse)

    def apply(self, values: Tensor, *, inverse: bool = False) -> Tensor:
        indices = torch.tensor(self.active_indices, device=values.device)
        factor = self.inverse_factor if inverse else self.factor
        transformed = torch.einsum(
            "ij,j...->i...",
            factor.to(values),
            values.index_select(0, indices),
        )
        return values.index_copy(0, indices, transformed)

    def to(self, reference: Tensor) -> "TemporalTransform":
        return TemporalTransform(
            self.active_indices,
            self.factor.to(reference),
            self.inverse_factor.to(reference),
        )
