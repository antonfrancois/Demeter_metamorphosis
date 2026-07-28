"""Optimization adapters that return normal lab setups and trajectories."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import torch

from demeter.metamorphosis import MetamorphosisSplines, metamorphosis
from demeter.metamorphosis.splines import SplinesVariables
from demeter.metamorphosis.var_classes import Momenta

from .core import (
    SplineSetup,
    SplineTrajectory,
    _kernel_operator,
    resolve_device,
    run_classic,
    run_spline,
)


@dataclass(frozen=True)
class RegistrationResult:
    setup: SplineSetup
    trajectory: SplineTrajectory
    loss_stock: Any
    elapsed_seconds: float
    model: str

    def payload(self) -> dict[str, Any]:
        losses = self.loss_stock
        if torch.is_tensor(losses):
            losses = losses.detach().cpu().clone()
        elif isinstance(losses, dict):
            losses = {
                name: value.detach().cpu().clone()
                for name, value in losses.items()
            }
        return {
            "model": self.model,
            "elapsed_seconds": self.elapsed_seconds,
            "loss_stock": losses,
        }


LBFGS_MAX_ITER = 5
LBFGS_HISTORY_SIZE = 10
GRAD_COEF = 1.0


def register_classic(
    setup: SplineSetup,
    *,
    device: str | torch.device = "auto",
    progress_callback: Callable[[int, int], None] | None = None,
) -> RegistrationResult:
    """Optimize one classic endpoint registration from zero momentum."""
    if setup.target.shape[0] != 1 or setup.target_times != (1.0,):
        raise ValueError("classic registration requires one target at time 1")
    run_device = resolve_device(device)
    start = perf_counter()
    optimizer = metamorphosis(
        source=setup.source.to(run_device),
        target=setup.target.to(run_device),
        momentum_ini=0.0,
        rho=setup.parameters.rho,
        cost_cst=setup.parameters.cost_cst,
        integration_steps=setup.parameters.n_steps,
        n_iter=setup.parameters.iterations,
        grad_coef=GRAD_COEF,
        kernelOperator=_kernel_operator(setup.parameters),
        safe_mode=False,
        integration_method="semiLagrangian",
        optimizer_method="LBFGS_torch",
        dx_convention="pixel",
        lbfgs_max_iter=LBFGS_MAX_ITER,
        lbfgs_history_size=LBFGS_HISTORY_SIZE,
        boundary="periodic",
    )
    momenta = optimizer.optimized_momenta
    if not isinstance(momenta, Momenta) or momenta.momentum_I is None:
        raise RuntimeError("classic registration produced no momentum")
    zero = torch.zeros_like(setup.source)
    optimized_setup = SplineSetup(
        source=setup.source,
        target=setup.target,
        initial_momentum=momenta.momentum_I.detach().cpu(),
        initial_force=zero,
        initial_jerk=zero,
        control_jerks=setup.source.new_zeros(
            (setup.n_controls,) + tuple(setup.source.shape)
        ),
        parameters=setup.parameters,
        source_path=setup.source_path,
        target_path=setup.target_path,
        target_times=setup.target_times,
        target_paths=setup.target_paths,
    )
    trajectory = run_classic(
        optimized_setup,
        device=run_device,
        progress_callback=progress_callback,
    )
    return RegistrationResult(
        optimized_setup,
        trajectory,
        optimizer.loss_stock,
        perf_counter() - start,
        "classic",
    )


def register_spline(
    setup: SplineSetup,
    *,
    device: str | torch.device = "auto",
    progress_callback: Callable[[int, int], None] | None = None,
) -> RegistrationResult:
    """Optimize all spline shooting fields from zero initial fields."""
    if setup.parameters.kernel != "sobolev":
        raise ValueError("spline registration requires the Sobolev operator")
    run_device = resolve_device(device)
    source = setup.source.to(run_device)
    start = perf_counter()
    optimizer = MetamorphosisSplines(
        source=source,
        target=setup.target.to(run_device),
        target_times=setup.target_times,
        variables_ini=SplinesVariables.zeros(
            source,
            n_controls=setup.n_controls,
        ),
        rho=setup.parameters.rho,
        cost_cst=setup.parameters.cost_cst,
        integration_steps=setup.parameters.n_steps,
        n_iter=setup.parameters.iterations,
        grad_coef=GRAD_COEF,
        kernelOperator=_kernel_operator(setup.parameters),
        control_times=setup.parameters.mesh_control_times,
        cg_eps=setup.parameters.cg_eps,
        safe_mode=False,
        lbfgs_max_iter=LBFGS_MAX_ITER,
        lbfgs_history_size=LBFGS_HISTORY_SIZE,
    )
    variables = optimizer.optimized_variables
    if variables is None:
        raise RuntimeError("spline registration produced no variables")
    initial_force = optimizer.mp.force_stock[0:1].detach().cpu()
    optimized_setup = SplineSetup(
        source=setup.source,
        target=setup.target,
        initial_momentum=variables.initial_momentum,
        initial_force=initial_force,
        initial_jerk=variables.initial_jerk,
        control_jerks=variables.control_jerks,
        parameters=setup.parameters,
        source_path=setup.source_path,
        target_path=setup.target_path,
        target_times=setup.target_times,
        target_paths=setup.target_paths,
    )
    trajectory = run_spline(
        optimized_setup,
        device=run_device,
        progress_callback=progress_callback,
    )
    return RegistrationResult(
        optimized_setup,
        trajectory,
        optimizer.loss_stock,
        perf_counter() - start,
        "splines",
    )
