"""Optimization adapters that return normal lab setups and trajectories."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from time import perf_counter
from typing import Any

import torch

from demeter.metamorphosis import metamorphosis, metamorphosis_regression
from demeter.metamorphosis.splines import (
    MetamorphosisSplineIntegrator,
    MetamorphosisSplineOptimizer,
    SplinesVariables,
)
from demeter.metamorphosis.var_classes import Momenta

from .core import (
    SplineSetup,
    SplineTrajectory,
    _kernel_operator,
    _trajectory_from_final_spline_integration,
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
REGRESSION_LBFGS_LR = 1.0


class _SelectedFieldSplineOptimizer(MetamorphosisSplineOptimizer):
    def _dict_or_torch_parameter_(self) -> list[torch.Tensor]:
        return [
            value
            for _, value in self._optimization_parameter
            if value.requires_grad and value.numel()
        ]


def _zeroed_setup(setup: SplineSetup) -> SplineSetup:
    zero = torch.zeros_like(setup.source)
    return replace(
        setup,
        initial_momentum=zero,
        initial_acceleration=zero.clone(),
        initial_jerk=zero.clone(),
        control_jerks=setup.source.new_zeros(
            (setup.n_controls,) + tuple(setup.source.shape)
        ),
    )


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
    if "initial_momentum" not in setup.parameters.optimized_fields:
        optimized_setup = _zeroed_setup(setup)
        return RegistrationResult(
            optimized_setup,
            run_classic(
                optimized_setup,
                device=run_device,
                progress_callback=progress_callback,
            ),
            torch.empty(0),
            perf_counter() - start,
            "classic",
        )
    optimizer = metamorphosis(
        source=setup.source.to(run_device),
        target=setup.target.to(run_device),
        momentum_ini=0.0,
        rho=setup.parameters.rho,
        cost_cst=setup.parameters.cost_cst,
        integration_steps=setup.parameters.n_steps,
        n_iter=setup.parameters.iterations,
        grad_coef=setup.parameters.lbfgs_lr,
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
        initial_acceleration=zero,
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
    """Optimize selected spline fields from a cold or geodesic-regression start."""
    if setup.parameters.kernel != "sobolev":
        raise ValueError("spline registration requires the Sobolev operator")
    run_device = resolve_device(device)
    source = setup.source.to(run_device)
    start = perf_counter()
    variables_ini = SplinesVariables.zeros(
        source,
        n_controls=setup.n_controls,
        requires_grad=False,
    )
    if setup.parameters.spline_initialization == "warm":
        regression = metamorphosis_regression(
            source=source,
            target=setup.target.to(run_device),
            target_times=setup.target_times,
            momentum_ini=0.0,
            rho=setup.parameters.rho,
            cost_cst=setup.parameters.cost_cst,
            integration_steps=setup.parameters.n_steps,
            n_iter=setup.parameters.iterations,
            grad_coef=REGRESSION_LBFGS_LR,
            kernelOperator=_kernel_operator(setup.parameters),
            safe_mode=False,
            integration_method="semiLagrangian",
            optimizer_method="LBFGS_torch",
            dx_convention="pixel",
            lbfgs_max_iter=LBFGS_MAX_ITER,
            lbfgs_history_size=LBFGS_HISTORY_SIZE,
            boundary="periodic",
        )
        momenta = regression.optimized_momenta
        if not isinstance(momenta, Momenta) or momenta.momentum_I is None:
            raise RuntimeError(
                "geodesic regression initialization produced no momentum"
            )
        variables_ini.initial_momentum.copy_(
            momenta.momentum_I.detach().to(source)
        )
    selected_fields = set(setup.parameters.optimized_fields)
    for name, value in variables_ini:
        value.requires_grad_(name == "control_jerks" or name in selected_fields)
    if not any(value.requires_grad and value.numel() for _, value in variables_ini):
        optimized_setup = replace(
            _zeroed_setup(setup),
            initial_momentum=variables_ini.initial_momentum.detach().cpu(),
        )
        return RegistrationResult(
            optimized_setup,
            run_spline(
                optimized_setup,
                device=run_device,
                progress_callback=progress_callback,
            ),
            {
                name: torch.empty(0)
                for name in ("data_loss", "acceleration_energy", "total_cost")
            },
            perf_counter() - start,
            "splines",
        )

    integrator = MetamorphosisSplineIntegrator(
        rho=setup.parameters.rho,
        control_times=setup.parameters.mesh_control_times,
        kernelOperator=_kernel_operator(setup.parameters),
        n_step=setup.parameters.n_steps,
        cg_eps=setup.parameters.cg_eps,
        dx_convention="pixel",
    )
    optimizer = _SelectedFieldSplineOptimizer(
        source=source,
        target=setup.target.to(run_device),
        target_times=setup.target_times,
        geodesic=integrator,
        cost_cst=setup.parameters.cost_cst,
        optimizer_method="LBFGS_torch",
        lbfgs_max_iter=LBFGS_MAX_ITER,
        lbfgs_history_size=LBFGS_HISTORY_SIZE,
    )
    optimizer.forward(
        variables_ini,
        n_iter=setup.parameters.iterations,
        grad_coef=setup.parameters.lbfgs_lr,
    )
    variables = optimizer.optimized_variables
    if variables is None:
        raise RuntimeError("spline registration produced no variables")
    trajectory = _trajectory_from_final_spline_integration(
        optimizer.mp,
        setup.parameters,
        setup.target,
        progress_callback=progress_callback,
    )
    optimized_setup = SplineSetup(
        source=setup.source,
        target=setup.target,
        initial_momentum=variables.initial_momentum,
        initial_acceleration=variables.initial_acceleration,
        initial_jerk=variables.initial_jerk,
        control_jerks=variables.control_jerks,
        parameters=setup.parameters,
        source_path=setup.source_path,
        target_path=setup.target_path,
        target_times=setup.target_times,
        target_paths=setup.target_paths,
    )
    return RegistrationResult(
        optimized_setup,
        trajectory,
        optimizer.loss_stock,
        perf_counter() - start,
        "splines",
    )
