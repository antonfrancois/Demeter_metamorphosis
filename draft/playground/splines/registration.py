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

    def loss_curves(self) -> dict[str, torch.Tensor]:
        """Return comparable displayed components for one optimization run."""
        cost = self.setup.parameters.spline.cost
        if self.setup.parameters.model == "splines":
            if not isinstance(self.loss_stock, dict):
                raise ValueError("spline loss history must be a dictionary")
            data = torch.as_tensor(self.loss_stock["data_loss"])
            regularized = cost * torch.as_tensor(
                self.loss_stock["acceleration_energy"]
            )
            full = torch.as_tensor(self.loss_stock["total_cost"])
        else:
            losses = torch.as_tensor(self.loss_stock)
            if losses.numel() == 0:
                empty = torch.empty(0)
                return {
                    "full": empty,
                    "data": empty.clone(),
                    "regularized": empty.clone(),
                }
            if losses.ndim != 2 or losses.shape[1] < 2:
                raise ValueError("classic loss history must have component columns")
            data = losses[:, 0]
            regularized = cost * losses[:, 1:].sum(dim=1)
            full = data + regularized
        return {
            "full": full.detach().cpu(),
            "data": data.detach().cpu(),
            "regularized": regularized.detach().cpu(),
        }

    @property
    def regularized_loss_label(self) -> str:
        return (
            "Regularized acceleration cost"
            if self.setup.parameters.model == "splines"
            else "Regularized momentum cost"
        )


LBFGS_MAX_ITER = 5
LBFGS_HISTORY_SIZE = 10
SHOOTING_OPTIMIZER_OPTIONS = {
    "safe_mode": False,
    "integration_method": "semiLagrangian",
    "optimizer_method": "LBFGS_torch",
    "dx_convention": "pixel",
    "lbfgs_max_iter": LBFGS_MAX_ITER,
    "lbfgs_history_size": LBFGS_HISTORY_SIZE,
    "boundary": "periodic",
}


def _zeroed_setup(setup: SplineSetup) -> SplineSetup:
    return replace(
        setup,
        variables=SplinesVariables.zeros(
            setup.images.source,
            setup.n_controls,
            requires_grad=False,
        ),
    )


def register_classic(
    setup: SplineSetup,
    *,
    device: str | torch.device = "auto",
    progress_callback: Callable[[int, int], None] | None = None,
) -> RegistrationResult:
    """Optimize one classic endpoint registration from zero momentum."""
    if setup.images.target.shape[0] != 1 or setup.images.target_times != (1.0,):
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
        )
    settings = setup.parameters.spline
    optimizer = metamorphosis(
        source=setup.images.source.to(run_device),
        target=setup.images.target.to(run_device),
        momentum_ini=0.0,
        rho=setup.parameters.rho,
        cost_cst=settings.cost,
        integration_steps=settings.steps,
        n_iter=settings.iterations,
        grad_coef=settings.learning_rate,
        kernelOperator=_kernel_operator(setup.parameters),
        **SHOOTING_OPTIMIZER_OPTIONS,
    )
    momenta = optimizer.optimized_momenta
    if not isinstance(momenta, Momenta) or momenta.momentum_I is None:
        raise RuntimeError("classic registration produced no momentum")
    optimized_setup = _zeroed_setup(setup)
    optimized_setup = replace(
        optimized_setup,
        variables=replace(
            optimized_setup.variables,
            initial_momentum=momenta.momentum_I.detach().cpu(),
        ),
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
    source = setup.images.source.to(run_device)
    start = perf_counter()
    variables_ini = SplinesVariables.zeros(
        source,
        n_controls=setup.n_controls,
        requires_grad=False,
    )
    if setup.parameters.initialization == "warm":
        settings = setup.parameters.regression
        regression = metamorphosis_regression(
            source=source,
            target=setup.images.target.to(run_device),
            target_times=setup.images.target_times,
            momentum_ini=0.0,
            rho=setup.parameters.rho,
            cost_cst=settings.cost,
            integration_steps=settings.steps,
            n_iter=settings.iterations,
            grad_coef=settings.learning_rate,
            kernelOperator=_kernel_operator(setup.parameters),
            **SHOOTING_OPTIMIZER_OPTIONS,
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
        optimized_setup = _zeroed_setup(setup)
        optimized_setup = replace(
            optimized_setup,
            variables=replace(
                optimized_setup.variables,
                initial_momentum=variables_ini.initial_momentum.detach().cpu(),
            ),
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
        )

    integrator = MetamorphosisSplineIntegrator(
        rho=setup.parameters.rho,
        control_times=setup.parameters.projected_control_times,
        kernelOperator=_kernel_operator(setup.parameters),
        n_step=setup.parameters.spline.steps,
        cg_eps=setup.parameters.cg_tolerance,
        dx_convention="pixel",
    )
    optimizer = MetamorphosisSplineOptimizer(
        source=source,
        target=setup.images.target.to(run_device),
        target_times=setup.images.target_times,
        geodesic=integrator,
        cost_cst=setup.parameters.spline.cost,
        optimizer_method="LBFGS_torch",
        lbfgs_max_iter=LBFGS_MAX_ITER,
        lbfgs_history_size=LBFGS_HISTORY_SIZE,
    )
    optimizer.forward(
        variables_ini,
        n_iter=setup.parameters.spline.iterations,
        grad_coef=setup.parameters.spline.learning_rate,
    )
    variables = optimizer.optimized_variables
    if variables is None:
        raise RuntimeError("spline registration produced no variables")
    trajectory = _trajectory_from_final_spline_integration(
        optimizer.mp,
        setup.parameters,
        setup.images.target,
        progress_callback=progress_callback,
    )
    optimized_setup = replace(
        setup,
        variables=variables.to("cpu").detach(),
    )
    return RegistrationResult(
        optimized_setup,
        trajectory,
        optimizer.loss_stock,
        perf_counter() - start,
    )
