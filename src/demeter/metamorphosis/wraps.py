import torch
# import __init__
import sys
from icecream import ic
from warnings import warn

from . import classic as cl
from .var_classes import Momenta
from . import constrained as cn
from . import simplex as sp
from . import joined as jn
from . import affine as aff
from . import affine_decoupled as ad
from . import splines as spl

from ..utils import torchbox as tb
from ..utils.decorators import time_it, monitor_gpu


def _commun_before(momentum_ini, source):
    if isinstance(momentum_ini, (int, float)):
        momentum_ini = torch.full_like(source, float(momentum_ini))
    if isinstance(momentum_ini, torch.Tensor):
        if momentum_ini.shape != source.shape:
            raise ValueError(
                "initial momentum and source must have the same shape, got "
                f"{momentum_ini.shape} and {source.shape}"
            )
        momentum_ini = Momenta(momentum_I=momentum_ini)
    if not isinstance(momentum_ini, Momenta):
        raise TypeError(
            "momentum_ini must be a scalar, tensor, or Momenta instance, got "
            f"{type(momentum_ini)}"
        )
    return momentum_ini.requires_grad_(True)


def _commun_after(mr, momentum_ini, safe_mode, n_iter, grad_coef,
                  convergence_tol=None, convergence_patience=3):
    if n_iter == 0:
        return mr
    if not safe_mode:
        mr.forward(momentum_ini, n_iter=n_iter, grad_coef=grad_coef,
                   convergence_tol=convergence_tol, convergence_patience=convergence_patience)
    else:
        mr.forward_safe_mode(momentum_ini, n_iter=n_iter, grad_coef=grad_coef,
                             convergence_tol=convergence_tol, convergence_patience=convergence_patience)
    return mr


@time_it
def lddmm(
    source,
    target,
    momentum_ini,
    kernelOperator,
    cost_cst,
    integration_steps,
    n_iter,
    grad_coef,
    data_term=None,
    sharp=False,
    safe_mode=False,
    integration_method="semiLagrangian",
    dx_convention="pixel",
    optimizer_method="LBFGS_torch",
    hamiltonian_integration=False,
    save_gpu_memory=False,
    lbfgs_max_iter: int = 20,
    lbfgs_history_size: int = 100,
    convergence_tol=None,
    convergence_patience=3,
    adam_scheduler="reduce_on_plateau",
    adam_grad_clip=None,
    boundary="legacy",
):
    """
    Perform a Large Deformation Diffeomorphic Metric Mapping (LDDMM) transformation between a source and a target.

    Parameters
    ----------
    source : torch.Tensor
        Source image of shape ``[B, C, H, W]`` or ``[B, C, H, W, D]``.
    target : torch.Tensor
        Target image with the same shape as ``source``.
    momentum_ini : torch.Tensor or float
        Initial momentum to optimize. A scalar is broadcast to ``source.shape``.
    kernelOperator : callable
        Reproducing kernel operator used by the geodesic integrator.
    cost_cst : float
        Weight of the regularization term.
    integration_steps : int
        Number of geodesic integration steps.
    n_iter : int
        Number of optimization iterations.
    grad_coef : float
        Multiplier applied to the optimization step.
    data_term : optional
        Data attachment term. Expected to be a ``DataCost`` subclass instance.
    sharp : bool, optional
        If ``True``, forces ``integration_method="sharp"``.
    safe_mode : bool, optional
        If ``True``, use ``forward_safe_mode`` for optimization.
    integration_method : str, optional
        Integrator method passed to ``Metamorphosis_integrator``.
    dx_convention : str, optional
        Spatial convention used for finite differences.
    optimizer_method : str, optional
        Optimizer used by ``Metamorphosis_Shooting``.
    hamiltonian_integration : bool, optional
        Enable Hamiltonian integration in the optimizer.
    save_gpu_memory : bool, optional
        Enable memory-saving mode in the geodesic integrator.
    lbfgs_max_iter : int, optional
        Maximum number of internal iterations for LBFGS steps.
    lbfgs_history_size : int, optional
        History size used by LBFGS.
    convergence_tol : float or None, optional
        Early-stopping relative tolerance on the data loss. See
        ``Optimize_geodesicShooting.forward`` for details. Default ``None`` (disabled).
    convergence_patience : int, optional
        Number of consecutive plateau steps before stopping. Default 3.
    adam_scheduler : str or None, optional
        LR scheduler for Adam. ``None`` (default) disables scheduling.
        Choices: ``'reduce_on_plateau'``, ``'cosine'``, ``'exponential'``.
        Ignored for non-Adam optimizers.
    adam_grad_clip : float or None, optional
        If set, gradient norm is clipped to this value before each Adam step.
        ``None`` (default) disables clipping. Ignored for non-Adam optimizers.
    boundary : {"legacy", "periodic"}, optional
        Spatial boundary mode used by the classical integrator. Periodic mode
        currently requires 2D pixel coordinates.

    Returns
    -------
    cl.Metamorphosis_Shooting
        Configured optimizer after running the requested optimization iterations.
    """

    momentum_ini = _commun_before(momentum_ini, source)
    if sharp:
        integration_method = "sharp"

    mp = cl.Metamorphosis_integrator(
        method=integration_method,
        rho=1,
        n_step=integration_steps,
        kernelOperator=kernelOperator,
        dx_convention=dx_convention,
        save_gpu_memory=save_gpu_memory,
        boundary=boundary,
    )
    mr = cl.Metamorphosis_Shooting(
        source,
        target,
        mp,
        cost_cst=cost_cst,
        optimizer_method=optimizer_method,
        data_term=data_term,
        lbfgs_max_iter=lbfgs_max_iter,
        lbfgs_history_size=lbfgs_history_size,
        hamiltonian_integration=hamiltonian_integration,
        adam_scheduler=adam_scheduler,
        adam_grad_clip=adam_grad_clip,
    )

    mr = _commun_after(mr, momentum_ini, safe_mode, n_iter, grad_coef,
                      convergence_tol=convergence_tol, convergence_patience=convergence_patience)

    return mr


@time_it
def metamorphosis(
    source,
    target,
    momentum_ini,
    rho,
    cost_cst,
    integration_steps,
    n_iter,
    grad_coef,
    kernelOperator,
    data_term=None,
    sharp=False,
    safe_mode=True,
    integration_method="semiLagrangian",
    optimizer_method="LBFGS_torch",
    dx_convention="pixel",
    hamiltonian_integration=False,
    lbfgs_max_iter: int = 20,
    lbfgs_history_size: int = 100,
    save_gpu_memory=False,
    convergence_tol=None,
    convergence_patience=3,
    adam_scheduler="reduce_on_plateau",
    adam_grad_clip=None,
    boundary="legacy",
):
    """
    Run metamorphosis registration from ``source`` to ``target``.

    This is the generic metamorphosis wrapper (with user-provided ``rho``),
    whereas :func:`lddmm` is the special case with ``rho=1``.

    Parameters
    ----------
    source : torch.Tensor
        Source image of shape ``[B, C, H, W]`` or ``[B, C, H, W, D]``.
    target : torch.Tensor
        Target image with the same shape as ``source``.
    momentum_ini : torch.Tensor or float
        Initial momentum to optimize. A scalar is broadcast to ``source.shape``.
    rho : float
        Metamorphosis trade-off parameter.
    cost_cst : float
        Weight of the regularization term.
    integration_steps : int
        Number of geodesic integration steps.
    n_iter : int
        Number of optimization iterations.
    grad_coef : float
        Multiplier applied to the optimization step.
    kernelOperator : callable
        Reproducing kernel operator used by the geodesic integrator.
    data_term : optional
        Data attachment term. Expected to be a ``DataCost`` subclass instance.
    sharp : bool, optional
        If ``True``, forces ``integration_method="sharp"``.
    safe_mode : bool, optional
        If ``True``, use ``forward_safe_mode`` for optimization.
    integration_method : str, optional
        Integrator method passed to ``Metamorphosis_integrator``.
    optimizer_method : str, optional
        Optimizer used by ``Metamorphosis_Shooting``.
    dx_convention : str, optional
        Spatial convention used for finite differences.
    hamiltonian_integration : bool, optional
        Enable Hamiltonian integration in the optimizer.
    lbfgs_max_iter : int, optional
        Maximum number of internal iterations for LBFGS steps.
    lbfgs_history_size : int, optional
        History size used by LBFGS.
    save_gpu_memory : bool, optional
        Enable memory-saving mode in the geodesic integrator.
    convergence_tol : float or None, optional
        Early-stopping relative tolerance on the data loss. See
        ``Optimize_geodesicShooting.forward`` for details. Default ``None`` (disabled).
    convergence_patience : int, optional
        Number of consecutive plateau steps before stopping. Default 3.
    adam_scheduler : str or None, optional
        LR scheduler for Adam. ``None`` (default) disables scheduling.
        Choices: ``'reduce_on_plateau'``, ``'cosine'``, ``'exponential'``.
        Ignored for non-Adam optimizers.
    adam_grad_clip : float or None, optional
        If set, gradient norm is clipped to this value before each Adam step.
        ``None`` (default) disables clipping. Ignored for non-Adam optimizers.
    boundary : {"legacy", "periodic"}, optional
        Spatial boundary mode used by the classical integrator. Periodic mode
        currently requires 2D pixel coordinates.

    Returns
    -------
    cl.Metamorphosis_Shooting
        Configured optimizer after running the requested optimization iterations.
    """

    momentum_ini = _commun_before(momentum_ini, source)
    if sharp:
        integration_method = "sharp"

    mp = cl.Metamorphosis_integrator(
        method=integration_method,
        rho=rho,
        kernelOperator=kernelOperator,
        n_step=integration_steps,
        dx_convention=dx_convention,
        save_gpu_memory=save_gpu_memory,
        boundary=boundary,
    )
    mr = cl.Metamorphosis_Shooting(
        source,
        target,
        mp,
        cost_cst=cost_cst,
        data_term=data_term,
        optimizer_method=optimizer_method,
        lbfgs_max_iter=lbfgs_max_iter,
        lbfgs_history_size=lbfgs_history_size,
        hamiltonian_integration=hamiltonian_integration,
        adam_scheduler=adam_scheduler,
        adam_grad_clip=adam_grad_clip,
    )

    mr = _commun_after(mr, momentum_ini, safe_mode, n_iter, grad_coef,
                      convergence_tol=convergence_tol, convergence_patience=convergence_patience)
    return mr


@time_it
def MetamorphosisSplines(
    source,
    target,
    target_times,
    variables_ini,
    rho,
    cost_cst,
    integration_steps,
    n_iter,
    grad_coef,
    kernelOperator,
    control_times=(),
    cg_eps=1e-6,
    safe_mode=False,
    optimizer_method="LBFGS_torch",
    lbfgs_max_iter=20,
    lbfgs_history_size=100,
    convergence_tol=None,
    convergence_patience=3,
    debug=False,
):
    """Fit a metamorphosis spline to images observed at normalized times.

    ``source`` is fixed at time zero. ``target`` has shape ``[N, 1, H, W]``
    and ``target_times`` contains its strictly increasing observation times.
    Both observation and control times must lie on the integration mesh.
    ``rho`` must satisfy ``0 <= rho < 1``, and ``cost_cst`` represents the
    observation variance ``sigma_I**2`` in equation (37).
    """
    if not isinstance(n_iter, int) or isinstance(n_iter, bool) or n_iter < 0:
        raise ValueError("n_iter must be a non-negative integer")
    control_times = tuple(control_times)
    if variables_ini is None:
        variables_ini = spl.SplinesVariables.zeros(
            source,
            n_controls=len(control_times),
        )
    if not isinstance(variables_ini, spl.SplinesVariables):
        raise TypeError("variables_ini must be SplinesVariables or None")
    variables_ini = spl.SplinesVariables(
        *(value.detach().clone() for _, value in variables_ini)
    ).requires_grad_(True)

    integrator = spl.MetamorphosisSplineIntegrator(
        rho=rho,
        control_times=control_times,
        kernelOperator=kernelOperator,
        n_step=integration_steps,
        cg_eps=cg_eps,
        dx_convention="pixel",
        debug=debug,
    )
    optimizer = spl.MetamorphosisSplineOptimizer(
        source=source,
        target=target,
        target_times=target_times,
        geodesic=integrator,
        cost_cst=cost_cst,
        optimizer_method=optimizer_method,
        lbfgs_max_iter=lbfgs_max_iter,
        lbfgs_history_size=lbfgs_history_size,
        debug=debug,
    )
    if n_iter == 0:
        return optimizer

    forward = optimizer.forward_safe_mode if safe_mode else optimizer.forward
    forward(
        variables_ini,
        n_iter=n_iter,
        grad_coef=grad_coef,
        convergence_tol=convergence_tol,
        convergence_patience=convergence_patience,
    )
    return optimizer


# ==================================================
#  From constrained.py


@time_it
def weighted_metamorphosis(
    source,
    target,
    momentum_ini,
    residual_mask,
    kernelOperator,
    cost_cst,
    n_iter,
    grad_coef,
    data_term=None,
    sharp=False,
    safe_mode=True,
    optimizer_method="adadelta",
    dx_convention="pixel",
    convergence_tol=None,
    convergence_patience=3,
    adam_scheduler="reduce_on_plateau",
    adam_grad_clip=None,
):
    print("plop")
    device = source.device
    momentum_ini = _commun_before(momentum_ini, source)

    mp_weighted = cn.ConstrainedMetamorphosis_integrator(
        residual_mask=residual_mask,
        kernelOperator=kernelOperator,
        sharp=sharp,
        dx_convention=dx_convention,
    )
    mr_weighted = cn.ConstrainedMetamorphosis_Shooting(
        source,
        target,
        mp_weighted,
        cost_cst=cost_cst,
        optimizer_method=optimizer_method,
        data_term=data_term,
        adam_scheduler=adam_scheduler,
        adam_grad_clip=adam_grad_clip,
    )

    mr_weighted = _commun_after(mr_weighted, momentum_ini, safe_mode, n_iter, grad_coef,
                               convergence_tol=convergence_tol, convergence_patience=convergence_patience)
    return mr_weighted


@time_it
def oriented_metamorphosis(
    source,
    target,
    momentum_ini,
    orienting_mask,
    orienting_field,
    kernelOperator,
    cost_cst,
    n_iter,
    grad_coef,
    dx_convention="pixel",
    hamiltonian_integration=False,
    safe_mode=True,
    convergence_tol=None,
    convergence_patience=3,
    adam_scheduler="reduce_on_plateau",
    adam_grad_clip=None,
):
    if hamiltonian_integration:
        raise NotImplementedError("Hamiltonian integration is not implemented yet for this function.")

    if type(momentum_ini) == int:
        momentum_ini = torch.zeros(source.shape)
    momentum_ini.requires_grad = True

    # start = time.time()
    mp_orient = cn.ConstrainedMetamorphosis_integrator(
        orienting_mask=orienting_mask,
        orienting_field=orienting_field,
        residual_mask=1,
        kernelOperator=kernelOperator,
        dx_convention=dx_convention,
        # n_step=20 # n_step is defined from mask.shape[0]
    )
    mr_orient = cn.ConstrainedMetamorphosis_Shooting(
        source, target, mp_orient, cost_cst=cost_cst, optimizer_method="LBFGS_torch",
        hamiltonian_integration=hamiltonian_integration,
        adam_scheduler=adam_scheduler,
        adam_grad_clip=adam_grad_clip,
    )
    mr_orient = _commun_after(mr_orient, momentum_ini, safe_mode, n_iter, grad_coef,
                             convergence_tol=convergence_tol, convergence_patience=convergence_patience)
    return mr_orient


@time_it
def constrained_metamorphosis(
    source,
    target,
    momentum_ini,
    orienting_mask,
    orienting_field,
    residual_mask,
    kernelOperator,
    cost_cst,
    n_iter,
    grad_coef,
    sharp=False,
    dx_convention="pixel",
    optimizer_method='LBFGS_torch',
    safe_mode=True,
    hamiltonian_integration=False,
    save_gpu_memory=False,
    convergence_tol=None,
    convergence_patience=3,
    adam_scheduler="reduce_on_plateau",
    adam_grad_clip=None,
):
    if hamiltonian_integration:
        raise NotImplementedError("Hamiltonian integration is not implemented yet for this function.")

    momentum_ini = _commun_before(momentum_ini, source)

    # start = time.time()
    mp_constr = cn.ConstrainedMetamorphosis_integrator(
        orienting_mask=orienting_mask,
        orienting_field=orienting_field,
        residual_mask=residual_mask,
        kernelOperator=kernelOperator,
        sharp=sharp,
        dx_convention=dx_convention,
        save_gpu_memory=save_gpu_memory,
        # n_step=20 # n_step is defined from mask.shape[0]
    )
    mr_constr = cn.ConstrainedMetamorphosis_Shooting(
        source, target, mp_constr, cost_cst=cost_cst, optimizer_method="LBFGS_torch",
        hamiltonian_integration=hamiltonian_integration,
        adam_scheduler=adam_scheduler,
        adam_grad_clip=adam_grad_clip,
    )
    # optimizer_method='adadelta')
    mr_constr = _commun_after(mr_constr, momentum_ini, safe_mode, n_iter, grad_coef,
                             convergence_tol=convergence_tol, convergence_patience=convergence_patience)
    return mr_constr


def joined_metamorphosis(
    source_image,
    target_image,
    source_mask,
    target_mask,
    momentum_ini,
    rho,
    kernelOperator,
    data_term=None,
    dx_convention="pixel",
    n_step=10,
    n_iter=1000,
    grad_coef=2,
    cost_cst=0.001,
    optimizer_method="LBFGS_torch",
    plot=False,
    safe_mode=False,
    debug=False,
    save_gpu_memory=False,
    convergence_tol=None,
    convergence_patience=3,
    adam_scheduler="reduce_on_plateau",
    adam_grad_clip=None,
):
    # source = torch.stack([source_image,source_mask],dim=1)
    # target = torch.stack([target_image,target_mask],dim=1)
    if type(momentum_ini) in [int, float]:
        shape = (1, 2) + source_image.shape[2:]
        momentum_ini = momentum_ini * torch.ones(
            shape, device=source_image.device, dtype=source_image.dtype
        )
    momentum_ini.requires_grad = True

    mp = jn.Weighted_joinedMask_Metamorphosis_integrator(
        rho=rho,
        kernelOperator=kernelOperator,
        n_step=n_step,
        dx_convention=dx_convention,
        save_gpu_memory = save_gpu_memory
        # debug=True
    )
    mr = jn.Weighted_joinedMask_Metamorphosis_Shooting(
        source_image,
        target_image,
        source_mask,
        target_mask,
        mp,
        cost_cst=cost_cst,
        data_term=data_term,
        optimizer_method=optimizer_method,
        adam_scheduler=adam_scheduler,
        adam_grad_clip=adam_grad_clip,
    )
    if safe_mode:
        mr.forward_safe_mode(momentum_ini, n_iter, grad_coef, plot,
                             convergence_tol=convergence_tol, convergence_patience=convergence_patience)
    else:
        mr.forward(momentum_ini, n_iter=n_iter, grad_coef=grad_coef, plot=plot, debug=debug,
                   convergence_tol=convergence_tol, convergence_patience=convergence_patience)
    return mr


def simplex_metamorphosis(
    source,
    target,
    momentum_ini,
    kernelOperator,
    rho,
    data_term,
    dx_convention="pixel",
    integration_steps=10,
    n_iter=1000,
    grad_coef=2,
    cost_cst=0.001,
    plot=False,
    safe_mode=False,
    ham=False,
    save_gpu_memory=False,
    lbfgs_max_iter: int = 20,
    lbfgs_history_size: int = 100,
    convergence_tol=None,
    convergence_patience=3,
    adam_scheduler="reduce_on_plateau",
    adam_grad_clip=None,
):

    if type(momentum_ini) in [int, float]:
        momentum_ini = momentum_ini * torch.ones(
            source.shape, device=source.device, dtype=source.dtype
        )
    momentum_ini.requires_grad = True

    mp = sp.Simplex_sqrt_Metamorphosis_integrator(
        rho=rho,
        kernelOperator=kernelOperator,
        n_step=integration_steps,
        dx_convention=dx_convention,
        save_gpu_memory=save_gpu_memory,
        # debug=True
    )
    mr = sp.Simplex_sqrt_Shooting(
        source.clone(),
        target.clone(),
        mp,
        cost_cst=cost_cst,
        data_term=data_term,
        optimizer_method="LBFGS_torch",
        hamiltonian_integration=ham,
        lbfgs_history_size=lbfgs_history_size,
        lbfgs_max_iter=lbfgs_max_iter,
        adam_scheduler=adam_scheduler,
        adam_grad_clip=adam_grad_clip,
    )
    if safe_mode:
        mr.forward_safe_mode(momentum_ini, n_iter, grad_coef, plot,
                             convergence_tol=convergence_tol, convergence_patience=convergence_patience)
    else:
        mr.forward(momentum_ini, n_iter=n_iter, grad_coef=grad_coef, plot=plot,
                   convergence_tol=convergence_tol, convergence_patience=convergence_patience)
    return mr

@monitor_gpu
def affine_along_metamorphosis(
        source,
        target,
        momenta_ini,
        kernelOperator,
        rho,
        data_term,
        # dx_convention="2square",
        integration_steps=10,
        n_iter=0,
        grad_coef=2,
        cost_cst=0.001,
        cost_field_cst=.5,
        cost_affine_cst=1,
        plot=False,
        safe_mode=False,
        constraints=True,
        optimizer_method="LBFGS_torch",
        hamiltonian_integration=False,
        save_gpu_memory=False,
        lbfgs_max_iter=20,
        lbfgs_history_size=100,
        adam_dt_step_field=1e-3,
        adam_dt_step_affine=5e-2,
        debug=False,
        convergence_tol=None,
        convergence_patience=3,
        adam_scheduler="reduce_on_plateau",
        adam_grad_clip=None,
        freeze_affine_at_handoff=False,
        reset_diffeo_state_at_handoff=False,
        reset_diffeo_values_at_handoff=False,
        log_affine_drift=True,
    ):
    """
    Run metamorphosis with an affine component (full affine or decoupled rigid-like).

    The affine mode is selected from ``momenta_ini`` keys:
    - ``"momentum_A"``: full affine mode via ``aff`` classes.
    - ``"momentum_R"``: decoupled affine/rigid mode via ``ad`` classes. should not be used in this function

    Parameters
    ----------
    source : torch.Tensor
        Source image.
    target : torch.Tensor
        Target image.
    momenta_ini : dict
        Dictionary of initial momenta. Supported keys include
        ``"momentum_I"``, ``"momentum_R"``, ``"momentum_T"``, and ``"momentum_A"``.
        Exactly one of ``"momentum_A"`` or ``"momentum_R"`` must be present.
    kernelOperator : callable
        Reproducing kernel operator used for the deformable field.
    rho : float
        Metamorphosis trade-off parameter.
    data_term : optional
        Data attachment term.
    integration_steps : int, optional
        Number of geodesic integration steps.
    n_iter : int, optional
        Number of optimization iterations.
    grad_coef : float, optional
        Multiplier applied to the optimization step.
    cost_cst : float, optional
        Weight of the regularization term.
    cost_field_cst : float, optional
        Weight of the deformable-field regularization.
    cost_affine_cst : float, optional
        Weight of the affine regularization.
    plot : bool, optional
        Kept for API compatibility.
    safe_mode : bool, optional
        If ``True``, use ``forward_safe_mode`` for optimization.
    constraints : bool, optional
        Enable or disable affine constraints in the integrator.
    optimizer_method : str, optional
        Optimizer used by the affine optimizer class.
    hamiltonian_integration : bool, optional
        Enable Hamiltonian integration in the optimizer.
    save_gpu_memory : bool, optional
        Enable memory-saving mode in the geodesic integrator.
    lbfgs_max_iter : int, optional
        Maximum number of internal iterations for LBFGS steps.
    lbfgs_history_size : int, optional
        History size used by LBFGS.
    adam_dt_step_field : float, optional
        Learning rate for field variables with Adam-like optimizers.
    adam_dt_step_affine : float, optional
        Learning rate for affine variables with Adam-like optimizers.
    debug : bool, optional
        Enable debug mode in integrator/optimizer.
    convergence_tol : float or None, optional
        Early-stopping relative tolerance on the data loss. See
        ``Optimize_geodesicShooting.forward`` for details. Default ``None`` (disabled).
    convergence_patience : int, optional
        Number of consecutive plateau steps before stopping. Default 3.

    Returns
    -------
    object
        The configured affine metamorphosis optimizer instance after optimization.

    Raises
    ------
    ValueError
        If both ``"momentum_A"`` and ``"momentum_R"`` are provided, or if neither
        key is present in ``momenta_ini``.
    """
    if n_iter > 0 and momenta_ini == 0:
        for key in momenta_ini.keys():
            if not key in ['momentum_I', 'momentum_R', 'momentum_T', 'momentum_A']:
                raise ValueError("momenta_ini must be a dictionary containing the keywords:"
                        " - 'momentum_I' for the image"
                        " - 'momentum_A' for a full affine transformation (momentum R) will be ignored"
                        " - 'momentum_T' for the translation"
                                 )

    if not isinstance(momenta_ini, int):
        if getattr(momenta_ini, "momentum_R", None) is not None:
            raise ValueError(f"In affine_along_metamorphosis momenta_ini must not contain 'momentum_R',"
                             f" if you wand to estimate a rotation individually,"
                             f" please use `affine_decoupled_along_metamorphosis` instead.")

    mp = aff.Affine_Metamorphosis_integrator(
        rho=rho,
        n_step=integration_steps,
        kernelOperator=kernelOperator,
        dx_convention="2square",
        debug=debug,
        save_gpu_memory = save_gpu_memory,
        constraints = constraints,
    )

    mr = aff.Affine_Metamorphosis_Optimizer(
        source=source.clone(),
        target=target.clone(),
        geodesic=mp,
        cost_cst=cost_cst,
        cost_field_cst=cost_field_cst,
        cost_affine_cst=cost_affine_cst,
        data_term=data_term,
        hamiltonian_integration=hamiltonian_integration,
        optimizer_method=optimizer_method,
        debug=debug,
        lbfgs_max_iter=lbfgs_max_iter,
        lbfgs_history_size=lbfgs_history_size,
        adam_dt_step_field=adam_dt_step_field,
        adam_dt_step_affine=adam_dt_step_affine,
        adam_scheduler=adam_scheduler,
        adam_grad_clip=adam_grad_clip,
        freeze_affine_at_handoff=freeze_affine_at_handoff,
        reset_diffeo_state_at_handoff=reset_diffeo_state_at_handoff,
        reset_diffeo_values_at_handoff=reset_diffeo_values_at_handoff,
        log_affine_drift=log_affine_drift,
    )
    mr = _commun_after(mr, momenta_ini, safe_mode, n_iter, grad_coef,
                      convergence_tol=convergence_tol, convergence_patience=convergence_patience)

    return mr


@monitor_gpu
def affine_decoupled_along_metamorphosis(
        source,
        target,
        momenta_ini,
        kernelOperator,
        rho,
        data_term,
        # dx_convention="2square",
        integration_steps=10,
        n_iter=0,
        grad_coef=2,
        cost_cst=0.001,
        cost_field_cst=.5,
        cost_affine_cst=1,
        plot=False,
        safe_mode=False,
        constraints=True,
        optimizer_method="LBFGS_torch",
        hamiltonian_integration=False,
        save_gpu_memory=False,
        lbfgs_max_iter=20,
        lbfgs_history_size=100,
        adam_dt_step_field=1e-3,
        adam_dt_step_affine=5e-2,
        debug=False,
        convergence_tol=None,
        convergence_patience=3,
        adam_scheduler="reduce_on_plateau",
        adam_grad_clip=None,
    ):
    """
    Run metamorphosis with an affine component (full affine or decoupled rigid-like).

    The affine mode is selected from ``momenta_ini`` keys:
    - ``"momentum_A"``: full affine mode via ``aff`` classes.
    - ``"momentum_R"``: decoupled affine/rigid mode via ``ad`` classes.

    Parameters
    ----------
    source : torch.Tensor
        Source image.
    target : torch.Tensor
        Target image.
    momenta_ini : dict
        Dictionary of initial momenta. Supported keys include
        ``"momentum_I"``, ``"momentum_R"``, ``"momentum_T"``, and ``"momentum_A"``.
        Exactly one of ``"momentum_A"`` or ``"momentum_R"`` must be present.
    kernelOperator : callable
        Reproducing kernel operator used for the deformable field.
    rho : float
        Metamorphosis trade-off parameter.
    data_term : optional
        Data attachment term.
    integration_steps : int, optional
        Number of geodesic integration steps.
    n_iter : int, optional
        Number of optimization iterations.
    grad_coef : float, optional
        Multiplier applied to the optimization step.
    cost_cst : float, optional
        Weight of the regularization term.
    cost_field_cst : float, optional
        Weight of the deformable-field regularization.
    cost_affine_cst : float, optional
        Weight of the affine regularization.
    plot : bool, optional
        Kept for API compatibility.
    safe_mode : bool, optional
        If ``True``, use ``forward_safe_mode`` for optimization.
    constraints : bool, optional
        Enable or disable affine constraints in the integrator.
    optimizer_method : str, optional
        Optimizer used by the affine optimizer class.
    hamiltonian_integration : bool, optional
        Enable Hamiltonian integration in the optimizer.
    save_gpu_memory : bool, optional
        Enable memory-saving mode in the geodesic integrator.
    lbfgs_max_iter : int, optional
        Maximum number of internal iterations for LBFGS steps.
    lbfgs_history_size : int, optional
        History size used by LBFGS.
    adam_dt_step_field : float, optional
        Learning rate for field variables with Adam-like optimizers.
    adam_dt_step_affine : float, optional
        Learning rate for affine variables with Adam-like optimizers.
    debug : bool, optional
        Enable debug mode in integrator/optimizer.
    convergence_tol : float or None, optional
        Early-stopping relative tolerance on the data loss. See
        ``Optimize_geodesicShooting.forward`` for details. Default ``None`` (disabled).
    convergence_patience : int, optional
        Number of consecutive plateau steps before stopping. Default 3.

    Returns
    -------
    object
        The configured affine metamorphosis optimizer instance after optimization.

    Raises
    ------
    ValueError
        If both ``"momentum_A"`` and ``"momentum_R"`` are provided, or if neither
        key is present in ``momenta_ini``.
    """
    if n_iter > 0 and momenta_ini == 0:
        for key in momenta_ini.keys():
            if not key in ['momentum_I', 'momentum_R', 'momentum_T', 'momentum_S']:
                raise ValueError("momenta_ini must be a dictionary containing the keywords:"
                        " - 'momentum_I' for the image"
                        " - 'momentum_R' for the rotation"
                        " - 'momentum_T' for the translation"
                        "- 'momentum_S' for the scaling"
                                 )

    if getattr(momenta_ini, "momentum_A", None) is not None:
        raise ValueError(f"In affine_along_metamorphosis momenta_ini must not contain 'momentum_A',"
                         f" if you wand to estimate a general affine deformation,"
                         f" please use `affine_along_metamorphosis` instead.")

    mp = ad.Affine_Decoupled_Metamorphosis_integrator(
        rho=rho,
        n_step=integration_steps,
        kernelOperator=kernelOperator,
        dx_convention="2square",
        debug=debug,
        save_gpu_memory = save_gpu_memory,
        constraints = constraints,
    )

    mr = ad.Affine_Decoupled_Metamorphosis_Optimizer(
        source=source.clone(),
        target=target.clone(),
        geodesic=mp,
        cost_cst=cost_cst,
        cost_field_cst=cost_field_cst,
        cost_affine_cst=cost_affine_cst,
        data_term=data_term,
        hamiltonian_integration=hamiltonian_integration,
        optimizer_method=optimizer_method,
        debug=debug,
        lbfgs_max_iter=lbfgs_max_iter,
        lbfgs_history_size=lbfgs_history_size,
        adam_dt_step_field=adam_dt_step_field,
        adam_dt_step_affine=adam_dt_step_affine,
        adam_scheduler=adam_scheduler,
        adam_grad_clip=adam_grad_clip,
    )
    mr = _commun_after(mr, momenta_ini, safe_mode, n_iter, grad_coef,
                      convergence_tol=convergence_tol, convergence_patience=convergence_patience)

    return mr
