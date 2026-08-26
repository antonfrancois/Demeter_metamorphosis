from math import isfinite

import torch


def _validate_positive_integer(value, name):
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a strictly positive integer")


def _precondition_residual(residual, residual_norm_sq, preconditioner):
    if preconditioner is None:
        return residual, residual_norm_sq
    preconditioned = preconditioner(residual)
    return preconditioned, (residual * preconditioned).sum()


def _optional_residual(value, return_residual):
    return float(value) if return_residual else None


def jacobi_preconditioner(diagonal):
    """Return a callable that applies the inverse of ``diagonal``."""
    inverse_diagonal = diagonal.reciprocal()
    return inverse_diagonal.mul


def conjugate_gradient(
    linear_operator,
    rhs,
    tolerance,
    max_iterations=None,
    *,
    x_0=None,
    preconditioner=None,
    return_residual=True,
    convergence_check_interval=1,
):
    """Solve an SPD system, optionally checking convergence in intervals."""
    tolerance = float(tolerance)
    if not isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and strictly positive")
    if max_iterations is None:
        max_iterations = max(64, min(4 * rhs.numel(), 10_000))
    _validate_positive_integer(max_iterations, "max_iterations")
    _validate_positive_integer(
        convergence_check_interval,
        "convergence_check_interval",
    )

    rhs_norm = torch.linalg.vector_norm(rhs)
    if rhs_norm == 0:
        return torch.zeros_like(rhs), 0, _optional_residual(0.0, return_residual)
    normalization = rhs_norm
    scaled_rhs = rhs / normalization
    if x_0 is None:
        solution = torch.zeros_like(rhs)
        residual = scaled_rhs.clone()
    else:
        solution = x_0 / normalization
        residual = scaled_rhs - linear_operator(solution)
    residual_norm_sq = residual.square().sum()
    threshold = tolerance**2
    if residual_norm_sq <= threshold:
        return (
            solution * normalization,
            0,
            _optional_residual(residual_norm_sq.sqrt(), return_residual),
        )

    preconditioned_residual, residual_product = _precondition_residual(
        residual,
        residual_norm_sq,
        preconditioner,
    )
    direction = preconditioned_residual.clone()
    for iteration in range(1, max_iterations + 1):
        applied = linear_operator(direction)
        curvature = (direction * applied).sum()
        previous_iteration_checked = (
            iteration == 1
            or (iteration - 1) % convergence_check_interval == 0
        )
        if previous_iteration_checked:
            step = residual_product / curvature
        else:
            # Exact convergence may be hidden until the next scheduled check.
            step = torch.where(
                curvature != 0,
                residual_product / curvature,
                torch.zeros_like(curvature),
            )
        solution += step * direction
        residual -= step * applied
        check_convergence = (
            iteration % convergence_check_interval == 0
            or iteration == max_iterations
        )
        next_residual_norm_sq = (
            residual.square().sum()
            if preconditioner is None or check_convergence
            else None
        )
        if check_convergence:
            assert next_residual_norm_sq is not None
            if next_residual_norm_sq <= threshold:
                candidate_solution = solution * normalization
                true_residual = (
                    rhs - linear_operator(candidate_solution)
                ) / normalization
                true_residual_norm_sq = true_residual.square().sum()
                if true_residual_norm_sq <= threshold:
                    return (
                        candidate_solution,
                        iteration,
                        _optional_residual(
                            true_residual_norm_sq.sqrt(),
                            return_residual,
                        ),
                    )
                residual = true_residual
                residual_norm_sq = true_residual_norm_sq
                preconditioned_residual, residual_product = _precondition_residual(
                    residual,
                    residual_norm_sq,
                    preconditioner,
                )
                direction = preconditioned_residual.clone()
                continue
        if preconditioner is None:
            preconditioned_residual = residual
            assert next_residual_norm_sq is not None
            next_residual_product = next_residual_norm_sq
        else:
            preconditioned_residual = preconditioner(residual)
            next_residual_product = (residual * preconditioned_residual).sum()
        if previous_iteration_checked or check_convergence:
            beta = next_residual_product / residual_product
        else:
            beta = torch.where(
                residual_product != 0,
                next_residual_product / residual_product,
                torch.zeros_like(residual_product),
            )
        direction = preconditioned_residual + beta * direction
        if next_residual_norm_sq is not None:
            residual_norm_sq = next_residual_norm_sq
        residual_product = next_residual_product

    true_residual = (
        rhs - linear_operator(solution * normalization)
    ) / normalization
    relative_residual = torch.linalg.vector_norm(true_residual)
    raise RuntimeError(
        "conjugate gradient did not converge within "
        f"{max_iterations} iterations (relative residual={float(relative_residual):.3e})"
    )
