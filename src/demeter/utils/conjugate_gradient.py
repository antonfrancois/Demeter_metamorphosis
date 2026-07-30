from math import isfinite

import torch


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
):
    """Solve an SPD system, optionally using an inverse preconditioner."""
    tolerance = float(tolerance)
    if not isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and strictly positive")
    if max_iterations is None:
        max_iterations = max(64, min(4 * rhs.numel(), 10_000))
    if (
        not isinstance(max_iterations, int)
        or isinstance(max_iterations, bool)
        or max_iterations < 1
    ):
        raise ValueError("max_iterations must be a strictly positive integer")

    rhs_norm = torch.linalg.vector_norm(rhs)
    normalization = rhs_norm.clamp_min(1)
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
        residual_value = (
            float(residual_norm_sq.sqrt()) if return_residual else None
        )
        return solution * normalization, 0, residual_value

    if preconditioner is None:
        preconditioned_residual = residual
        residual_product = residual_norm_sq
    else:
        preconditioned_residual = preconditioner(residual)
        residual_product = (residual * preconditioned_residual).sum()
    direction = preconditioned_residual.clone()
    for iteration in range(1, max_iterations + 1):
        applied = linear_operator(direction)
        curvature = (direction * applied).sum()
        step = residual_product / curvature
        solution += step * direction
        residual -= step * applied
        next_residual_norm_sq = residual.square().sum()
        if next_residual_norm_sq <= threshold:
            candidate_solution = solution * normalization
            true_residual = (
                rhs - linear_operator(candidate_solution)
            ) / normalization
            true_residual_norm_sq = true_residual.square().sum()
            if true_residual_norm_sq <= threshold:
                residual_value = (
                    float(true_residual_norm_sq.sqrt())
                    if return_residual
                    else None
                )
                return (
                    candidate_solution,
                    iteration,
                    residual_value,
                )
            residual = true_residual
            residual_norm_sq = true_residual_norm_sq
            if preconditioner is None:
                preconditioned_residual = residual
                residual_product = residual_norm_sq
            else:
                preconditioned_residual = preconditioner(residual)
                residual_product = (residual * preconditioned_residual).sum()
            direction = preconditioned_residual.clone()
            continue
        if preconditioner is None:
            preconditioned_residual = residual
            next_residual_product = next_residual_norm_sq
        else:
            preconditioned_residual = preconditioner(residual)
            next_residual_product = (residual * preconditioned_residual).sum()
        direction = preconditioned_residual + (
            next_residual_product / residual_product
        ) * direction
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
