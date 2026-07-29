from math import isfinite

import torch


def conjugate_gradient(linear_operator, rhs, tolerance, max_iterations=None):
    """Solve an SPD system and return its true mixed relative residual."""
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
    if not torch.isfinite(rhs_norm):
        raise ValueError("rhs must contain only finite values with a finite norm")
    normalization = rhs_norm.clamp_min(1)
    scaled_rhs = rhs / normalization
    solution = torch.zeros_like(rhs)
    residual = scaled_rhs.clone()
    residual_norm_sq = residual.square().sum()
    threshold = tolerance**2
    if residual_norm_sq <= threshold:
        return solution, 0, float(residual_norm_sq.sqrt())

    direction = residual.clone()
    for iteration in range(1, max_iterations + 1):
        applied = linear_operator(direction)
        if applied.shape != direction.shape:
            raise ValueError("linear_operator must preserve the rhs shape")
        curvature = (direction * applied).sum()
        if not torch.isfinite(curvature) or curvature <= 0:
            raise RuntimeError(
                "conjugate gradient encountered non-positive or non-finite curvature"
            )
        step = residual_norm_sq / curvature
        solution += step * direction
        residual -= step * applied
        next_residual_norm_sq = residual.square().sum()
        if not torch.isfinite(next_residual_norm_sq):
            raise RuntimeError("conjugate gradient produced a non-finite residual")
        if next_residual_norm_sq <= threshold:
            candidate_solution = solution * normalization
            true_residual = (
                rhs - linear_operator(candidate_solution)
            ) / normalization
            true_residual_norm_sq = true_residual.square().sum()
            if not torch.isfinite(true_residual_norm_sq):
                raise RuntimeError("conjugate gradient produced a non-finite residual")
            if true_residual_norm_sq <= threshold:
                return (
                    candidate_solution,
                    iteration,
                    float(true_residual_norm_sq.sqrt()),
                )
            residual = true_residual
            direction = residual.clone()
            residual_norm_sq = true_residual_norm_sq
            continue
        if iteration % 50 == 0:
            residual = (
                rhs - linear_operator(solution * normalization)
            ) / normalization
            next_residual_norm_sq = residual.square().sum()
            if not torch.isfinite(next_residual_norm_sq):
                raise RuntimeError("conjugate gradient produced a non-finite residual")
            direction = residual.clone()
            residual_norm_sq = next_residual_norm_sq
            continue
        direction = residual + (next_residual_norm_sq / residual_norm_sq) * direction
        residual_norm_sq = next_residual_norm_sq

    true_residual = (
        rhs - linear_operator(solution * normalization)
    ) / normalization
    relative_residual = torch.linalg.vector_norm(true_residual)
    raise RuntimeError(
        "conjugate gradient did not converge within "
        f"{max_iterations} iterations (relative residual={float(relative_residual):.3e})"
    )
