import torch


def conjugate_gradient(linear_operator, rhs, tolerance):
    """Solve a symmetric positive-definite linear system."""
    solution = torch.zeros_like(rhs)
    residual = rhs.clone()
    residual_norm_sq = residual.square().sum()
    threshold = tolerance**2 * residual_norm_sq.clamp_min(1)
    if residual_norm_sq <= threshold:
        return solution, 0

    direction = residual.clone()
    for iteration in range(1, rhs.numel() + 1):
        applied = linear_operator(direction)
        step = residual_norm_sq / (direction * applied).sum()
        solution += step * direction
        residual -= step * applied
        next_residual_norm_sq = residual.square().sum()
        if next_residual_norm_sq <= threshold:
            return solution, iteration
        direction = residual + (next_residual_norm_sq / residual_norm_sq) * direction
        residual_norm_sq = next_residual_norm_sq

    raise RuntimeError("conjugate gradient did not converge")
