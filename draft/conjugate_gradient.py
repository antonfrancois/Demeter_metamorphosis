import torch


def conjugate_gradient(linear_operator, rhs, tolerance):
    """Solve an SPD system and return its mixed relative residual."""
    solution = torch.zeros_like(rhs)
    residual = rhs.clone()
    residual_norm_sq = residual.square().sum()
    normalization = residual_norm_sq.clamp_min(1).sqrt()
    threshold = (tolerance * normalization).square()
    if residual_norm_sq <= threshold:
        return solution, 0, float(residual_norm_sq.sqrt() / normalization)

    direction = residual.clone()
    for iteration in range(1, rhs.numel() + 1):
        applied = linear_operator(direction)
        step = residual_norm_sq / (direction * applied).sum()
        solution += step * direction
        residual -= step * applied
        next_residual_norm_sq = residual.square().sum()
        if next_residual_norm_sq <= threshold:
            relative_residual = next_residual_norm_sq.sqrt() / normalization
            return solution, iteration, float(relative_residual)
        direction = residual + (next_residual_norm_sq / residual_norm_sq) * direction
        residual_norm_sq = next_residual_norm_sq

    raise RuntimeError("conjugate gradient did not converge")
