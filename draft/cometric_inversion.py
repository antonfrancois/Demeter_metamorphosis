import torch

from demeter.utils import torchbox as tb


def _apply_cometric(grad_image, covector, rho, kernel_operator):
    vector_momentum = (covector.unsqueeze(2) * grad_image).sum(dim=1)
    velocity = kernel_operator(vector_momentum)
    deformation = (velocity.unsqueeze(1) * grad_image).sum(dim=2)
    return (1 - rho) * covector + rho * deformation


def _solve(image, acceleration, rho, kernel_operator, eps, dx_convention):
    from pykeops.common.operations import ConjugateGradientSolver

    grad_image = tb.spatialGradient(image, dx_convention=dx_convention)

    def apply_cometric(covector):
        return _apply_cometric(grad_image, covector, rho, kernel_operator)

    return ConjugateGradientSolver("torch", apply_cometric, acceleration, eps=eps)


def apply_cometric(
    image,
    covector,
    rho,
    kernel_operator,
    dx_convention="pixel",
):
    r"""Apply ``A_I`` to a scalar covector shaped ``[B, 1, H, W]``.

    .. math::
        A_I u = (1-\rho)u + \rho\,K(u\nabla I)\cdot\nabla I.
    """
    rho = float(rho)
    if image.ndim != 4 or image.shape[1] != 1 or image.shape != covector.shape:
        raise ValueError("image and covector must have shape [B, 1, H, W]")
    if not torch.is_floating_point(image) or not torch.is_floating_point(covector):
        raise TypeError("image and covector must be floating point")
    if not torch.isfinite(image).all() or not torch.isfinite(covector).all():
        raise ValueError("image and covector must contain only finite values")
    if not 0 <= rho <= 1:
        raise ValueError("rho must be in [0, 1]")
    if rho == 0:
        return covector

    grad_image = tb.spatialGradient(image, dx_convention=dx_convention)
    return _apply_cometric(grad_image, covector, rho, kernel_operator)


class _CometricInverse(torch.autograd.Function):
    """Implicit backward around KeOps' in-place generic CG implementation."""

    @staticmethod
    def forward(ctx, image, acceleration, rho, kernel_operator, eps, dx_convention):
        solution = _solve(
            image, acceleration, rho, kernel_operator, eps, dx_convention
        )
        ctx.save_for_backward(image, solution)
        ctx.rho = rho
        ctx.kernel_operator = kernel_operator
        ctx.eps = eps
        ctx.dx_convention = dx_convention
        return solution

    @staticmethod
    def backward(ctx, grad_output):
        image, solution = ctx.saved_tensors
        with torch.no_grad():
            adjoint = _solve(
                image,
                grad_output,
                ctx.rho,
                ctx.kernel_operator,
                ctx.eps,
                ctx.dx_convention,
            )

        grad_image = None
        if ctx.needs_input_grad[0]:
            with torch.enable_grad():
                image_var = image.detach().requires_grad_(True)
                grad_image = tb.spatialGradient(
                    image_var, dx_convention=ctx.dx_convention
                )
                applied = _apply_cometric(
                    grad_image,
                    solution,
                    ctx.rho,
                    ctx.kernel_operator,
                )
                grad_image = -torch.autograd.grad(applied, image_var, adjoint)[0]

        grad_acceleration = adjoint if ctx.needs_input_grad[1] else None
        return grad_image, grad_acceleration, None, None, None, None


def invert_cometric(
    image,
    acceleration,
    rho,
    kernel_operator,
    eps=1e-6,
    dx_convention="pixel",
):
    r"""Solve ``A_I u = a`` for scalar fields shaped ``[B, 1, H, W]``.

    .. math::
        A_I u = (1-\rho)u + \rho\,K(u\nabla I)\cdot\nabla I.
    """
    rho = float(rho)
    eps = float(eps)
    if image.ndim != 4 or image.shape[1] != 1 or image.shape != acceleration.shape:
        raise ValueError("image and acceleration must have shape [B, 1, H, W]")
    if not torch.is_floating_point(image) or not torch.is_floating_point(acceleration):
        raise TypeError("image and acceleration must be floating point")
    if not torch.isfinite(image).all() or not torch.isfinite(acceleration).all():
        raise ValueError("image and acceleration must contain only finite values")
    if not 0 <= rho < 1:
        raise ValueError("rho must be in [0, 1) for this positive-definite CG solve")
    if not torch.isfinite(torch.tensor(eps)) or eps <= 0:
        raise ValueError("eps must be finite and strictly positive")
    if rho == 0:
        return acceleration

    return _CometricInverse.apply(
        image, acceleration, rho, kernel_operator, eps, dx_convention
    )
