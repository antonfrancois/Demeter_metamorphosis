from math import isfinite
from time import perf_counter

import torch

from . import torchbox as tb
from .conjugate_gradient import conjugate_gradient


def _apply_cometric(image_gradient, covector, rho, kernel_operator):
    if rho == 0:
        return covector

    vector_momentum = covector * image_gradient[:, 0]
    velocity = kernel_operator(vector_momentum)
    deformation = (velocity * image_gradient[:, 0]).sum(dim=1, keepdim=True)
    return (1 - rho) * covector + rho * deformation


def _solve(
    image_gradient,
    acceleration,
    rho,
    kernel_operator,
    eps,
    stats=None,
    x_0=None,
):
    def linear_operator(covector):
        return _apply_cometric(image_gradient, covector, rho, kernel_operator)

    start = 0.0
    if stats is not None:
        if image_gradient.is_cuda:
            torch.cuda.synchronize(image_gradient.device)
        start = perf_counter()
    solution, iterations, residual = conjugate_gradient(
        linear_operator,
        acceleration,
        eps,
        x_0=x_0,
    )
    if stats is not None:
        if image_gradient.is_cuda:
            torch.cuda.synchronize(image_gradient.device)
        stats["residual"] = residual
        stats["iterations"] = iterations
        stats["elapsed_seconds"] = perf_counter() - start
    return solution


class _CometricInverse(torch.autograd.Function):
    """Implicit backward around the in-place CG implementation."""

    @staticmethod
    def forward(
        ctx,
        image_gradient,
        acceleration,
        rho,
        kernel_operator,
        eps,
        stats,
        x_0,
    ):
        solution = _solve(
            image_gradient,
            acceleration,
            rho,
            kernel_operator,
            eps,
            stats,
            x_0,
        )
        ctx.save_for_backward(image_gradient, solution)
        ctx.rho = rho
        ctx.kernel_operator = kernel_operator
        ctx.eps = eps
        return solution

    @staticmethod
    def backward(ctx, grad_output):
        image_gradient, solution = ctx.saved_tensors
        with torch.no_grad():
            adjoint = _solve(
                image_gradient,
                grad_output,
                ctx.rho,
                ctx.kernel_operator,
                ctx.eps,
            )

        grad_image_gradient = None
        if ctx.needs_input_grad[0]:
            with torch.enable_grad():
                image_gradient_var = image_gradient.detach().requires_grad_(True)
                applied = _apply_cometric(
                    image_gradient_var,
                    solution,
                    ctx.rho,
                    ctx.kernel_operator,
                )
                grad_image_gradient = -torch.autograd.grad(
                    applied, image_gradient_var, adjoint
                )[0]

        grad_acceleration = adjoint if ctx.needs_input_grad[1] else None
        return grad_image_gradient, grad_acceleration, None, None, None, None, None


class CometricOperator:
    r"""Image-dependent cometric with a cached periodic spatial gradient.

    .. math::
        A_I u = (1-\rho)u + \rho\,K(u\nabla I)\cdot\nabla I.
    """

    def __init__(
        self,
        image,
        rho,
        kernel_operator,
        dx_convention="pixel",
        gradient_boundary="periodic",
    ):
        self.rho = float(rho)
        if not 0 <= self.rho <= 1:
            raise ValueError("rho must be in [0, 1]")

        self.kernel_operator = kernel_operator
        self.image_gradient = tb.spatialGradient(
            image,
            dx_convention=dx_convention,
            boundary=gradient_boundary,
        )

    def __call__(self, covector):
        return _apply_cometric(
            self.image_gradient, covector, self.rho, self.kernel_operator
        )

    def inverse(self, acceleration, eps=1e-6, return_info=False, *, x_0=None):
        """Solve ``A_I u = a`` with conjugate gradients.

        The optional residual is ``||a - A_I u|| / max(||a||, 1)``.
        ``x_0`` is a non-differentiable initial guess in physical force units.
        """
        if self.rho == 0:
            if return_info:
                return acceleration, 0, 0.0, 0.0
            return acceleration
        if self.rho == 1:
            raise ValueError(
                "rho must be smaller than 1 for this positive-definite CG solve"
            )

        eps = float(eps)
        if not isfinite(eps) or eps <= 0:
            raise ValueError("eps must be finite and strictly positive")

        stats = {} if return_info else None
        solution = _CometricInverse.apply(
            self.image_gradient,
            acceleration,
            self.rho,
            self.kernel_operator,
            eps,
            stats,
            x_0,
        )
        if return_info:
            assert stats is not None
            return (
                solution,
                stats["iterations"],
                stats["elapsed_seconds"],
                stats["residual"],
            )
        return solution
