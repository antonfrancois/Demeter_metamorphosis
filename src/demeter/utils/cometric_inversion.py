from math import isfinite
from time import perf_counter

import torch

from . import torchbox as tb
from .conjugate_gradient import conjugate_gradient, jacobi_preconditioner


_CG_CONVERGENCE_CHECK_INTERVAL = 2


def _apply_cometric(image_gradient, covector, rho, kernel_operator):
    if rho == 0:
        return covector

    vector_momentum = covector * image_gradient[:, 0]
    velocity = kernel_operator(vector_momentum)
    deformation = (velocity * image_gradient[:, 0]).sum(dim=1, keepdim=True)
    return torch.lerp(covector, deformation, rho)


def _jacobi_preconditioner(image_gradient, rho, kernel_operator):
    kernel_at_zero = getattr(kernel_operator, "inverse_kernel_at_zero", None)
    if kernel_at_zero is None:
        return None

    block = kernel_at_zero(image_gradient[:, 0])
    gradient_x, gradient_y = image_gradient[:, 0].unbind(dim=1)
    diagonal = (
        block[0, 0] * gradient_x.square()
        + (block[0, 1] + block[1, 0]) * gradient_x * gradient_y
        + block[1, 1] * gradient_y.square()
    ).unsqueeze(1)
    diagonal.mul_(rho).add_(1 - rho)
    return jacobi_preconditioner(diagonal)


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
        preconditioner=_jacobi_preconditioner(
            image_gradient, rho, kernel_operator
        ),
        return_residual=stats is not None,
        convergence_check_interval=_CG_CONVERGENCE_CHECK_INTERVAL,
    )
    if stats is not None:
        if image_gradient.is_cuda:
            torch.cuda.synchronize(image_gradient.device)
        assert residual is not None
        stats["residual"] = residual
        stats["iterations"] = iterations
        stats["elapsed_seconds"] = perf_counter() - start
    return solution


class _CometricInverse(torch.autograd.Function):
    """Implicit backward with extrapolated adjacent warm starts around CG."""

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
        adjoint_warm_starts,
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
        ctx.adjoint_warm_starts = adjoint_warm_starts
        if adjoint_warm_starts is not None:
            ctx.adjoint_index = len(adjoint_warm_starts)
            adjoint_warm_starts.append(None)
        return solution

    @staticmethod
    def backward(ctx, grad_output):
        image_gradient, solution = ctx.saved_tensors
        adjoint_x_0 = None
        next_adjoint = None
        if ctx.adjoint_warm_starts is not None:
            entry = ctx.adjoint_warm_starts[ctx.adjoint_index]
            ctx.adjoint_warm_starts[ctx.adjoint_index] = None
            if entry is not None:
                adjoint_x_0, next_adjoint = entry
        with torch.no_grad():
            adjoint = _solve(
                image_gradient,
                grad_output,
                ctx.rho,
                ctx.kernel_operator,
                ctx.eps,
                x_0=adjoint_x_0,
            )
        if ctx.adjoint_warm_starts is not None and ctx.adjoint_index > 0:
            current_adjoint = adjoint.detach()
            extrapolated = (
                current_adjoint
                if next_adjoint is None
                else 2 * current_adjoint - next_adjoint
            )
            ctx.adjoint_warm_starts[ctx.adjoint_index - 1] = (
                extrapolated,
                current_adjoint,
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
        return (
            grad_image_gradient,
            grad_acceleration,
            None,
            None,
            None,
            None,
            None,
            None,
        )


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

    def inverse(
        self,
        acceleration,
        eps=1e-6,
        return_info=False,
        *,
        x_0=None,
        _adjoint_warm_starts=None,
    ):
        """Solve ``A_I u = a`` with conjugate gradients.

        The optional residual is ``||a - A_I u|| / ||a||`` for nonzero ``a``.
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
            _adjoint_warm_starts,
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
