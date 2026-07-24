import torch

from draft.playground.compare_spline_geodesic import (
    compare_geodesics,
    run_classical_geodesic,
)
from draft.playground.splines.core import SplineParameters


def test_rho_zero_classical_geodesic_and_spline_are_identical():
    torch.manual_seed(21)
    source = torch.rand(1, 1, 7, 9)
    momentum = 0.01 * torch.randn_like(source)
    comparison = compare_geodesics(
        source,
        momentum,
        SplineParameters(rho=0, n_steps=4, control_steps=()),
        device="cpu",
    )

    assert comparison["metrics"]["image"]["maximum_absolute"] < 1e-7
    for quantity in ("momentum", "velocity"):
        assert comparison["metrics"][quantity]["maximum_absolute"] == 0
    assert comparison["metrics"]["geodesic_invariant"] == {
        "maximum_force": 0.0,
        "maximum_acceleration": 0.0,
        "maximum_jerk": 0.0,
    }


def test_periodic_nonzero_geodesic_matches_zero_jerk_spline_numerically():
    torch.manual_seed(22)
    source = torch.rand(1, 1, 8, 9)
    momentum = 0.1 * torch.randn_like(source)
    comparison = compare_geodesics(
        source,
        momentum,
        SplineParameters(
            rho=0.25,
            alpha=0.2,
            beta=0.2,
            gamma=0.3,
            n_steps=4,
            control_steps=(),
        ),
        device="cpu",
    )

    assert comparison["metrics"]["image"]["maximum_absolute"] < 1e-6
    assert comparison["metrics"]["momentum"]["maximum_absolute"] < 1e-6
    assert comparison["metrics"]["velocity"]["maximum_absolute"] < 1e-6
    assert comparison["metrics"]["geodesic_invariant"] == {
        "maximum_force": 0.0,
        "maximum_acceleration": 0.0,
        "maximum_jerk": 0.0,
    }


def test_periodic_classical_geodesic_is_integer_shift_equivariant():
    torch.manual_seed(23)
    source = torch.rand(1, 1, 7, 9)
    momentum = 1e-3 * torch.randn_like(source)
    parameters = SplineParameters(
        rho=0.25,
        alpha=0.2,
        beta=0.2,
        gamma=0.3,
        n_steps=3,
        control_steps=(),
    )
    shift = (2, -3)
    trajectory = run_classical_geodesic(
        source,
        momentum,
        parameters,
        device="cpu",
    )
    shifted = run_classical_geodesic(
        torch.roll(source, shift, dims=(-2, -1)),
        torch.roll(momentum, shift, dims=(-2, -1)),
        parameters,
        device="cpu",
    )

    for quantity in ("images", "momentum", "velocity"):
        torch.testing.assert_close(
            shifted[quantity],
            torch.roll(trajectory[quantity], shift, dims=(-2, -1)),
            atol=2e-6,
            rtol=1e-5,
        )
