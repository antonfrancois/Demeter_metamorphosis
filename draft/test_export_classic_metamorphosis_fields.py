import json
from types import SimpleNamespace

import torch

from draft.export_classic_metamorphosis_fields import (
    build_kernel_operator,
    extract_trajectory,
    load_image,
    resize_target_to_source,
    run_registration,
    save_trajectory,
)
from demeter.utils.cometric_inversion import CometricOperator
from draft.playground.field_playground_core import load_field_file
from draft.playground.compare_spline_geodesic import load_comparison_input
from draft.playground.splines.core import (
    SplineParameters,
    load_scalar_field,
    load_setup,
)
from demeter.metamorphosis.var_classes import Momenta
from demeter.utils.reproducing_kernels import GaussianRKHS, SobolevFluidOperator


def test_resize_target_to_source_preserves_source_resolution():
    source = torch.rand(1, 1, 8, 10)
    target = torch.rand(1, 1, 14, 12)

    resized = resize_target_to_source(source, target)

    assert resized.shape == source.shape
    assert source.shape == (1, 1, 8, 10)
    assert resized.is_contiguous()


def test_exporter_builds_periodic_gaussian_or_sobolev_kernel():
    gaussian = build_kernel_operator(
        "gaussian",
        sigma=(1.5, 2.0),
        kernel_reach=2,
    )
    assert isinstance(gaussian, GaussianRKHS)
    assert gaussian.border_type == "circular"
    assert gaussian.sigma == (1.5, 2.0)
    field = torch.randn(1, 2, 12, 13)
    shift = (2, -3)
    torch.testing.assert_close(
        gaussian(torch.roll(field, shift, dims=(-2, -1))),
        torch.roll(gaussian(field), shift, dims=(-2, -1)),
        atol=1e-6,
        rtol=1e-6,
    )

    sobolev = build_kernel_operator(
        "sobolev",
        alpha=0.3,
        beta=0.4,
        gamma=0.2,
    )
    assert isinstance(sobolev, SobolevFluidOperator)
    assert sobolev.boundary == "periodic"
    assert (sobolev.alpha, sobolev.beta, sobolev.gamma) == (0.3, 0.4, 0.2)


def test_extract_trajectory_includes_zero_and_endpoint_states():
    torch.manual_seed(7)
    steps, height, width = 3, 8, 10
    source = torch.rand(1, 1, height, width)
    initial_momentum = torch.rand_like(source)
    image_stock = torch.rand(steps, 1, height, width)
    momentum_stock = torch.rand(steps, 1, height, width)
    registration = SimpleNamespace(
        source=source,
        optimized_momenta=Momenta(momentum_I=initial_momentum),
        mp=SimpleNamespace(
            n_step=steps,
            image_stock=image_stock,
            momentum_stock=[
                Momenta(momentum_I=momentum_stock[index : index + 1])
                for index in range(steps)
            ],
        ),
    )
    operator = SobolevFluidOperator(alpha=0.2, beta=0.2, gamma=0.1)

    trajectory = extract_trajectory(registration, operator, rho=0.4)

    torch.testing.assert_close(
        trajectory["times"], torch.tensor([0.0, 1 / 3, 2 / 3, 1.0])
    )
    torch.testing.assert_close(trajectory["images"][0:1], source)
    torch.testing.assert_close(
        trajectory["image_momenta"][0:1], initial_momentum
    )
    torch.testing.assert_close(trajectory["images"][1:], image_stock)
    torch.testing.assert_close(
        trajectory["image_momenta"][1:], momentum_stock
    )
    torch.testing.assert_close(
        operator.apply_operator(trajectory["velocities"]),
        trajectory["vector_momenta"],
        rtol=2e-4,
        atol=2e-5,
    )
    expected_image_velocities = CometricOperator(
        trajectory["images"], 0.4, operator, gradient_boundary="periodic"
    )(trajectory["image_momenta"])
    torch.testing.assert_close(
        trajectory["image_velocities"], expected_image_velocities
    )


def test_saved_frames_are_loadable_by_playground(tmp_path):
    steps, height, width = 2, 6, 7
    target = torch.rand(1, 1, height, width)
    trajectory = {
        "times": torch.linspace(0, 1, steps + 1),
        "images": torch.rand(steps + 1, 1, height, width),
        "image_momenta": torch.rand(steps + 1, 1, height, width),
        "image_velocities": torch.rand(steps + 1, 1, height, width),
        "vector_momenta": torch.rand(steps + 1, 2, height, width),
        "velocities": torch.rand(steps + 1, 2, height, width),
    }
    parameters = {
        "kernel": "sobolev",
        "boundary": "periodic",
        "rho": 0.4,
        "alpha": 0.2,
        "beta": 0.2,
        "gamma": 0.001,
        "cg_eps": 1e-5,
        "integration_steps": steps,
    }
    spline_parameters = SplineParameters(
        rho=0.4,
        alpha=0.2,
        beta=0.2,
        gamma=0.001,
        n_steps=steps,
        control_steps=(),
    )

    output = save_trajectory(
        trajectory,
        tmp_path / "fields",
        source_path=tmp_path / "source.png",
        target_path=tmp_path / "target.png",
        target_image=target,
        parameters=parameters,
        spline_parameters=spline_parameters,
    )

    velocity = load_field_file(output / "vector/velocity/velocity_t000.pt")
    momentum = load_field_file(output / "vector/momentum/momentum_t002.pt")
    image_momentum = load_field_file(
        output / "scalar/momentum/image_momentum_t001.pt"
    )
    image_velocity = load_field_file(
        output / "scalar/velocity/image_velocity_t002.pt"
    )
    assert velocity.kind == "velocity"
    assert momentum.kind == "vector_momentum"
    assert image_momentum.kind == "u"
    assert image_velocity.kind == "a"
    torch.testing.assert_close(velocity.field, trajectory["velocities"][0:1])
    torch.testing.assert_close(momentum.field, trajectory["vector_momenta"][2:3])
    torch.testing.assert_close(
        image_momentum.field, trajectory["image_momenta"][1:2]
    )
    torch.testing.assert_close(
        image_velocity.field, trajectory["image_velocities"][2:3]
    )
    assert velocity.metadata["time"] == 0
    assert momentum.metadata["time"] == 1
    assert image_momentum.metadata["time"] == 0.5
    assert image_momentum.metadata["field_role"] == "image_momentum"
    assert image_velocity.metadata["field_role"] == "image_velocity"
    assert velocity.metadata["parameters"] == parameters
    assert (output / "trajectory.pt").is_file()
    assert (output / "manifest.json").is_file()
    initial_momentum = load_scalar_field(
        output / "scalar/momentum/image_momentum_t000.pt",
        (height, width),
    )
    torch.testing.assert_close(initial_momentum, trajectory["image_momenta"][0:1])
    setup = load_setup(output / "spline_setup.pt")
    torch.testing.assert_close(setup.source, trajectory["images"][0:1])
    torch.testing.assert_close(setup.target, target)
    torch.testing.assert_close(
        setup.initial_momentum,
        trajectory["image_momenta"][0:1],
    )
    assert torch.count_nonzero(setup.initial_force) == 0
    assert torch.count_nonzero(setup.initial_jerk) == 0
    assert setup.control_jerks.shape[0] == 0
    assert setup.parameters == spline_parameters
    comparison_inputs = (
        output,
        output / "manifest.json",
        output / "spline_setup.pt",
        output / "scalar/momentum/image_momentum_t000.pt",
    )
    for comparison_input in comparison_inputs:
        comparison_source, comparison_momentum, comparison_parameters = (
            load_comparison_input(
                comparison_input,
                source_path=None,
                rho=None,
                alpha=None,
                beta=None,
                gamma=None,
                cg_eps=None,
                steps=None,
            )
        )
        torch.testing.assert_close(comparison_source, trajectory["images"][0:1])
        torch.testing.assert_close(
            comparison_momentum,
            trajectory["image_momenta"][0:1],
        )
        assert comparison_parameters == spline_parameters
    saved_source = load_image(output / "images/source.png")
    saved_target = load_image(output / "images/target.png")
    saved_final = load_image(output / "images/final.png")
    torch.testing.assert_close(
        saved_source, trajectory["images"][0:1], rtol=0, atol=2 / 255
    )
    torch.testing.assert_close(saved_target, target, rtol=0, atol=2 / 255)
    torch.testing.assert_close(
        saved_final, trajectory["images"][-1:], rtol=0, atol=2 / 255
    )
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["images"] == {
        "source": "images/source.png",
        "target": "images/target.png",
        "final": "images/final.png",
    }
    assert manifest["initial_momentum"] == (
        "scalar/momentum/image_momentum_t000.pt"
    )
    assert manifest["spline_setup"] == "spline_setup.pt"
    assert manifest["frames"][1]["image_momentum"] == (
        "scalar/momentum/image_momentum_t001.pt"
    )
    assert manifest["frames"][2]["image_velocity"] == (
        "scalar/velocity/image_velocity_t002.pt"
    )


def test_periodic_registration_runs_with_both_kernel_choices():
    torch.manual_seed(3)
    source = torch.rand(1, 1, 8, 9)
    target = torch.roll(source, shifts=1, dims=-1)
    operators = (
        build_kernel_operator(
            "gaussian",
            sigma=(1.0, 1.0),
            kernel_reach=1,
        ),
        build_kernel_operator(
            "sobolev",
            alpha=0.2,
            beta=0.2,
            gamma=0.3,
        ),
    )

    for operator in operators:
        registration = run_registration(
            source,
            target,
            rho=0.25,
            operator=operator,
            integration_steps=2,
            iterations=1,
            cost_cst=1e-3,
            grad_coef=1.0,
            device=torch.device("cpu"),
        )

        assert registration.mp.boundary == "periodic"
        assert isinstance(registration.optimized_momenta, Momenta)
        assert all(
            isinstance(momentum, Momenta)
            for momentum in registration.mp.momentum_stock
        )
        trajectory = extract_trajectory(registration, operator, rho=0.25)
        assert trajectory["images"].shape == (3, 1, 8, 9)
        assert trajectory["image_momenta"].shape == (3, 1, 8, 9)
