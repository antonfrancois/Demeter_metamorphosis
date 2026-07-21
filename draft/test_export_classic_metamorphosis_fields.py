import json
from types import SimpleNamespace

import torch

from draft.export_classic_metamorphosis_fields import (
    extract_trajectory,
    load_image,
    resize_target_to_source,
    save_trajectory,
)
from draft.cometric_inversion import CometricOperator
from draft.playground.field_playground_core import load_field_file
from draft.sobolevfluid_operator import SobolevFluidOperator


def test_resize_target_to_source_preserves_source_resolution():
    source = torch.rand(1, 1, 8, 10)
    target = torch.rand(1, 1, 14, 12)

    resized = resize_target_to_source(source, target)

    assert resized.shape == source.shape
    assert source.shape == (1, 1, 8, 10)
    assert resized.is_contiguous()


def test_extract_trajectory_includes_zero_and_endpoint_states():
    torch.manual_seed(7)
    steps, height, width = 3, 8, 10
    source = torch.rand(1, 1, height, width)
    initial_momentum = torch.rand_like(source)
    image_stock = torch.rand(steps, 1, height, width)
    momentum_stock = torch.rand(steps, 1, height, width)
    registration = SimpleNamespace(
        source=source,
        to_analyse=(initial_momentum, None),
        mp=SimpleNamespace(
            n_step=steps,
            image_stock=image_stock,
            momentum_stock=momentum_stock,
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
        trajectory["images"], 0.4, operator
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
        "rho": 0.4,
        "alpha": 0.2,
        "beta": 0.2,
        "gamma": 0.001,
    }

    output = save_trajectory(
        trajectory,
        tmp_path / "fields",
        source_path=tmp_path / "source.png",
        target_path=tmp_path / "target.png",
        target_image=target,
        parameters=parameters,
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
    assert manifest["frames"][1]["image_momentum"] == (
        "scalar/momentum/image_momentum_t001.pt"
    )
    assert manifest["frames"][2]["image_velocity"] == (
        "scalar/velocity/image_velocity_t002.pt"
    )
