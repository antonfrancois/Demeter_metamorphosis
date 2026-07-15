import os

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.quiver import Quiver
import numpy as np
import pytest
import torch
from types import SimpleNamespace

from draft.cometric_inversion import CometricOperator
from draft.conjugate_gradient import conjugate_gradient
from draft.sobolevfluid_operator import SobolevFluidOperator
from draft.playground.field_playground import (
    FieldPlayground,
    LoadedField,
    add_vector_arrow,
    analyze_field,
    coerce_field,
    erase_stroke,
    load_field_file,
    paint_scalar_stroke,
    resize_field,
)


def test_vector_and_scalar_editing_primitives():
    vector = torch.zeros(1, 2, 31, 41)
    vector = add_vector_arrow(vector, (20, 15), (25, 12), sigma=3, gain=2)
    assert torch.allclose(vector[0, :, 15, 20], torch.tensor([10.0, -6.0]))
    maximum_index = vector.square().sum(dim=1)[0].argmax()
    assert divmod(int(maximum_index), vector.shape[-1]) == (15, 20)
    assert vector[0, :, 0, 0].norm() < 1e-6

    scalar = torch.zeros(1, 1, 31, 41)
    scalar = paint_scalar_stroke(scalar, [(8, 10), (30, 20)], sigma=2, amplitude=-1.5)
    assert scalar.min() < -1.4
    erased = erase_stroke(scalar, [(8, 10)], sigma=3)
    assert erased[0, 0, 10, 8].abs() < 1e-6
    assert erased[0, 0, 20, 30] < -1


def test_field_layout_coercion_and_vector_resize():
    grid_field = np.zeros((10, 20, 2), dtype=np.float32)
    grid_field[..., 0] = 4
    field, mode = coerce_field(grid_field)
    assert mode == "vector"
    assert field.shape == (1, 2, 10, 20)

    resized = resize_field(field, (20, 10))
    assert resized.shape == (1, 2, 20, 10)
    assert torch.allclose(resized[:, 0], torch.full((1, 20, 10), 2.0))
    momentum_resized = resize_field(
        field, (20, 10), scale_vector_displacement=False
    )
    assert torch.allclose(momentum_resized[:, 0], torch.full((1, 20, 10), 4.0))


def test_vector_analysis_round_trips():
    image = torch.rand(1, 1, 25, 29)
    velocity = torch.randn(1, 2, 25, 29)
    result = analyze_field(
        image,
        velocity,
        "velocity",
        alpha=0.4,
        beta=0.2,
        gamma=0.3,
    )
    assert result.counterpart.shape == velocity.shape
    assert result.roundtrip.shape == velocity.shape
    assert result.relative_roundtrip < 2e-5
    assert result.squared_norm > 0


def test_sobolev_operator_round_trips_and_has_exact_nyquist_symbol():
    torch.manual_seed(3)
    for dtype, tolerance in ((torch.float32, 2e-4), (torch.float64, 1e-11)):
        field = torch.randn(1, 2, 32, 34, dtype=dtype)
        operator = SobolevFluidOperator(alpha=0.2, beta=0.2, gamma=0.001)
        sharp_flat = operator.apply_operator(operator.apply_inverse(field))
        flat_sharp = operator.apply_inverse(operator.apply_operator(field))
        assert (sharp_flat - field).norm() / field.norm() < tolerance
        assert (flat_sharp - field).norm() / field.norm() < tolerance

        other = torch.randn_like(field)
        torch.testing.assert_close(
            (operator.apply_operator(field) * other).sum(),
            (field * operator.apply_operator(other)).sum(),
        )
        assert (field * operator.apply_operator(field)).sum() > 0

        _, mixed_symbol, _ = operator._symbol(field)
        assert torch.count_nonzero(mixed_symbol[0]) == 0
        assert torch.count_nonzero(mixed_symbol[:, 0]) == 0
        assert torch.count_nonzero(mixed_symbol[16]) == 0
        assert torch.count_nonzero(mixed_symbol[:, -1]) == 0


def test_scalar_forward_and_inverse_round_trip():
    torch.manual_seed(4)
    image = torch.rand(1, 1, 20, 24)
    covector = torch.randn(1, 1, 20, 24) * 0.1
    result = analyze_field(
        image,
        covector,
        "u",
        alpha=0.4,
        beta=0.2,
        gamma=0.3,
        rho=0.25,
        solve_inverse=True,
        cg_eps=1e-7,
    )
    assert result.counterpart.shape == covector.shape
    assert result.kernel_response.shape == (1, 2, 20, 24)
    assert result.relative_roundtrip < 1e-4

    inverse = analyze_field(
        image,
        result.counterpart,
        "a",
        alpha=0.4,
        beta=0.2,
        gamma=0.3,
        rho=0.25,
        solve_inverse=True,
        cg_eps=1e-7,
    )
    assert inverse.relative_roundtrip < 1e-4
    assert torch.allclose(inverse.counterpart, covector, atol=2e-5, rtol=2e-4)
    assert inverse.solver_iterations >= 1
    assert inverse.solver_time >= 0


def test_cometric_inverse_exposes_solver_info():
    torch.manual_seed(5)
    image = torch.rand(1, 1, 8, 9)
    acceleration = torch.rand_like(image)
    solution, iterations, elapsed = CometricOperator(
        image,
        0.25,
        lambda value: value,
    ).inverse(
        acceleration,
        eps=1e-7,
        return_info=True,
    )
    assert solution.shape == acceleration.shape
    assert iterations >= 1
    assert elapsed >= 0


def test_conjugate_gradient_solves_spd_system_and_zero_rhs():
    matrix = torch.tensor([[4.0, 1.0], [1.0, 3.0]], dtype=torch.float64)
    rhs = torch.tensor([1.0, 2.0], dtype=torch.float64)
    solution, iterations = conjugate_gradient(matrix.mv, rhs, 1e-12)
    torch.testing.assert_close(solution, torch.linalg.solve(matrix, rhs))
    assert iterations == 2

    solution, iterations = conjugate_gradient(
        matrix.mv, torch.zeros_like(rhs), 1e-12
    )
    assert torch.equal(solution, torch.zeros_like(rhs))
    assert iterations == 0


def test_invalid_cg_tolerance_and_field_file_inputs(tmp_path):
    image = torch.zeros(1, 1, 8, 9)
    acceleration = torch.zeros_like(image)
    operator = lambda value: value
    with pytest.raises(ValueError, match="eps"):
        CometricOperator(image, 0.5, operator).inverse(acceleration, eps=0)

    invalid = tmp_path / "invalid.pt"
    torch.save(
        {"field": torch.zeros(1, 1, 8, 9), "field_kind": "velocity"},
        invalid,
    )
    with pytest.raises(ValueError, match="incompatible"):
        load_field_file(invalid)


def test_saved_template_reloads_and_headless_ui_renders(tmp_path):
    image = torch.zeros(1, 1, 32, 36)
    image[..., 8:24, 10:26] = 1
    app = FieldPlayground(image, device="cpu")
    assert app.brush_slider.val == 1
    assert app.alpha_slider.val == pytest.approx(0.2)
    assert app.beta_slider.val == pytest.approx(0.2)
    assert app.gamma_slider.valtext.get_text() == "0.001"
    assert app.analysis is None
    assert app.kind == "vector_momentum"
    app.mode_radio.set_active(1)
    assert app.kind == "u"
    app.mode_radio.set_active(0)

    display_field = torch.full((1, 2, 32, 36), 1e-3)
    display_field[0, :, 16, 18] = torch.tensor([5.0, 2.0])
    app.fields["vector"] = display_field
    app._invalidate("test")
    arrows = [artist for artist in app.input_ax.collections if isinstance(artist, Quiver)]
    assert len(arrows) == 1
    assert arrows[0].U.size == 1
    assert arrows[0].U[0] == pytest.approx(5.0)
    assert "display" not in app.input_ax.get_title()
    field = add_vector_arrow(torch.zeros(1, 2, 32, 36), (12, 12), (20, 17), 4)
    app.set_template(
        LoadedField(
            field,
            "vector_momentum",
            metadata={
                "parameters": {"alpha": 3.0, "beta": 0.3, "gamma": 10.0, "rho": 0.2}
            },
        ),
        "test template",
    )
    assert app.alpha_slider.val == pytest.approx(3.0)
    assert 10 ** app.gamma_slider.val == pytest.approx(10.0)
    assert app.rho_slider.val == pytest.approx(0.2)
    assert app.analysis is None

    press = SimpleNamespace(
        inaxes=app.input_ax, xdata=12.0, ydata=12.0, button=1, key=None
    )
    app._on_press(press)
    app._on_motion(
        SimpleNamespace(inaxes=app.output_ax, xdata=0.5, ydata=0.5)
    )
    assert app.drag.points == [(12.0, 12.0)]
    app._cancel_drag()

    app._on_press(
        SimpleNamespace(
            inaxes=app.input_ax, xdata=20.0, ydata=20.0, button=1, key=None
        )
    )
    app._on_motion(
        SimpleNamespace(inaxes=app.input_ax, xdata=25.0, ydata=24.0)
    )
    preview_artist = app.drag.artist
    assert app.drag.points == [(20.0, 20.0)]
    assert app.drag.end == (25.0, 24.0)
    app._on_motion(
        SimpleNamespace(inaxes=app.input_ax, xdata=27.0, ydata=26.0)
    )
    assert app.drag.artist is preview_artist
    assert app.drag.points == [(20.0, 20.0)]
    assert len(app.input_ax.patches) == 1
    assert app.drag.background is not None
    app._on_release(
        SimpleNamespace(inaxes=app.input_ax, xdata=27.0, ydata=26.0)
    )
    assert app.analysis is None
    assert app.drag is None
    edited_field = app.fields["vector"].clone()
    expected_field = add_vector_arrow(field, (20, 20), (27, 26), sigma=1, gain=1)
    assert torch.equal(edited_field, expected_field)

    app.run()
    assert app.analysis is not None
    assert app.output_ax.get_title().startswith(r"Output: velocity $v=Km$")
    assert any(r"q_{95}" in text.get_text() for text in app.output_ax.texts)
    assert r"K(m)" not in app.norm_text.get_text()
    assert isinstance(app.error_colorbar.mappable.norm, LogNorm)
    assert app.error_colorbar.ax.get_ylabel() == ""
    error = (app.analysis.roundtrip - app.fields["vector"]).square().sum(1).sqrt()
    precision_floor = float(
        torch.finfo(error.dtype).eps
        * app.fields["vector"].norm()
        / np.sqrt(error.numel())
    )
    assert app.error_colorbar.mappable.norm.vmin == pytest.approx(precision_floor)
    heatmap = app.error_colorbar.mappable.get_array()
    assert not np.ma.getmaskarray(heatmap).any()
    assert float(heatmap.min()) == pytest.approx(precision_floor)
    assert r"\left\Vert L(Km)-m\right\Vert_2" in app.detail_ax.get_title()
    error_legend = app.detail_ax.texts[-1].get_text()
    assert r"q_{99}" in error_legend
    assert r"\max |e|" in error_legend
    axes_with_colorbar = len(app.fig.axes)
    app.refresh()
    assert len(app.fig.axes) == axes_with_colorbar
    app.kind_radio.set_active(0)
    app.run()
    assert r"\left\Vert K(Lv)-v\right\Vert_2" in app.detail_ax.get_title()
    assert app.error_colorbar.ax.get_ylabel() == ""

    app.mode_radio.set_active(1)
    assert app.error_colorbar is None
    app.fields["scalar"].fill_(0.1)
    app.kind_radio.set_active(0)
    app.run()
    solver_legend = app.output_ax.texts[-1].get_text()
    assert solver_legend.count("\n") == 2
    assert "iterations" in solver_legend
    assert "time" in solver_legend
    assert "error" in solver_legend
    app.mode_radio.set_active(0)
    app.kind_radio.set_active(1)

    output = app.save(tmp_path / "field.pt")
    app.fig.savefig(tmp_path / "playground.png")
    loaded = load_field_file(output)

    assert loaded.kind == "vector_momentum"
    assert torch.equal(loaded.field, edited_field)
    assert loaded.image.shape == image.shape
    assert (tmp_path / "playground.png").stat().st_size > 10_000
    plt.close(app.fig)
