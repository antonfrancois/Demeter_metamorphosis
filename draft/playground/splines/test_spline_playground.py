import os
from dataclasses import replace
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[3]
for path in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseEvent
import pytest
import torch
from types import SimpleNamespace

from demeter.utils import torchbox as tb
from demeter.utils.cometric_inversion import CometricOperator
from demeter.utils.reproducing_kernels import SobolevFluidOperator
from draft.playground.field_playground_core import (
    prepare_vector_display,
    scaled_field_title,
)
from draft.playground.splines.app import DUAL_COLOR, SplinePlayground
from draft.playground.splines.core import (
    SplineParameters,
    SplineSetup,
    load_setup,
    resolve_device,
    run_classical,
    run_spline,
    save_setup,
    zero_setup,
)
from draft.playground.splines.main import _parameter_overrides, _replace_parameters
from draft.playground.splines.styles import INK_COLOR


def test_parameters_require_ordered_interior_control_nodes():
    parameters = SplineParameters(n_steps=8, control_steps=(2, 5))
    assert parameters.control_times == (0.25, 0.625)
    assert parameters.mesh_control_times == (0.25, 0.625)

    midpoint = SplineParameters(n_steps=16, control_steps=(8,))
    refined = replace(midpoint, n_steps=40)
    assert refined.control_times == (0.5,)
    assert refined.control_steps == (20,)
    assert replace(refined, n_steps=16).control_steps == (8,)

    with pytest.raises(ValueError, match="interior"):
        SplineParameters(n_steps=8, control_steps=(0,))
    with pytest.raises(ValueError, match="strictly increasing"):
        SplineParameters(n_steps=8, control_steps=(5, 2))
    with pytest.raises(TypeError, match="integers"):
        SplineParameters(n_steps=8, control_steps=(2.0,))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="distinct"):
        SplineParameters(
            n_steps=4,
            control_times=(0.5 - 1e-10, 0.5 + 1e-10),
        )
    expected_device = "cuda" if torch.cuda.is_available() else "cpu"
    assert resolve_device("auto").type == expected_device
    with pytest.raises(TypeError, match="progress_callback must be callable"):
        run_spline(
            zero_setup(torch.zeros(1, 1, 3, 4), parameters=SplineParameters(n_steps=1)),
            device="cpu",
            progress_callback=object(),  # type: ignore[arg-type]
        )


def test_run_converts_initial_force_and_aligns_interval_fields_to_nodes():
    torch.manual_seed(12)
    source = torch.rand(1, 1, 8, 9)
    parameters = SplineParameters(
        alpha=0.4,
        beta=0.2,
        gamma=0.3,
        rho=0.25,
        cg_eps=1e-7,
        n_steps=2,
    )
    setup = zero_setup(source, source, parameters)
    setup.initial_force.normal_(std=1e-3)
    setup.initial_momentum.normal_(std=1e-3)

    progress = []
    trajectory = run_spline(
        setup,
        device="cpu",
        progress_callback=lambda completed, total: progress.append(
            (completed, total)
        ),
    )
    kernel = SobolevFluidOperator(alpha=0.4, beta=0.2, gamma=0.3)
    expected_acceleration = CometricOperator(source, 0.25, kernel)(
        setup.initial_force
    )

    torch.testing.assert_close(
        trajectory.acceleration[0],
        expected_acceleration[0],
    )
    torch.testing.assert_close(
        trajectory.force[0],
        setup.initial_force[0],
        atol=2e-6,
        rtol=2e-4,
    )
    assert trajectory.images.shape == (3, 1, 8, 9)
    assert trajectory.deformed_source.shape == (3, 1, 8, 9)
    assert trajectory.photometric_only.shape == (3, 1, 8, 9)
    assert trajectory.momentum.shape == (3, 1, 8, 9)
    assert trajectory.force.shape == (3, 1, 8, 9)
    assert trajectory.acceleration.shape == (3, 1, 8, 9)
    assert trajectory.jerk.shape == (3, 1, 8, 9)
    assert trajectory.velocity.shape == (3, 2, 8, 9)
    assert trajectory.vector_momentum.shape == (3, 2, 8, 9)
    assert trajectory.target_mse.shape == (3,)
    for tensor in (
        trajectory.images,
        trajectory.deformed_source,
        trajectory.photometric_only,
        trajectory.momentum,
        trajectory.force,
        trajectory.acceleration,
        trajectory.jerk,
        trajectory.velocity,
        trajectory.vector_momentum,
        trajectory.target_mse,
    ):
        assert tensor.device.type == "cpu"
        assert not tensor.requires_grad
    assert progress == [(1, 2), (2, 2)]
    torch.testing.assert_close(trajectory.deformed_source[0], source[0])
    torch.testing.assert_close(trajectory.photometric_only[0], source[0])
    torch.testing.assert_close(
        trajectory.photometric_only[1],
        source[0] + 0.5 * (1 - parameters.rho) * setup.initial_momentum[0],
    )
    assert set(trajectory.field_energies) == {
        "momentum",
        "force",
        "acceleration",
        "jerk",
        "velocity",
        "vector_momentum",
    }
    assert all(values.shape == (3,) for values in trajectory.field_energies.values())
    expected_vector_momentum = kernel.apply_operator(trajectory.velocity)
    torch.testing.assert_close(
        trajectory.vector_momentum,
        expected_vector_momentum,
    )
    expected_vector_energy = (
        trajectory.velocity * trajectory.vector_momentum
    ).sum(dim=(1, 2, 3))
    torch.testing.assert_close(
        trajectory.field_energies["vector_momentum"],
        expected_vector_energy,
    )
    torch.testing.assert_close(
        trajectory.field_energies["vector_momentum"],
        trajectory.field_energies["velocity"],
    )
    initial_gradient = tb.spatialGradient(
        source,
        dx_convention="pixel",
        boundary="periodic",
    )[:, 0]
    initial_transport = -0.25 * kernel(
        setup.initial_momentum * initial_gradient
    )
    identity = tb.make_regular_grid(
        source.shape[-2:],
        dx_convention="pixel",
        device=source.device,
    ).to(source)
    expected_deformed_source = tb.imgDeform(
        source,
        identity - 0.5 * tb.im2grid(initial_transport),
        dx_convention="pixel",
        clamp=False,
        boundary="periodic",
    )
    torch.testing.assert_close(
        trajectory.deformed_source[1],
        expected_deformed_source[0],
    )

    endpoint_cometric = CometricOperator(
        trajectory.images[-1][None],
        0.25,
        SobolevFluidOperator(alpha=0.4, beta=0.2, gamma=0.3),
    )
    torch.testing.assert_close(
        endpoint_cometric(trajectory.force[-1][None]),
        trajectory.acceleration[-1][None],
        atol=2e-5,
        rtol=2e-4,
    )
    endpoint_image = trajectory.images[-1][None]
    endpoint_gradient = tb.spatialGradient(
        endpoint_image,
        dx_convention="pixel",
        boundary="periodic",
    )[:, 0]
    expected_velocity = -(0.25**0.5) * kernel(
        trajectory.momentum[-1][None] * endpoint_gradient
    )
    torch.testing.assert_close(trajectory.velocity[-1][None], expected_velocity)


def test_classical_run_matches_geodesic_spline_and_zeroes_spline_fields():
    torch.manual_seed(13)
    source = torch.rand(1, 1, 8, 9)
    target = torch.roll(source, shifts=1, dims=-1)
    setup = zero_setup(
        source,
        target,
        SplineParameters(
            rho=0.25,
            gamma=0.3,
            n_steps=3,
            control_steps=(),
        ),
    )
    setup.initial_momentum.normal_(std=0.1)
    progress = []

    classical = run_classical(
        setup,
        device="cpu",
        progress_callback=lambda completed, total: progress.append(
            (completed, total)
        ),
    )
    spline = run_spline(setup, device="cpu")

    for name in (
        "images",
        "deformed_source",
        "photometric_only",
        "momentum",
        "velocity",
        "vector_momentum",
        "target_mse",
    ):
        torch.testing.assert_close(
            getattr(classical, name),
            getattr(spline, name),
            atol=2e-6,
            rtol=2e-6,
        )
    for name in ("force", "acceleration", "jerk"):
        assert torch.count_nonzero(getattr(classical, name)) == 0
        assert torch.count_nonzero(classical.field_energies[name]) == 0
    assert progress == [(1, 3), (2, 3), (3, 3)]


def test_classical_run_rejects_drawn_non_momentum_fields():
    setup = zero_setup(
        torch.zeros(1, 1, 5, 6),
        parameters=SplineParameters(n_steps=2, control_steps=(1,)),
    )
    for field, message in (
        (setup.initial_force, "initial force"),
        (setup.initial_jerk, "initial jerk"),
        (setup.control_jerks, "control jerk"),
    ):
        field.fill_(1)
        with pytest.raises(ValueError, match=message):
            run_classical(setup, device="cpu")
        field.zero_()


def test_classical_button_uses_shared_workspace_and_reports_invalid_fields():
    source = torch.zeros(1, 1, 8, 9)
    source[..., 2:6, 3:7] = 0.5
    setup = zero_setup(
        source,
        torch.roll(source, shifts=1, dims=-1),
        SplineParameters(rho=0.25, gamma=0.3, n_steps=2),
    )
    setup.initial_momentum.fill_(0.05)
    app = SplinePlayground(setup, device="cpu")

    app.run_classical()
    assert app.last_error is None
    assert app.cache is not None
    assert app.cache.images.shape == (3, 1, 8, 9)
    assert torch.count_nonzero(app.cache.force) == 0
    assert torch.count_nonzero(app.cache.acceleration) == 0
    assert torch.count_nonzero(app.cache.jerk) == 0
    assert "Classical metamorphosis complete" in app.status_text.get_text()
    app.set_time_index(2)
    assert app.current_ax.get_title().startswith("Current image")

    app.fields["initial_force"].fill_(1)
    app.run_classical()
    assert app.cache is None
    assert isinstance(app.last_error, ValueError)
    assert "accepts only initial momentum" in app.status_text.get_text()
    plt.close(app.fig)


def test_setup_canonicalization_rejects_invalid_data_and_breaks_aliases():
    source = torch.zeros(1, 1, 5, 6)
    shared = torch.zeros_like(source)
    setup = SplineSetup(
        source=source,
        target=source,
        initial_momentum=shared,
        initial_force=shared,
        initial_jerk=shared,
        control_jerks=source.new_zeros((0,) + tuple(source.shape)),
        parameters=SplineParameters(n_steps=2),
    )
    source.fill_(1)
    setup.initial_momentum.fill_(2)
    assert torch.count_nonzero(setup.source) == 0
    assert torch.count_nonzero(setup.initial_force) == 0
    assert torch.count_nonzero(setup.initial_jerk) == 0

    with pytest.raises(ValueError, match="source must have shape"):
        zero_setup(torch.zeros(1, 1, 2, 3, 4))
    invalid_target = torch.zeros(1, 1, 5, 6)
    invalid_target[..., 2, 3] = torch.nan
    with pytest.raises(ValueError, match="target must contain only finite"):
        zero_setup(torch.zeros(1, 1, 5, 6), invalid_target)
    with pytest.raises(ValueError, match="source must contain only finite"):
        zero_setup(torch.full((1, 1, 5, 6), torch.inf))


def test_control_right_limit_and_setup_round_trip(tmp_path):
    source = torch.zeros(1, 1, 6, 7)
    parameters = SplineParameters(rho=0, n_steps=4, control_steps=(2,))
    setup = zero_setup(source, source, parameters)
    setup.initial_jerk.fill_(1)
    setup.control_jerks.fill_(3)
    setup.initial_force.fill_(0.5)

    path = save_setup(setup, tmp_path / "spline")
    restored = load_setup(path)
    trajectory = run_spline(restored, device="cpu")

    assert path.suffix == ".pt"
    assert restored.parameters == parameters
    assert torch.equal(restored.initial_force, setup.initial_force)
    assert torch.equal(restored.control_jerks, setup.control_jerks)
    assert trajectory.jerk[:, 0].mean(dim=(1, 2)).tolist() == [
        1.0,
        1.0,
        3.0,
        3.0,
        3.0,
    ]
    torch.testing.assert_close(trajectory.force[0], setup.initial_force[0])
    torch.testing.assert_close(
        trajectory.deformed_source,
        source[0].expand_as(trajectory.deformed_source),
    )
    torch.testing.assert_close(
        trajectory.photometric_only,
        trajectory.images,
    )

    assert setup.payload()["parameters"]["control_times"] == (0.5,)
    legacy = setup.payload()
    del legacy["parameters"]["control_times"]
    legacy_path = tmp_path / "legacy.pt"
    torch.save(legacy, legacy_path)
    assert load_setup(legacy_path).parameters.control_times == (0.5,)

    mismatched = setup.payload()
    mismatched["parameters"]["control_steps"] = (1,)
    mismatched_path = tmp_path / "mismatched.pt"
    torch.save(mismatched, mismatched_path)
    with pytest.raises(ValueError, match="do not match"):
        load_setup(mismatched_path)

    malformed = setup.payload()
    del malformed["parameters"]["rho"]
    malformed_path = tmp_path / "malformed.pt"
    torch.save(malformed, malformed_path)
    with pytest.raises(ValueError, match="missing parameters: rho"):
        load_setup(malformed_path)


def test_headless_editor_run_timeline_and_control_markers(tmp_path):
    source = torch.zeros(1, 1, 18, 20)
    source[..., 5:13, 6:14] = 0.6
    target = torch.roll(source, shifts=2, dims=-1)
    setup = zero_setup(
        source,
        target,
        SplineParameters(rho=0, n_steps=4, control_steps=(2,)),
    )
    app = SplinePlayground(setup, device="cpu")

    app.editor.on_press(
        SimpleNamespace(
            inaxes=app.source_ax,
            xdata=8.0,
            ydata=9.0,
            button=1,
            key=None,
        )
    )
    app.editor.on_motion(
        SimpleNamespace(inaxes=app.source_ax, xdata=13.0, ydata=9.0)
    )
    app.editor.on_release(
        SimpleNamespace(
            inaxes=app.source_ax,
            xdata=13.0,
            ydata=9.0,
            button=1,
        )
    )
    assert torch.count_nonzero(app.fields["initial_momentum"]) > 0
    assert app.cache is None
    app.clear()
    assert torch.count_nonzero(app.fields["initial_momentum"]) == 0
    app.undo()
    assert torch.count_nonzero(app.fields["initial_momentum"]) > 0
    app.fields["initial_force"].fill_(0.25)
    app.clear_all()
    assert all(torch.count_nonzero(field) == 0 for field in app.fields.values())
    app.undo()
    assert torch.count_nonzero(app.fields["initial_momentum"]) > 0

    app.input_radio.set_active(3)
    assert app.editor.active_key == "control_jerk:0"
    assert int(app.time_slider.val) == 2
    assert app.current_ax.get_title() == "Current image (run required)"
    assert "I_0" in app.target_footer.get_text()
    assert len(app._control_markers) == 2
    app._select_control_time(1)
    assert app.overlay_control_selector.selected_index == 0
    assert app.control_time_editor.selected_index == 0

    app.set_time_index(0)
    marker = next(iter(app._control_markers))
    app._on_pick(SimpleNamespace(artist=marker))
    assert int(app.time_slider.val) == 2

    app.run()
    assert app.cache is not None
    assert app.cache.images.shape == (5, 1, 18, 20)
    app.current_radio.set_active(1)
    app.set_time_index(4)
    assert app.current_ax.get_title().endswith("$p$")
    assert r"\Vert p(t)\Vert_{A_{I(t)}}^2" in app.current_footer.get_text()
    assert r"\mathrm{MSE}" in app.target_footer.get_text()
    assert "device: cpu" in app.status_text.get_text()
    assert "steps: 4" in app.status_text.get_text()
    assert app.time_slider.label.get_text() == "t"
    assert app.status_text.get_fontsize() >= 11
    assert all(
        footer.get_fontsize() >= 13
        for footer in (app.source_footer, app.current_footer, app.target_footer)
    )

    replacement_target = tmp_path / "replacement-target.png"
    plt.imsave(replacement_target, torch.ones(18, 20).numpy(), cmap="gray")
    cached_images = app.cache.images
    old_mse = app.cache.target_mse.clone()
    app.load_target(replacement_target)
    assert app.cache is not None
    assert app.cache.images is cached_images
    assert not torch.equal(app.cache.target_mse, old_mse)

    app.target_radio.set_active(1)
    assert app.target_image.get_cmap().name == "magma"
    assert "Absolute error" in app.target_ax.get_title()

    screenshot = tmp_path / "spline-playground.png"
    app.fig.savefig(screenshot, dpi=100)
    assert screenshot.stat().st_size > 10_000
    plt.close(app.fig)


def test_zero_control_selection_rho_and_new_setup_extent_are_consistent():
    setup = zero_setup(
        torch.zeros(1, 1, 10, 12),
        parameters=SplineParameters(rho=0.99, n_steps=2),
    )
    app = SplinePlayground(setup, device="cpu")
    assert app.rho_slider.val == pytest.approx(0.99)
    assert app.steps_slider.val == 2
    assert app.steps_slider.valmin == 1
    assert app.steps_slider.valmax == 40
    assert app.make_setup().parameters.rho == pytest.approx(0.99)
    assert app.menu_button.ax.get_position().y0 > app.file_button.ax.get_position().y0
    assert app.file_button.ax.get_position().y0 > app.run_button.ax.get_position().y0
    assert (
        app.run_button.ax.get_position().y0
        > app.classical_button.ax.get_position().y0
        > app.clear_button.ax.get_position().y0
    )
    shortcuts = next(
        text for text in app.fig.texts if text.get_text().startswith("P  parameters")
    )
    assert shortcuts.get_position() == pytest.approx((0.012, 0.975))
    assert shortcuts.get_ha() == "left"
    assert shortcuts.get_va() == "top"
    assert shortcuts.get_text().count("\n") == 7
    assert not any("A_I^{-1}" in text.get_text() for text in app.fig.texts)

    app._on_key_press(SimpleNamespace(key="p"))
    assert app.parameter_menu_open
    assert app.parameter_menu.backdrop_ax.get_visible()
    assert all(slider.active for slider in app.parameter_menu.sliders)
    assert app.device_radio.active
    assert app.device_radio.value_selected == "CPU"
    assert not any(button.active for button in app.file_menu.buttons)
    assert not app.source_ax.get_visible()
    app.fig.canvas.grab_mouse(app.rho_slider.ax)
    app.rho_slider.set_val(0.5)
    assert app.parameters.rho == pytest.approx(0.99)
    app.fig.canvas.release_mouse(app.rho_slider.ax)
    app._on_button_release(None)
    assert app.parameters.rho == pytest.approx(0.5)
    assert app._workspace_dirty
    app.steps_slider.set_val(4)
    assert app.parameters.n_steps == 4
    assert app.time_slider.valmax == 4
    app._on_key_press(SimpleNamespace(key="p"))
    assert not app.parameter_menu_open
    assert not app.device_radio.active
    assert app.source_ax.get_visible()
    assert not app._workspace_dirty

    app.input_radio.set_active(1)
    assert app.editor.active_key == "initial_force"
    app.input_radio.set_active(3)
    assert app.input_kind == "initial_jerk"
    assert app.editor.active_key == "initial_jerk"
    assert app.input_radio.value_selected == app.input_radio.labels[2].get_text()
    app.set_menu_visible(True)
    assert not app.overlay_control_selector.axis.get_visible()
    app.set_menu_visible(False)

    replacement = zero_setup(
        torch.zeros(1, 1, 6, 9),
        parameters=SplineParameters(n_steps=3, control_steps=(1, 2)),
    )
    app.apply_setup(replacement)
    expected_extent = [-0.5, 8.5, -0.5, 5.5]
    for image in (app.source_image, app.current_image, app.target_image):
        assert list(image.get_extent()) == expected_extent
    assert app.time_slider.valmax == 3
    assert app.steps_slider.val == 3
    assert app.steps_slider.valmin == 1
    assert app.steps_slider.valmax == 40
    assert len(app._control_markers) == 4
    controls_heading = next(
        text for text in app.fig.texts if text.get_text() == "CONTROLS"
    )
    assert controls_heading.get_ha() == "center"
    operator_text = next(
        text
        for text in app.parameter_menu.panel_axes["model"].texts
        if text.get_text().startswith("OPERATOR")
    )
    assert operator_text.get_ha() == "center"
    assert "Lv=" in operator_text.get_text()

    app.input_radio.set_active(3)
    app.set_menu_visible(True)
    assert type(app.overlay_control_selector) is type(app.control_time_editor)
    assert not app.overlay_control_selector.editable
    assert app.overlay_control_selector.axis.get_visible()
    app.fig.canvas.draw()
    axis = app.overlay_control_selector.axis
    x, y = axis.transData.transform((2 / 3, 0.55))
    for event_name in ("button_press_event", "button_release_event"):
        event = MouseEvent(event_name, app.fig.canvas, x, y, button=1)
        app.fig.canvas.callbacks.process(event_name, event)
    assert app.control_index == 1
    assert app.editor.active_key == "control_jerk:1"
    assert app.overlay_control_selector.selected_index == 1
    assert app.control_time_editor.selected_index == 1
    assert int(app.time_slider.val) == 2
    plt.close(app.fig)


def test_control_times_rescale_and_can_be_edited_on_the_parameter_timeline():
    source = torch.zeros(1, 1, 10, 12)
    setup = zero_setup(
        source,
        parameters=SplineParameters(n_steps=16, control_steps=(8,)),
    )
    setup.control_jerks[0].fill_(3)
    app = SplinePlayground(setup, device="cpu")

    app.steps_slider.set_val(40)
    assert app.parameters.control_times == (0.5,)
    assert app.parameters.control_steps == (20,)
    assert torch.all(app.fields["control_jerk:0"] == 3)
    assert 20 in app._control_markers.values()
    app.steps_slider.set_val(16)
    assert app.parameters.control_steps == (8,)

    app._remove_control_time(0)
    assert app.parameters.control_times == ()
    app.set_parameter_menu_visible(True)
    app.fig.canvas.draw()
    axis = app.control_time_editor.axis
    assert not axis.get_xticks().size
    endpoint_labels = {
        text.get_text(): text for text in axis.texts if text.get_text() in ("0", "1")
    }
    assert endpoint_labels["0"].get_position() == pytest.approx((0, 0.40))
    assert endpoint_labels["1"].get_position() == pytest.approx((1, 0.40))
    assert 0.55 - endpoint_labels["0"].get_position()[1] < 0.2

    def dispatch(name, xdata, button=1):
        x, y = axis.transData.transform((xdata, 0.55))
        event = MouseEvent(name, app.fig.canvas, x, y, button=button)
        app.fig.canvas.callbacks.process(name, event)

    dispatch("button_press_event", 0.5)
    dispatch("button_release_event", 0.5)
    assert app.parameters.control_times == (0.5,)
    assert app.parameters.control_steps == (8,)
    app.fields["control_jerk:0"].fill_(7)

    dispatch("button_press_event", 0.5)
    dispatch("motion_notify_event", 0.75)
    dispatch("button_release_event", 0.75)
    assert app.parameters.control_times == (0.75,)
    assert app.parameters.control_steps == (12,)
    assert torch.all(app.fields["control_jerk:0"] == 7)

    dispatch("button_press_event", 0.75, button=3)
    assert app.parameters.control_times == ()
    assert not any(key.startswith("control_jerk:") for key in app.fields)
    plt.close(app.fig)


def test_parameter_device_selector_defaults_to_cuda_and_can_choose_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    setup = zero_setup(
        torch.zeros(1, 1, 6, 8),
        parameters=SplineParameters(n_steps=2),
    )
    app = SplinePlayground(setup)

    assert app.device == "cuda"
    assert [label.get_text() for label in app.device_radio.labels] == ["CUDA", "CPU"]
    assert app.device_radio.value_selected == "CUDA"
    app.set_parameter_menu_visible(True)
    app.device_radio.set_active(1)
    assert app.device == "cpu"
    assert app.device_radio.value_selected == "CPU"
    assert "device: cpu" in app.status_text.get_text()
    plt.close(app.fig)


def test_cli_step_override_preserves_control_time_and_field():
    setup = zero_setup(
        torch.zeros(1, 1, 5, 6),
        parameters=SplineParameters(n_steps=16, control_steps=(8,)),
    )
    setup.control_jerks[0].fill_(4)
    args = SimpleNamespace(
        alpha=None,
        beta=None,
        gamma=None,
        rho=None,
        cg_eps=None,
        steps=40,
        control_steps=None,
    )
    parameters = _parameter_overrides(args, setup.parameters)
    replaced = _replace_parameters(setup, parameters)
    assert parameters.control_times == (0.5,)
    assert parameters.control_steps == (20,)
    torch.testing.assert_close(replaced.control_jerks, setup.control_jerks)


def test_overlay_menu_and_current_image_modes():
    source = torch.zeros(1, 1, 12, 14)
    source[..., 3:9, 4:10] = 0.5
    setup = zero_setup(
        source,
        source,
        SplineParameters(rho=0.25, n_steps=2, control_steps=(1,)),
    )
    setup.initial_momentum.fill_(0.1)
    app = SplinePlayground(setup, device="cpu")

    assert app.input_radio.labels[3].get_text().startswith("Control jerk")
    image_labels = [label.get_text() for label in app.current_image_radio.labels]
    assert image_labels[1] == r"Deformation only  $I_D$"
    field_labels = [label.get_text() for label in app.current_radio.labels]
    assert r"$u=A_I^{-1}a$" in field_labels[2]
    assert r"$a=A_Iu$" in field_labels[3]
    assert r"$v=Km$" in field_labels[5]
    assert not app.menu_open
    assert not app.overlay_menu.backdrop_ax.get_visible()
    assert not app.overlay_control_selector.axis.get_visible()
    for radio in (app.input_radio, app.current_radio):
        buttons = getattr(radio, "_buttons", None)
        if buttons is not None:
            assert not buttons.get_visible()
        else:
            assert all(not circle.get_visible() for circle in radio.circles)
    app._show_run_progress(1, 2)
    assert "1/2 (50%)" in app.status_text.get_text()
    app._show_run_progress(199, 200)
    assert "199/200 (99%)" in app.status_text.get_text()
    app.fig.canvas.grab_mouse(app.rho_slider.ax)
    app.run()
    assert app.cache is None
    app.fig.canvas.release_mouse(app.rho_slider.ax)
    app.run()
    assert app.cache is not None
    assert app.time_slider.active
    assert app.editor.enabled

    app.fig.canvas.grab_mouse(app.rho_slider.ax)
    app.set_menu_visible(True)
    assert not app.menu_open
    app.fig.canvas.release_mouse(app.rho_slider.ax)

    app._on_key_press(SimpleNamespace(key="m"))
    assert app.menu_open
    assert app.overlay_menu.backdrop_ax.get_visible()
    assert all(axis.get_visible() for axis in app.overlay_menu.column_axes.values())
    assert not app.overlay_control_selector.axis.get_visible()
    assert not app.overlay_menu.control_time_label.get_visible()
    assert not app.source_ax.get_visible()
    app.input_radio.set_active(3)
    assert app.overlay_control_selector.axis.get_visible()
    assert app.overlay_menu.control_time_label.get_visible()
    app.input_radio.set_active(0)
    assert not app.overlay_control_selector.axis.get_visible()
    assert not app.overlay_menu.control_time_label.get_visible()
    assert not app.time_slider.active
    assert not any(button.active for button in app.file_menu.buttons)
    assert not any(slider.active for slider in app.parameter_menu.sliders)
    old_time = app.time_slider.val
    app._running = True
    marker = next(iter(app._control_markers))
    app._on_pick(SimpleNamespace(artist=marker))
    assert app.time_slider.val == old_time
    app._running = False
    app._on_key_press(SimpleNamespace(key="right"))
    assert app.time_slider.val == old_time

    file_actions = []
    app._run_file_action = lambda action: file_actions.append(action)
    buttons = getattr(app.current_image_radio, "_buttons", None)
    if buttons is not None:
        point = buttons.get_offsets()[1]
    else:
        point = app.current_image_radio.circles[1].center
    x, y = app.current_image_radio.ax.transAxes.transform(point)
    for event_name in ("button_press_event", "button_release_event"):
        event = MouseEvent(event_name, app.fig.canvas, x, y, button=1)
        app.fig.canvas.callbacks.process(event_name, event)
    assert app.current_image_mode == "deformation"
    assert file_actions == []

    app.current_radio.set_active(0)
    app.target_radio.set_active(1)
    app._on_key_press(SimpleNamespace(key="escape"))
    assert not app.menu_open
    assert app.time_slider.active
    assert app.source_ax.get_visible()
    assert app.current_image_mode == "deformation"
    assert app.current_field is None
    assert app.target_mode == "Absolute error"
    assert app.current_ax.get_title().startswith("Deformation only")
    assert "$I_{D," in app.current_ax.get_title()
    assert r"I_D(t)" in app.target_ax.get_title()
    assert app.source_ax.title.get_color() == INK_COLOR
    assert app.current_ax.title.get_color() == INK_COLOR

    app.current_image_radio.set_active(2)
    assert app.current_image_mode == "photometric"
    assert app.current_image.get_cmap().name == "gray"
    torch.testing.assert_close(
        app.renderer.current_image_tensor(
            app.source,
            app.cache,
            app.current_image_mode,
            app._time_index(),
        ),
        app.cache.photometric_only[app._time_index()],
    )
    assert "Photometric only" in app.current_ax.get_title()
    assert r"|I_{\mathrm{phot}}(t)-I_\mathrm{target}|" in (
        app.target_ax.get_title()
    )
    assert r"I_{\mathrm{phot}}(t)" in app.target_footer.get_text()

    app.current_radio.set_active(6)
    assert app.current_field == "vector_momentum"
    assert r"\Vert m(t)\Vert_{V^*}^2" in app.current_footer.get_text()
    quivers = [
        artist
        for artist in app._dynamic_artists[app.current_ax]
        if artist.__class__.__name__ == "Quiver"
    ]
    assert len(quivers) == 1
    assert quivers[0].get_facecolor()[0] == pytest.approx(
        matplotlib.colors.to_rgba(DUAL_COLOR)
    )
    displayed = app.cache.vector_momentum[app._time_index()]
    values, x, y, factor = prepare_vector_display(displayed)
    assert quivers[0].X.tolist() == x.tolist()
    assert quivers[0].Y.tolist() == y.tolist()
    torch.testing.assert_close(
        torch.as_tensor(quivers[0].U),
        values[0, 0, y, x] * factor,
        check_dtype=False,
    )
    torch.testing.assert_close(
        torch.as_tensor(quivers[0].V),
        values[0, 1, y, x] * factor,
        check_dtype=False,
    )
    index = app._time_index()
    title = rf"Photometric only $I_{{\mathrm{{phot}},{index}}}$ + $m$"
    assert app.current_ax.get_title() == scaled_field_title(title, factor)

    app._on_key_press(SimpleNamespace(key="l"))
    assert app.file_menu_open
    assert app.file_menu.backdrop_ax.get_visible()
    app._on_key_press(SimpleNamespace(key="l"))
    assert not app.file_menu_open
    plt.close(app.fig)
