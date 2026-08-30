"""Matplotlib application for editing and viewing metamorphosis splines.

Version: July 23, 2026.
"""

from __future__ import annotations

from bisect import bisect_left
from contextlib import suppress
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Any

import torch

from demeter.metamorphosis.splines import SplinesVariables
from demeter.utils.spline_data import load_timed_image_directory

from ..field_playground_core import resize_field
from .core import (
    SplineParameters,
    SplineSetup,
    SplineTrajectory,
    load_scalar_field,
    load_setup,
    minimum_mesh_steps,
    resolve_device,
    run_classic,
    run_spline,
    save_setup,
    target_mse,
    zero_setup,
)
from .editor import ScalarFieldEditor
from .images import load_image
from .menus import (
    MAX_STEPS,
    build_file_menu,
    build_image_menu,
    build_observation_menu,
    build_overlay_menu,
    build_parameter_menu,
)
from .menus.common import set_radio_active_color
from .menus.dialogs import choose_directory, choose_file, choose_files
from .registration import (
    RegistrationResult,
    register_classic as optimize_classic,
    register_spline as optimize_spline,
)
from .project_io import (
    load_project as load_project_directory,
    save_project as save_project_directory,
)
from .rendering import SplineRenderer, field_color
from .state import ImageSeries
from .styles import (
    CURRENT_FIELDS,
    CURRENT_IMAGE_LABELS,
    CURRENT_IMAGE_MODES,
    CURRENT_LABELS,
    DUAL_COLOR,
    FIELD_CLASS,
    INPUT_FIELDS,
    INPUT_LABELS,
    TARGET_ACTIVE_COLOR,
    TARGET_COLOR,
)
from .video_export import save_current_panel_video, trajectory_video_filename
from .workspace import build_workspace


DEFAULT_DRAWING_AMPLITUDE = 0.5


class SplinePlayground:
    """Edit shooting fields, run a spline, and inspect its cached trajectory."""

    def __init__(
        self,
        setup: SplineSetup,
        *,
        device: str = "auto",
        output_path: str | Path | None = None,
    ) -> None:
        if max(
            setup.parameters.spline.steps,
            setup.parameters.regression.steps,
        ) > MAX_STEPS:
            raise ValueError(
                f"the interactive playground supports at most {MAX_STEPS} steps"
            )
        self.device = str(resolve_device(device))
        self.output_path = Path(output_path).expanduser() if output_path else None
        self.cache: SplineTrajectory | None = None
        self.last_error: Exception | None = None
        self.last_registration: RegistrationResult | None = None
        self.fields: dict[str, torch.Tensor] = {}
        self._set_setup_state(setup)

        self.input_kind = "initial_momentum"
        self.current_image_mode = "full"
        self.current_field: str | None = "velocity"
        self.target_mode = "Target"
        self.show_input_image = True
        self.show_current_image = True
        self.control_index = 0
        self.active_modal: str | None = None
        self._running = False
        self._syncing_widgets = False
        self._pending_parameter_change = False
        self._last_progress_draw = 0.0
        self._workspace_dirty = False
        self._dynamic_artists: dict[Any, list[Any]] = {}
        self._control_markers: dict[Any, int] = {}
        self._target_markers: dict[Any, int] = {}

        self._build_figure()
        self.editor = ScalarFieldEditor(
            self.fig,
            self.source_ax,
            self.fields,
            active_key=self._active_field_key(),
            brush=lambda: float(self.sliders["brush"].val),
            amplitude=self._drawing_amplitude,
            color=DUAL_COLOR,
            on_change=self._on_field_changed,
        )
        self._connect_events()
        self._refresh_control_widgets()
        self._refresh_observation_widgets()
        self._set_status("Paint an input field, then run a trajectory.")
        self._render()

    @property
    def target(self) -> torch.Tensor:
        index = self.target_index
        return self.series.targets[index : index + 1]

    @property
    def target_index(self) -> int:
        step = self._time_index() if hasattr(self, "time_slider") else 0
        return self.series.target_at(step, self.parameters.spline.steps)

    @property
    def target_path(self) -> str:
        return self.series.paths[self.target_index]

    def _set_setup_state(self, setup: SplineSetup) -> None:
        self.parameters = setup.parameters
        self.series = ImageSeries.from_batch(setup.images)
        self._set_fields_from_setup(setup)
        self._drawing_amplitudes = dict.fromkeys(
            self.fields, DEFAULT_DRAWING_AMPLITUDE
        )

    def _set_fields_from_setup(self, setup: SplineSetup) -> None:
        self.fields.clear()
        self.fields.update(
            initial_momentum=setup.variables.initial_momentum.clone(),
            initial_acceleration=setup.variables.initial_acceleration.clone(),
            initial_jerk=setup.variables.initial_jerk.clone(),
        )
        for index, field in enumerate(setup.variables.control_jerks):
            self.fields[f"control_jerk:{index}"] = field.clone()

    def _build_figure(self) -> None:
        workspace = build_workspace(
            self.series.source,
            self.target,
            self.parameters.spline.steps,
            self.parameters.model,
            self._dynamic_artists,
        )
        self.fig = workspace.fig
        self.axes = workspace.axes
        self.source_ax, self.current_ax, self.target_ax = self.axes
        self.source_image, self.current_image, self.target_image = workspace.images
        self.colorbar_axes = workspace.colorbar_axes
        (
            self.source_colorbar_ax,
            self.current_colorbar_ax,
            self.target_colorbar_ax,
        ) = self.colorbar_axes
        self.source_footer, self.current_footer, self.target_footer = (
            workspace.footers
        )
        self.buttons = workspace.buttons
        self.time_slider = workspace.time_slider
        self.status_text = workspace.status_text
        self.renderer = SplineRenderer(
            self.axes,
            (self.source_image, self.current_image, self.target_image),
            self.colorbar_axes,
            (self.source_footer, self.current_footer, self.target_footer),
            self._dynamic_artists,
        )

        self._build_parameter_menu()
        self._build_overlay_menu()
        self._build_image_menu()
        self._build_file_menu()
        self._build_observation_menu()
        self.time_slider.on_changed(self._on_time)
        self._bind_buttons(
            (self.buttons["run"], self.run),
            (self.buttons["register"], self.register),
            (
                self.buttons["parameters"],
                lambda: self.set_parameter_menu_visible(not self.parameter_menu_open),
            ),
            (
                self.parameter_menu.close_button,
                lambda: self.set_parameter_menu_visible(False),
            ),
            (self.buttons["view"], lambda: self.set_menu_visible(not self.menu_open)),
            (self.overlay_menu.close_button, lambda: self.set_menu_visible(False)),
            (
                self.buttons["files"],
                lambda: self.set_file_menu_visible(not self.file_menu_open),
            ),
            (
                self.buttons["images"],
                lambda: self.set_image_menu_visible(
                    self.active_modal not in ("images", "observations")
                ),
            ),
            (self.buttons["clear"], self.clear),
            (self.buttons["clear_all"], self.clear_all),
        )
        self.parameter_menu.bind(
            on_parameter=self._on_parameter_change,
            on_model=self._on_model_change,
            on_kernel=self._on_operator_change,
            on_initialization=self._on_spline_initialization_change,
            on_amplitude=self._on_amplitude_change,
            on_spacing=self._on_spacing_change,
            on_device=self._on_device_change,
        )

        self._workspace_widgets = [*self.buttons.values(), self.time_slider]
        self._set_modal(None)

    def _build_parameter_menu(self) -> None:
        self.parameter_menu = build_parameter_menu(
            self.fig,
            self.parameters,
            device=self.device,
            on_control_add=self._add_control_time,
            on_control_move=self._move_control_time,
            on_control_remove=self._remove_control_time,
            on_control_select=self._select_control_time,
            on_message=self._show_message,
        )
        self.sliders = self.parameter_menu.sliders
        self.radios = self.parameter_menu.radios
        self.renderer.vector_spacing = round(self.sliders["spacing"].val)
        self.control_time_editor = self.parameter_menu.control_times

    def _build_overlay_menu(self) -> None:
        self.overlay_menu = build_overlay_menu(
            self.fig,
            self.parameters,
            on_control_select=self._select_control_time,
        )
        self.overlay_radios = self.overlay_menu.radios
        self.overlay_checks = self.overlay_menu.checks
        self.overlay_control_selector = self.overlay_menu.control_times
        self.overlay_menu.bind(
            on_input=self._on_input_field,
            on_input_image=self._on_input_image_toggle,
            on_current_image=self._on_current_image_toggle,
            on_image_mode=self._on_current_image_mode,
            on_current_field=self._on_current_field,
            on_target_mode=self._on_target_mode,
            on_target_loss=self._on_target_loss_change,
        )

    def _build_file_menu(self) -> None:
        actions = (
            ("LOAD FIELD", lambda: self._run_modal_action(self.load_field_dialog)),
            ("LOAD PROJECT", lambda: self._run_modal_action(self.load_project_dialog)),
            ("SAVE FIELD", lambda: self._run_modal_action(self.save_field_dialog)),
            ("SAVE PROJECT", lambda: self._run_modal_action(self.save_project_dialog)),
            ("SAVE VIDEO", lambda: self._run_modal_action(self.save_video_dialog)),
        )
        self.file_menu = build_file_menu(self.fig, actions)
        self._bind_buttons(
            (self.file_menu.close_button, lambda: self.set_file_menu_visible(False))
        )

    def _build_image_menu(self) -> None:
        self.image_menu = build_image_menu(self.fig)
        self._bind_buttons(
            (
                self.image_menu.load_source_button,
                lambda: self._run_modal_action(self.load_source_dialog),
            ),
            (
                self.image_menu.load_target_button,
                lambda: self._run_modal_action(self.load_target_dialog),
            ),
            (self.image_menu.close_button, lambda: self.set_image_menu_visible(False)),
        )

    def _build_observation_menu(self) -> None:
        self.observation_menu = build_observation_menu(
            self.fig,
            on_select=self._select_image,
            on_place=self._place_image,
            on_unplace=self._unplace_image,
        )
        self._bind_buttons(
            (
                self.observation_menu.load_directory_button,
                self.load_timed_directory_dialog,
            ),
            (self.observation_menu.add_images_button, self.add_images_dialog),
            (self.observation_menu.remove_button, self.remove_selected_image),
            (
                self.observation_menu.close_button,
                lambda: self.set_image_menu_visible(False),
            ),
        )

    @staticmethod
    def _bind_buttons(*actions) -> None:
        for button, action in actions:
            button.on_clicked(lambda _event, callback=action: callback())

    def _run_modal_action(self, action) -> None:
        self._set_modal(None)
        action()

    def _connect_events(self) -> None:
        canvas = self.fig.canvas
        default_key_handler = getattr(canvas.manager, "key_press_handler_id", None)
        if default_key_handler is not None:
            canvas.mpl_disconnect(default_key_handler)
        canvas.mpl_connect("key_press_event", self._on_key_press)
        canvas.mpl_connect("pick_event", self._on_pick)
        canvas.mpl_connect("button_release_event", self._on_button_release)
        canvas.mpl_connect("resize_event", lambda _event: self.editor.cancel())

    def set_menu_visible(self, visible: bool) -> None:
        self._set_modal("view" if visible else None)

    @property
    def menu_open(self) -> bool:
        return self.active_modal == "view"

    def set_file_menu_visible(self, visible: bool) -> None:
        self._set_modal("files" if visible else None)

    @property
    def file_menu_open(self) -> bool:
        return self.active_modal == "files"

    def set_image_menu_visible(self, visible: bool) -> None:
        if not visible:
            self._set_modal(None)
        elif self.parameters.model == "splines":
            self._set_modal("observations")
        else:
            self._set_modal("images")

    @property
    def image_menu_open(self) -> bool:
        return self.active_modal == "images"

    def set_parameter_menu_visible(self, visible: bool) -> None:
        self._set_modal("parameters" if visible else None)

    @property
    def parameter_menu_open(self) -> bool:
        return self.active_modal == "parameters"

    def _set_modal(self, modal: str | None) -> None:
        if self._running:
            return
        if (
            modal != self.active_modal
            and getattr(self.fig.canvas, "mouse_grabber", None) is not None
        ):
            return
        self.active_modal = modal
        if modal is not None and hasattr(self, "editor"):
            self.editor.cancel()
        self.overlay_menu.set_visible(
            self.menu_open,
            show_control_selector=(
                self.input_kind == "control_jerk"
                and bool(self.parameters.control_times)
            ),
            target_mode=self.target_mode,
        )
        self.file_menu.set_visible(self.file_menu_open)
        self.image_menu.set_visible(self.image_menu_open)
        self.observation_menu.set_visible(modal == "observations")
        self.parameter_menu.set_visible(self.parameter_menu_open)
        self._set_workspace_active(modal is None)
        self._set_workspace_visible(modal is None)
        if modal is None and self._workspace_dirty:
            self._workspace_dirty = False
            self._render()
        else:
            self.fig.canvas.draw_idle()

    def _set_workspace_active(self, active: bool) -> None:
        for widget in self._workspace_widgets:
            widget.active = active
        if hasattr(self, "editor"):
            self.editor.enabled = active

    def _set_workspace_visible(self, visible: bool) -> None:
        for axis in (
            *self.axes,
            *(widget.ax for widget in self._workspace_widgets),
        ):
            axis.set_visible(visible)
        self.renderer.set_colorbars_visible(visible)

    def _active_field_key(self) -> str:
        if self.input_kind != "control_jerk":
            return self.input_kind
        if not self.parameters.control_times:
            return "initial_jerk"
        index = min(self.control_index, len(self.parameters.control_times) - 1)
        return f"control_jerk:{index}"

    def _drawing_amplitude(self) -> float:
        return self._drawing_amplitudes.setdefault(
            self._active_field_key(),
            DEFAULT_DRAWING_AMPLITUDE,
        )

    def _sync_amplitude_slider(self) -> None:
        syncing = self._syncing_widgets
        self._syncing_widgets = True
        try:
            self.sliders["amplitude"].set_val(self._drawing_amplitude())
        finally:
            self._syncing_widgets = syncing

    def _current_parameters(self) -> SplineParameters:
        return self.parameter_menu.read(self.parameters)

    def _minimum_regression_steps(self) -> int:
        times = self.parameters.projected_control_times + tuple(
            float(time) for time in self.series.times if time is not None
        )
        return minimum_mesh_steps(times, max_steps=MAX_STEPS)

    def _use_minimum_regression_mesh(self) -> None:
        n_steps = self._minimum_regression_steps()
        self.parameters = replace(
            self.parameters,
            regression=replace(self.parameters.regression, steps=n_steps),
        )
        syncing = self._syncing_widgets
        self._syncing_widgets = True
        try:
            self.sliders["regression_steps"].set_val(n_steps)
        finally:
            self._syncing_widgets = syncing

    def make_setup(
        self,
        model: str | None = None,
        *,
        preserve_targets: bool = False,
    ) -> SplineSetup:
        parameters = self._current_parameters()
        if model is not None:
            parameters = replace(parameters, model=model)
        if parameters.control_times:
            controls = torch.stack(
                [
                    self.fields[f"control_jerk:{index}"]
                    for index in range(len(parameters.control_times))
                ],
                dim=0,
            )
        else:
            controls = self.series.source.new_zeros(
                (0,) + tuple(self.series.source.shape)
            )
        if parameters.model == "classic" and not preserve_targets:
            images = self.series.to_batch(self.target_index)
        else:
            images = self.series.to_batch()
        return SplineSetup(
            images=images,
            variables=SplinesVariables(
                self.fields["initial_momentum"],
                self.fields["initial_acceleration"],
                self.fields["initial_jerk"],
                controls,
            ),
            parameters=parameters,
        )

    def _set_status(self, message: str) -> None:
        height, width = self.series.source.shape[-2:]
        self.status_text.set_text(
            f"{message}  |  device: {self.device}  |  size: {height}x{width}"
        )

    def _show_message(self, message: str) -> None:
        self._set_status(message)
        self.fig.canvas.draw_idle()

    def _invalidate(self, message: str) -> None:
        self.cache = None
        self.last_registration = None
        if int(self.time_slider.val) != 0:
            self._syncing_widgets = True
            self.time_slider.set_val(0)
            self._syncing_widgets = False
        self._set_status(message)
        if self.active_modal is None:
            self._render()
        else:
            self._workspace_dirty = True

    def _render_panels(self, *renderers) -> None:
        if self.active_modal is not None:
            self._workspace_dirty = True
            return
        for render in renderers:
            render()

    def _on_field_changed(self, message: str) -> None:
        if self._running:
            return
        self._invalidate(message)

    def _on_input_field(self, _label: str) -> None:
        if self._syncing_widgets:
            return
        selected = INPUT_FIELDS[INPUT_LABELS.index(_label)]
        if selected == "control_jerk" and not self.parameters.control_times:
            self.input_kind = "initial_jerk"
            self.editor.set_active("initial_jerk")
            self._sync_amplitude_slider()
            self._syncing_widgets = True
            self.overlay_radios["input"].set_active(2)
            self._syncing_widgets = False
            self._set_status("No control nodes are configured.")
            self._update_overlay_control_selector_visibility()
            self._render_panels(self._render_source)
            self.fig.canvas.draw_idle()
            return
        self.input_kind = selected
        self.editor.set_active(self._active_field_key())
        self._sync_amplitude_slider()
        if selected == "control_jerk":
            self.jump_to_control(self.control_index)
        self._update_overlay_control_selector_visibility()
        self._render_panels(self._render_source)
        self.fig.canvas.draw_idle()

    def _on_current_image_mode(self, label: str) -> None:
        if self._syncing_widgets:
            return
        index = CURRENT_IMAGE_LABELS.index(label)
        self.current_image_mode = CURRENT_IMAGE_MODES[index]
        set_radio_active_color(
            self.overlay_radios["image_mode"],
            index,
            "#168a8a",
        )
        if self.cache is not None:
            self._render_panels(self._render_current, self._render_target)
        self.fig.canvas.draw_idle()

    def _on_input_image_toggle(self, _label: str) -> None:
        if self._syncing_widgets:
            return
        self.show_input_image = self.overlay_checks["input_image"].get_status()[0]
        self._render_panels(self._render_source)
        self.fig.canvas.draw_idle()

    def _on_current_image_toggle(self, _label: str) -> None:
        if self._syncing_widgets:
            return
        self.show_current_image = self.overlay_checks["current_image"].get_status()[0]
        self._render_panels(self._render_current)
        self.fig.canvas.draw_idle()

    def _on_current_field(self, _label: str) -> None:
        if self._syncing_widgets:
            return
        index = CURRENT_LABELS.index(_label)
        self.current_field = CURRENT_FIELDS[index]
        color = (
            "#168a8a"
            if self.current_field is None
            else field_color(FIELD_CLASS[self.current_field])
        )
        set_radio_active_color(self.overlay_radios["current_field"], index, color)
        if self.cache is not None:
            self._render_panels(self._render_current)
        self.fig.canvas.draw_idle()

    def _on_target_mode(self, label: str) -> None:
        if self._syncing_widgets:
            return
        self.target_mode = label
        self.overlay_menu.set_target_controls_visible(self.menu_open, label)
        self._render_panels(self._render_target)
        self.fig.canvas.draw_idle()

    def _on_target_loss_change(self, _label: str) -> None:
        if self._syncing_widgets:
            return
        self._render_panels(self._render_target)
        self.fig.canvas.draw_idle()

    def _on_time(self, _value: float) -> None:
        if self._syncing_widgets or self._running:
            return
        if self.active_modal is not None:
            self._workspace_dirty = True
            self._render_metrics()
            return
        if self.cache is not None:
            self._render_current()
        self._render_target()
        self._render_metrics()
        self.fig.canvas.draw_idle()

    def _on_parameter_change(self, _value: float) -> None:
        if self._syncing_widgets or self._running:
            return
        if getattr(self.fig.canvas, "mouse_grabber", None) is not None:
            self._pending_parameter_change = True
            return
        self._apply_parameter_change()

    def _on_button_release(self, _event) -> None:
        if not self._pending_parameter_change:
            return
        self._pending_parameter_change = False
        self._apply_parameter_change()

    def _apply_parameter_change(self) -> None:
        previous = self.parameters
        previous_steps = previous.spline.steps
        def restore_mesh_widgets() -> None:
            self._syncing_widgets = True
            try:
                self.sliders["spline_steps"].set_val(previous.spline.steps)
                self.sliders["regression_steps"].set_val(
                    previous.regression.steps
                )
                self.radios["initialization"].set_active(
                    0 if previous.initialization == "cold" else 1
                )
            finally:
                self._syncing_widgets = False

        try:
            parameters = self._current_parameters()
        except ValueError as error:
            restore_mesh_widgets()
            self._show_message(f"Invalid step count: {error}")
            return
        meshes = [("spline", parameters.spline.steps)]
        if parameters.initialization == "warm":
            meshes.append(("regression", parameters.regression.steps))
        for mesh_name, n_steps in meshes:
            for time in self.series.times:
                if time is None:
                    continue
                exact_step = time * n_steps
                if abs(exact_step - round(exact_step)) > 1e-6:
                    restore_mesh_widgets()
                    self._show_message(
                        f"Target time {time:.3g} is not on the "
                        f"{mesh_name} {n_steps}-step mesh."
                    )
                    return
        self.parameters = parameters
        if self.parameters.spline.steps != previous_steps:
            self.time_slider.valmax = self.parameters.spline.steps
            self.time_slider.ax.set_xlim(0, self.parameters.spline.steps)
            self._refresh_control_widgets()
            self._refresh_observation_widgets()
        self._invalidate("Parameters changed. Press Run.")

    def _on_spline_initialization_change(self, label: str) -> None:
        if self._syncing_widgets:
            return
        if label == "Warm":
            try:
                self._use_minimum_regression_mesh()
            except ValueError as error:
                self._show_message(f"Cannot build regression mesh: {error}")
                return
        self._on_parameter_change(0.0)
        self.fig.canvas.draw_idle()

    def _on_model_change(self, label: str) -> None:
        if self._syncing_widgets or self._running:
            return
        previous = self.parameters
        if label == "Spline" and self.radios["kernel"].value_selected == "Gaussian":
            self._syncing_widgets = True
            self.radios["kernel"].set_active(0)
            self._syncing_widgets = False
        try:
            self.parameters = self._current_parameters()
        except ValueError as error:
            self._syncing_widgets = True
            self.radios["model"].set_active(0 if previous.model == "classic" else 1)
            self.radios["kernel"].set_active(0 if previous.kernel == "sobolev" else 1)
            self._syncing_widgets = False
            self._show_message(f"Invalid model change: {error}")
            return
        self._update_action_labels()
        model_label = "spline" if self.parameters.model == "splines" else "classic"
        self._invalidate(f"Model changed to {model_label}. Press Run.")

    def _update_action_labels(self) -> None:
        model = "SPLINE" if self.parameters.model == "splines" else "CLASSIC"
        self.buttons["run"].label.set_text(f"RUN {model}")
        self.buttons["register"].label.set_text(f"REGISTER {model}")

    def _on_amplitude_change(self, value: float) -> None:
        if self._syncing_widgets or self._running:
            return
        self._drawing_amplitudes[self._active_field_key()] = float(value)
        self._render_panels(self._render_source)
        self.fig.canvas.draw_idle()

    def _on_spacing_change(self, value: float) -> None:
        if self._syncing_widgets or self._running:
            return
        self.renderer.vector_spacing = int(round(value))
        self._render_panels(self._render_current)
        self.fig.canvas.draw_idle()

    def _on_operator_change(self, label: str) -> None:
        if label == "Gaussian" and self.radios["model"].value_selected == "Spline":
            self._syncing_widgets = True
            self.radios["model"].set_active(0)
            self._syncing_widgets = False
        self._on_parameter_change(0.0)
        self._update_action_labels()

    def _on_device_change(self, label: str) -> None:
        if self._syncing_widgets or self._running:
            return
        self.device = label.lower()
        self._set_status("Computing device changed.")
        self.fig.canvas.draw_idle()

    def _refresh_observation_widgets(self) -> None:
        self.observation_menu.editor.set_state(
            self.parameters.spline.steps,
            self.series.names(),
            (0.0, *self.series.times),
            self.series.selected,
        )
        self._sync_control_time_widgets()
        self._refresh_target_markers()

    def _select_target(self, index: int) -> None:
        self.series.selected = min(max(int(index), 0), len(self.series.targets) - 1) + 1
        self._refresh_observation_widgets()
        self._render_panels(self._render_target)
        self.fig.canvas.draw_idle()

    def _select_image(self, index: int) -> None:
        self.series.selected = min(max(int(index), 0), len(self.series.targets))
        if self.series.selected:
            self._render_panels(self._render_target)
        self._refresh_observation_widgets()
        self.fig.canvas.draw_idle()

    def _place_image(self, index: int, time: float) -> None:
        if index == 0:
            if time != 0:
                self._show_message("Replace the source by placing another image at node 0.")
            return
        if time == 0:
            self._promote_target_to_source(index - 1)
        else:
            self._place_target(index - 1, time)

    def _unplace_image(self, index: int) -> None:
        if index == 0:
            self._show_message("The source at node 0 cannot be unplaced.")
            return
        self._unplace_target(index - 1)

    def _promote_target_to_source(self, index: int) -> None:
        if not 0 <= index < len(self.series.targets):
            return
        new_source_path = self.series.paths[index]
        self.series.promote(index)
        if new_source_path:
            with suppress(FileNotFoundError):
                image, _resolved = load_image(new_source_path)
                self.series.source = image.to(dtype=self.series.source.dtype)
        self._set_native_image_size(reload_targets=True)
        self._invalidate("Source image changed. Press Run.")
        self._refresh_observation_widgets()

    def _place_target(self, index: int, time: float) -> None:
        try:
            self.series.place(index, time)
        except ValueError as error:
            self._show_message(str(error))
            return
        if self.parameters.initialization == "warm":
            self._use_minimum_regression_mesh()
        self.last_registration = None
        self._refresh_observation_widgets()
        self._set_status(
            f"Placed target {self.series.number(index)} at t={time:.3g}."
        )
        self._render_panels(self._render_target)
        self.fig.canvas.draw_idle()

    def _unplace_target(self, index: int) -> None:
        if not 0 <= index < len(self.series.times):
            return
        target_number = self.series.number(index)
        self.series.times[index] = None
        if self.parameters.initialization == "warm":
            self._use_minimum_regression_mesh()
        self.last_registration = None
        self._refresh_observation_widgets()
        self._set_status(f"Target {target_number} is now unplaced.")
        self.fig.canvas.draw_idle()

    def remove_selected_image(self) -> None:
        if self.series.selected == 0:
            self._show_message("Replace the source before removing it.")
            return
        if len(self.series.targets) <= 1:
            self._show_message("At least one target image is required.")
            return
        self.series.remove_selected()
        self.last_registration = None
        self._update_target_mse_cache()
        self._refresh_observation_widgets()
        self._render_target()
        self.fig.canvas.draw_idle()

    def _update_target_mse_cache(self) -> None:
        if self.cache is None:
            return
        previous_cache = self.cache
        targets = self.series.targets.to(dtype=self.cache.images.dtype)
        self.cache = replace(
            self.cache,
            target_mse=target_mse(self.cache.images, targets),
        )
        if (
            self.last_registration is not None
            and self.last_registration.trajectory is previous_cache
        ):
            self.last_registration = replace(
                self.last_registration,
                trajectory=self.cache,
            )

    def _set_targets_from_setup(self, setup: SplineSetup) -> None:
        self.series = ImageSeries.from_batch(setup.images)
        self.series.selected = self.target_index + 1
        self._refresh_observation_widgets()

    def _control_fields(self) -> list[torch.Tensor]:
        return [
            self.fields[f"control_jerk:{index}"]
            for index in range(len(self.parameters.control_times))
        ]

    def _control_amplitudes(self) -> list[float]:
        return [
            self._drawing_amplitudes.get(
                f"control_jerk:{index}",
                DEFAULT_DRAWING_AMPLITUDE,
            )
            for index in range(len(self.parameters.control_times))
        ]

    def _replace_control_fields(
        self,
        fields: list[torch.Tensor],
        amplitudes: list[float],
    ) -> None:
        for key in tuple(self.fields):
            if key.startswith("control_jerk:"):
                del self.fields[key]
                self._drawing_amplitudes.pop(key, None)
        for index, field in enumerate(fields):
            key = f"control_jerk:{index}"
            self.fields[key] = field
            self._drawing_amplitudes[key] = amplitudes[index]
        self.editor.clear_history()

    def _add_control_time(self, time: float) -> None:
        times = list(self.parameters.control_times)
        index = bisect_left(times, time)
        times.insert(index, time)
        fields = self._control_fields()
        fields.insert(index, torch.zeros_like(self.fields["initial_jerk"]))
        amplitudes = self._control_amplitudes()
        amplitudes.insert(index, DEFAULT_DRAWING_AMPLITUDE)
        try:
            parameters = replace(self.parameters, control_times=tuple(times))
        except ValueError as error:
            self._show_message(f"Cannot add control time: {error}")
            return
        self.parameters = parameters
        if self.parameters.initialization == "warm":
            self._use_minimum_regression_mesh()
        self._replace_control_fields(fields, amplitudes)
        self.control_index = index
        self._refresh_control_widgets()
        if self.input_kind == "control_jerk":
            self.editor.set_active(self._active_field_key())
            self._sync_amplitude_slider()
        self._invalidate("Control time added with a zero field. Press Run.")

    def _move_control_time(self, index: int, time: float) -> None:
        times = list(self.parameters.control_times)
        if not 0 <= index < len(times):
            return
        times[index] = time
        try:
            self.parameters = replace(
                self.parameters,
                control_times=tuple(times),
            )
            if self.parameters.initialization == "warm":
                self._use_minimum_regression_mesh()
        except ValueError as error:
            self._show_message(f"Cannot move control time: {error}")
            self._refresh_control_widgets()
            return
        self.control_index = index
        self._refresh_control_widgets()
        self._invalidate("Control time moved. Press Run.")

    def _remove_control_time(self, index: int) -> None:
        times = list(self.parameters.control_times)
        fields = self._control_fields()
        amplitudes = self._control_amplitudes()
        if not 0 <= index < len(times):
            return
        times.pop(index)
        fields.pop(index)
        amplitudes.pop(index)
        self.parameters = replace(
            self.parameters,
            control_times=tuple(times),
        )
        if self.parameters.initialization == "warm":
            self._use_minimum_regression_mesh()
        self._replace_control_fields(fields, amplitudes)
        self.control_index = min(index, max(0, len(times) - 1))
        if not times and self.input_kind == "control_jerk":
            self.input_kind = "initial_jerk"
            self._syncing_widgets = True
            self.overlay_radios["input"].set_active(2)
            self._syncing_widgets = False
        self.editor.set_active(self._active_field_key())
        self._sync_amplitude_slider()
        self._refresh_control_widgets()
        self._invalidate("Control time removed. Press Run.")

    def _select_control_time(self, index: int) -> None:
        if not self.parameters.control_times:
            return
        self.control_index = min(max(index, 0), len(self.parameters.control_times) - 1)
        self._sync_control_time_widgets()
        if self.input_kind == "control_jerk":
            self.editor.set_active(self._active_field_key())
            self._sync_amplitude_slider()
            self.jump_to_control(self.control_index)
            self._render_panels(self._render_source)
        self.fig.canvas.draw_idle()

    def _on_key_press(self, event) -> None:
        key = (event.key or "").lower()
        if self._running:
            return
        modal = {
            "p": "parameters",
            "v": "view",
            "l": "files",
            "i": "observations" if self.parameters.model == "splines" else "images",
        }.get(key)
        if modal is not None:
            self._set_modal(None if self.active_modal == modal else modal)
            return
        if key == "escape":
            self._cancel_active_interaction()
            return
        if self.active_modal is not None:
            return
        action = {
            "r": self.run,
            "g": self.register,
            "ctrl+z": self.undo,
            "ctrl+s": lambda: self.save_setup_dialog(quick=True),
            "ctrl+o": self.load_setup_dialog,
            "c": self.clear,
        }.get(key)
        if action is not None:
            action()
            return
        time_offset = {"left": -1, "right": 1}.get(key)
        if time_offset is not None:
            self.set_time_index(int(self.time_slider.val) + time_offset)
        elif key in ("[", "]"):
            self._jump_relative_control(-1 if key == "[" else 1)
        else:
            input_index = {"1": 0, "2": 1, "3": 2, "4": 3}.get(key)
            if input_index is not None:
                self.overlay_radios["input"].set_active(input_index)

    def _cancel_active_interaction(self) -> None:
        if self.active_modal is not None:
            self._set_modal(None)
        elif self.editor.cancel():
            self.fig.canvas.draw_idle()

    def _on_pick(self, event) -> None:
        if self.active_modal is not None or self._running:
            return
        target_index = self._target_markers.get(event.artist)
        if target_index is not None:
            self._select_target(target_index)
            time = self.series.times[target_index]
            if time is not None:
                self.set_time_index(round(time * self.parameters.spline.steps))
            return
        step = self._control_markers.get(event.artist)
        if step is not None:
            self.set_time_index(step)

    def _jump_relative_control(self, direction: int) -> None:
        controls = self.parameters.control_nodes
        if not controls:
            return
        current = int(self.time_slider.val)
        if direction > 0:
            step = next((step for step in controls if step > current), controls[0])
        else:
            step = next(
                (step for step in reversed(controls) if step < current),
                controls[-1],
            )
        self.set_time_index(step)

    def jump_to_control(self, index: int) -> None:
        controls = self.parameters.control_nodes
        if not controls:
            return
        index = min(max(int(index), 0), len(controls) - 1)
        self.set_time_index(controls[index])

    def set_time_index(self, index: int) -> None:
        index = min(max(int(index), 0), self.parameters.spline.steps)
        if int(self.time_slider.val) != index:
            self.time_slider.set_val(index)

    def _refresh_control_widgets(self) -> None:
        controls = self.parameters.control_nodes
        count = len(controls)
        self.control_index = min(self.control_index, max(0, count - 1))
        self._sync_control_time_widgets()
        self._update_overlay_control_selector_visibility()
        self._refresh_control_markers()

    def _update_overlay_control_selector_visibility(self) -> None:
        self.overlay_menu.set_control_selector_visible(
            self.menu_open
            and self.input_kind == "control_jerk"
            and bool(self.parameters.control_times)
        )

    def _sync_control_time_widgets(self) -> None:
        image_steps = tuple(
            round(time * self.parameters.spline.steps)
            for time in self.series.times
            if time is not None
        )
        self.control_time_editor.set_state(
            self.parameters.spline.steps,
            self.parameters.control_nodes,
            self.control_index,
            image_steps=image_steps,
        )
        self.overlay_control_selector.set_state(
            self.parameters.spline.steps,
            self.parameters.control_nodes,
            self.control_index,
        )

    def _refresh_control_markers(self) -> None:
        for artist in self._control_markers:
            with suppress(ValueError):
                artist.remove()
        self._control_markers.clear()
        for step in self.parameters.control_nodes:
            line = self.time_slider.ax.axvline(
                step,
                color=DUAL_COLOR,
                linewidth=1.4,
                alpha=0.8,
                zorder=5,
                picker=5,
            )
            (marker,) = self.time_slider.ax.plot(
                [step],
                [1.12],
                marker="v",
                markersize=7,
                color=DUAL_COLOR,
                transform=self.time_slider.ax.get_xaxis_transform(),
                clip_on=False,
                picker=5,
            )
            self._control_markers[line] = step
            self._control_markers[marker] = step

    def _refresh_target_markers(self) -> None:
        for artist in self._target_markers:
            with suppress(ValueError):
                artist.remove()
        self._target_markers.clear()
        for index, time in enumerate(self.series.times):
            if time is None:
                continue
            step = round(time * self.parameters.spline.steps)
            color = TARGET_ACTIVE_COLOR if index == self.target_index else TARGET_COLOR
            line = self.time_slider.ax.axvline(
                step,
                color=color,
                linewidth=1.2,
                linestyle="--",
                alpha=0.9,
                zorder=6,
                picker=5,
            )
            (marker,) = self.time_slider.ax.plot(
                [step],
                [-0.35],
                marker="^",
                markersize=8 if index == self.target_index else 7,
                color=color,
                transform=self.time_slider.ax.get_xaxis_transform(),
                clip_on=False,
                picker=5,
            )
            self._target_markers[line] = index
            self._target_markers[marker] = index

    def run(self) -> None:
        if self._current_parameters().model == "classic":
            self.run_classic()
        else:
            self.run_spline()

    def run_spline(self) -> None:
        self._run_trajectory("spline", run_spline, "splines")

    def run_classic(self) -> None:
        self._run_trajectory("classic", run_classic, "classic")

    def _run_trajectory(self, label: str, runner, model: str) -> None:
        if not self._begin_operation(
            f"Computing {label}... 0/{self.parameters.spline.steps} (0%)"
        ):
            return
        self.last_registration = None
        try:
            setup = self.make_setup(model)
            self._running_label = label
            trajectory = runner(
                setup,
                device=self.device,
                progress_callback=self._show_run_progress,
            )
            self.cache = trajectory
            self.parameters = setup.parameters
            if model == "splines":
                self._set_targets_from_setup(setup)
            else:
                self._update_target_mse_cache()
            self._set_status(
                f"{label.capitalize()} complete in {trajectory.elapsed_seconds:.3g}s."
            )
        except Exception as error:
            self.cache = None
            self.last_error = error
            self._set_status(f"ERROR: {type(error).__name__}: {error}")
        finally:
            self._finish_operation()
        self._render_current()
        self._render_target()
        self._render_metrics()
        self.fig.canvas.draw_idle()

    def register(self) -> None:
        if self._current_parameters().model == "classic":
            self.register_classic()
        else:
            self.register_spline()

    def register_classic(self) -> None:
        self._run_registration("classic", optimize_classic)

    def register_spline(self) -> None:
        self._run_registration("splines", optimize_spline)

    def _run_registration(self, model: str, runner) -> None:
        model_label = model
        if model == "splines":
            initialization = self._current_parameters().initialization
            model_label = f"{initialization} spline"
        if not self._begin_operation(
            f"Optimizing {model_label} from the images..."
        ):
            return
        try:
            setup = self.make_setup(model)
            self._running_label = f"optimized {model_label} replay"
            result = runner(
                setup,
                device=self.device,
                progress_callback=self._show_run_progress,
            )
            self._apply_registration_result(result)
            self._set_status(
                f"{model_label.capitalize()} registration complete in "
                f"{result.elapsed_seconds:.3g}s. Optimized fields loaded."
            )
        except Exception as error:
            self.last_error = error
            self._set_status(f"ERROR: {type(error).__name__}: {error}")
        finally:
            self._finish_operation()
        self._render()

    def _begin_operation(self, status: str) -> bool:
        if self._running or getattr(self.fig.canvas, "mouse_grabber", None) is not None:
            return False
        self._running = True
        self.editor.cancel()
        self._set_workspace_active(False)
        self._set_status(status)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        self._last_progress_draw = perf_counter()
        self.last_error = None
        return True

    def _finish_operation(self) -> None:
        self._running = False
        self._set_workspace_active(self.active_modal is None)

    def _apply_registration_result(self, result: RegistrationResult) -> None:
        self.parameters = result.setup.parameters
        self._set_fields_from_setup(result.setup)
        self.editor.fields = self.fields
        self.editor.clear_history()
        self.editor.set_active(self._active_field_key())
        self.cache = result.trajectory
        self.last_registration = result
        if result.setup.parameters.model == "splines":
            self._set_targets_from_setup(result.setup)
        else:
            self._update_target_mse_cache()
        self._drawing_amplitudes = dict.fromkeys(
            self.fields, DEFAULT_DRAWING_AMPLITUDE
        )
        self._sync_amplitude_slider()

    def _show_run_progress(self, completed: int, total: int) -> None:
        percent = int(100 * completed / total)
        self._set_status(
            f"Computing {getattr(self, '_running_label', 'spline')}... "
            f"{completed}/{total} ({percent}%)"
        )
        now = perf_counter()
        if completed == total or now - self._last_progress_draw >= 0.1:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            self._last_progress_draw = now

    def undo(self) -> None:
        if not self.editor.undo():
            self._set_status("Nothing to undo for this field.")
            self.fig.canvas.draw_idle()

    def clear(self) -> None:
        if not self.editor.clear():
            self._set_status("The selected field is already clear.")
            self.fig.canvas.draw_idle()

    def clear_all(self) -> None:
        if not self.editor.clear_all():
            self._set_status("All fields are already clear.")
            self.fig.canvas.draw_idle()

    def _render_source(self) -> None:
        self.editor.cancel()
        self.renderer.render_source(
            self.series.source,
            self.editor.field,
            self.input_kind,
            self.control_index,
            self._current_parameters(),
            self._drawing_amplitude(),
            self.show_input_image,
        )

    def _time_index(self) -> int:
        return min(int(round(self.time_slider.val)), self.parameters.spline.steps)

    def _render_current(self) -> None:
        self.renderer.render_current(
            self.series.source,
            self.cache,
            self.current_image_mode,
            self.current_field,
            self._time_index(),
            self.show_current_image,
        )

    def _render_target(self) -> None:
        time = self.series.times[self.target_index]
        loss_curves = None
        regularized_loss_label = "Regularized cost"
        if self.last_registration is not None:
            loss_curves = self.last_registration.loss_curves()
            regularized_loss_label = (
                self.last_registration.regularized_loss_label
            )
        self.renderer.render_target(
            self.series.source,
            self.target,
            self.cache,
            self.current_image_mode,
            self.target_mode,
            self._time_index(),
            self.target_index,
            self.series.number(self.target_index),
            len(self.series.targets),
            time,
            loss_curves,
            self.overlay_checks["target_loss"].get_status(),
            regularized_loss_label,
        )

    def _render_metrics(self) -> None:
        index = self._time_index()
        time = index / self.parameters.spline.steps
        self.time_slider.valtext.set_text(
            f"{index}/{self.parameters.spline.steps}   t={time:.3f}"
        )

    def _render(self) -> None:
        self._render_source()
        self._render_current()
        self._render_target()
        self._render_metrics()
        self.fig.canvas.draw_idle()

    def _choose_file(
        self,
        purpose: str,
        *,
        initial_name: str | None = None,
    ) -> Path | None:
        return choose_file(
            purpose,
            output_path=self.output_path,
            initial_name=initial_name,
        )

    def _set_native_image_size(self, *, reload_targets: bool = False) -> None:
        size = tuple(self.series.source.shape[-2:])
        if reload_targets:
            targets = []
            for index, path in enumerate(self.series.paths):
                target = None
                if path:
                    with suppress(FileNotFoundError):
                        target, _resolved = load_image(path, size)
                if target is None:
                    target = resize_field(
                        self.series.targets[index : index + 1],
                        size,
                        scale_vector_displacement=False,
                    )
                targets.append(target.to(dtype=self.series.source.dtype))
            self.series.targets = torch.cat(targets).contiguous()
        else:
            self.series.targets = resize_field(
                self.series.targets,
                size,
                scale_vector_displacement=False,
            ).contiguous()
        for key, field in self.fields.items():
            self.fields[key] = resize_field(
                field,
                size,
                scale_vector_displacement=False,
            ).contiguous()
        self.editor.clear_history()
        height, width = size
        extent = (-0.5, width - 0.5, -0.5, height - 0.5)
        for axis in self.axes:
            axis.set_xlim(-0.5, width - 0.5)
            axis.set_ylim(-0.5, height - 0.5)
        for image in (self.source_image, self.current_image, self.target_image):
            image.set_extent(extent)

    def load_source(self, path: str | Path) -> None:
        image, resolved = load_image(path)
        self.series.source = image.to(dtype=self.series.source.dtype)
        self.series.source_path = str(resolved)
        self._set_native_image_size()
        self._refresh_observation_widgets()
        self._invalidate(f"Loaded source from {resolved}. Press Run.")

    def load_target(self, path: str | Path) -> None:
        image, resolved = load_image(path, self.series.source.shape[-2:])
        self.series.replace_targets(
            image.to(dtype=self.series.source.dtype),
            str(resolved),
        )
        self.last_registration = None
        self._update_target_mse_cache()
        self._refresh_observation_widgets()
        self._set_status(f"Loaded classic endpoint target from {resolved}.")
        self._render_target()
        self.fig.canvas.draw_idle()

    def add_images(self, paths: tuple[str | Path, ...]) -> None:
        if not paths:
            return
        images = []
        resolved_paths = []
        for path in paths:
            image, resolved = load_image(path, self.series.source.shape[-2:])
            images.append(image.to(dtype=self.series.source.dtype))
            resolved_paths.append(str(resolved))
        self.series.add(images, resolved_paths)
        self.last_registration = None
        self._update_target_mse_cache()
        self._refresh_observation_widgets()
        self._set_status("Images added. Place every unmarked image on the timeline.")
        self._render_target()
        self.fig.canvas.draw_idle()

    def load_timed_directory(self, path: str | Path) -> None:
        batch = load_timed_image_directory(path)
        parameters = replace(self._current_parameters(), model="splines")
        setup = zero_setup(
            batch.source,
            batch.target,
            parameters,
            source_path=batch.source_path,
            target_times=batch.target_times,
            target_paths=batch.target_paths,
        )
        self.apply_setup(setup)
        self._set_status(f"Loaded timed image directory {Path(path).expanduser()}.")

    def save_project(self, path: str | Path) -> Path:
        setup = self.make_setup(preserve_targets=True)
        trajectory = self.cache
        registration = (
            self.last_registration
            if trajectory is not None
            and self.last_registration is not None
            and self.last_registration.trajectory is trajectory
            else None
        )
        if trajectory is not None:
            trajectory = replace(
                trajectory,
                target_mse=trajectory.target_mse[
                    list(self.series.order())
                ],
            )
        destination = save_project_directory(
            setup,
            path,
            trajectory=trajectory,
            registration=registration,
        )
        self._set_status(f"Saved project to {destination}.")
        self.fig.canvas.draw_idle()
        return destination

    def load_project(self, path: str | Path) -> None:
        project = load_project_directory(path)
        self.apply_setup(project.setup)
        self.cache = project.trajectory
        self.last_registration = project.registration
        self.last_error = None
        artifacts = ["setup"]
        if self.cache is not None:
            artifacts.append("trajectory")
        if self.last_registration is not None:
            artifacts.append("optimization")
        self._set_status(
            f"Loaded project {Path(path).expanduser()} ({', '.join(artifacts)})."
        )
        self._render()

    def save_video(self, path: str | Path) -> Path:
        if self.cache is None:
            raise ValueError("run or load a trajectory before saving a video")
        self._set_status("Saving current trajectory video...")
        self.fig.canvas.draw()
        destination = save_current_panel_video(
            path,
            figure=self.fig,
            renderer=self.renderer,
            source=self.series.source,
            trajectory=self.cache,
            image_mode=self.current_image_mode,
            current_field=self.current_field,
            show_image=self.show_current_image,
            restore_index=self._time_index(),
        )
        self._set_status(f"Saved video to {destination}.")
        self.fig.canvas.draw_idle()
        return destination

    def load_field(self, path: str | Path) -> None:
        field = load_scalar_field(
            path,
            self.series.source.shape[-2:],
            dtype=self.series.source.dtype,
        )
        self.editor.replace(field)

    def save_field(self, path: str | Path) -> Path:
        path = Path(path).expanduser()
        if not path.suffix:
            path = path.with_suffix(".pt")
        path.parent.mkdir(parents=True, exist_ok=True)
        role = self._active_field_key()
        torch.save(
            {
                "format_version": 1,
                "field": self.editor.field.detach().cpu().clone(),
                "field_kind": "a" if role == "initial_acceleration" else "u",
                "field_role": role,
            },
            path,
        )
        self._set_status(f"Saved {role} field to {path}.")
        self.fig.canvas.draw_idle()
        return path

    def apply_setup(self, setup: SplineSetup) -> None:
        if max(
            setup.parameters.spline.steps,
            setup.parameters.regression.steps,
        ) > MAX_STEPS:
            raise ValueError(
                f"the interactive playground supports at most {MAX_STEPS} steps"
            )
        self.editor.cancel()
        self._set_setup_state(setup)
        self.editor.fields = self.fields
        self.editor.clear_history()
        self._set_native_image_size()
        self.control_index = 0
        if not self.parameters.control_times and self.input_kind == "control_jerk":
            self.input_kind = "initial_jerk"

        self._syncing_widgets = True
        try:
            self.overlay_radios["input"].set_active(INPUT_FIELDS.index(self.input_kind))
            self.parameter_menu.sync(self.parameters)
            self.time_slider.valmin = 0
            self.time_slider.valmax = self.parameters.spline.steps
            self.time_slider.ax.set_xlim(0, self.parameters.spline.steps)
            self.time_slider.set_val(0)
        finally:
            self._syncing_widgets = False
        self.editor.set_active(self._active_field_key())
        self._sync_amplitude_slider()
        self.cache = None
        self.last_registration = None
        self._refresh_control_widgets()
        self._refresh_observation_widgets()
        self._update_action_labels()
        self._set_status("Setup loaded. Press Run or Register.")
        self._render()

    def save(self, path: str | Path) -> Path:
        saved = save_setup(self.make_setup(preserve_targets=True), path)
        self.output_path = saved
        self._set_status(f"Saved spline setup to {saved}.")
        self.fig.canvas.draw_idle()
        return saved

    def load_setup_path(self, path: str | Path) -> None:
        self.apply_setup(load_setup(path))
        self.output_path = Path(path).expanduser()

    def load_source_dialog(self) -> None:
        self._file_dialog("source", self.load_source, "SOURCE LOAD")

    def load_target_dialog(self) -> None:
        self._file_dialog("target", self.load_target, "TARGET LOAD")

    def add_images_dialog(self) -> None:
        self._choose_and_apply(choose_files, self.add_images, "IMAGE LOAD")

    def load_timed_directory_dialog(self) -> None:
        self._directory_dialog(
            "load_timed_images",
            self.load_timed_directory,
            "DIRECTORY LOAD",
        )

    def load_project_dialog(self) -> None:
        self._directory_dialog("load_project", self.load_project, "PROJECT LOAD")

    def load_field_dialog(self) -> None:
        self._file_dialog("load_field", self.load_field, "FIELD LOAD")

    def save_project_dialog(self) -> None:
        self._directory_dialog("save_project", self.save_project, "PROJECT SAVE")

    def save_field_dialog(self) -> None:
        self._file_dialog("save_field", self.save_field, "FIELD SAVE")

    def save_video_dialog(self) -> None:
        self._file_dialog(
            "save_video",
            self.save_video,
            "VIDEO SAVE",
            initial_name=trajectory_video_filename(
                self.current_field,
                self.show_current_image,
            ),
        )

    def load_setup_dialog(self) -> None:
        self._file_dialog("load_setup", self.load_setup_path, "SETUP LOAD")

    def save_setup_dialog(self, quick: bool = False) -> None:
        self._choose_and_apply(
            lambda: (
                self.output_path
                if quick and self.output_path
                else self._choose_file("save_setup")
            ),
            self.save,
            "SAVE",
        )

    def _file_dialog(
        self,
        purpose: str,
        action,
        label: str,
        *,
        initial_name: str | None = None,
    ) -> None:
        self._choose_and_apply(
            lambda: self._choose_file(purpose, initial_name=initial_name),
            action,
            label,
        )

    def _directory_dialog(self, purpose: str, action, label: str) -> None:
        self._choose_and_apply(
            lambda: choose_directory(purpose),
            action,
            label,
        )

    def _choose_and_apply(self, chooser, action, label: str) -> None:
        try:
            selected = chooser()
            if selected is not None:
                action(selected)
        except Exception as error:
            self._set_status(f"{label} ERROR: {type(error).__name__}: {error}")
            self.fig.canvas.draw_idle()
