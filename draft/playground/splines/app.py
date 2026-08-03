"""Matplotlib application for editing and viewing metamorphosis splines.

Version: July 23, 2026.
"""

from __future__ import annotations

from dataclasses import replace
from bisect import bisect_left
from pathlib import Path
from time import perf_counter
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
import torch

from demeter.utils.spline_data import load_timed_image_directory

from ..field_playground_core import resize_field
from .core import (
    OPTIMIZABLE_INITIAL_FIELDS,
    SplineParameters,
    SplineSetup,
    SplineTrajectory,
    load_scalar_field,
    load_setup,
    resolve_device,
    run_classic,
    run_spline,
    save_setup,
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
        if setup.parameters.n_steps > MAX_STEPS:
            raise ValueError(
                f"the interactive playground supports at most {MAX_STEPS} steps"
            )
        self.device = str(resolve_device(device))
        self.output_path = Path(output_path).expanduser() if output_path else None
        self.cache: SplineTrajectory | None = None
        self.last_error: Exception | None = None
        self.last_registration: RegistrationResult | None = None
        self.parameters = setup.parameters
        self.source = setup.source.clone()
        self.source_path = setup.source_path
        self._targets = setup.target.clone()
        self.target_times: list[float | None] = list(setup.target_times)
        self.target_paths = list(setup.target_paths)
        self.target_index = 0
        self.image_index = 1
        self.fields: dict[str, torch.Tensor] = {}
        self._set_fields_from_setup(setup)
        self._drawing_amplitudes = {
            key: DEFAULT_DRAWING_AMPLITUDE for key in self.fields
        }

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
            brush=lambda: float(self.brush_slider.val),
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
        return self._targets[self.target_index:self.target_index + 1]

    @property
    def target_path(self) -> str:
        return self.target_paths[self.target_index] if self.target_paths else ""

    def _set_fields_from_setup(self, setup: SplineSetup) -> None:
        self.fields.clear()
        self.fields.update(
            initial_momentum=setup.initial_momentum.clone(),
            initial_acceleration=setup.initial_acceleration.clone(),
            initial_jerk=setup.initial_jerk.clone(),
        )
        for index, field in enumerate(setup.control_jerks):
            self.fields[f"control_jerk:{index}"] = field.clone()

    def _build_figure(self) -> None:
        reserved_keymaps = {
            "keymap.save": {"s", "ctrl+s"},
            "keymap.home": {"r"},
            "keymap.back": {"c", "left"},
            "keymap.forward": {"r", "right"},
            "keymap.pan": {"p"},
        }
        self._original_keymaps = {
            setting: list(plt.rcParams[setting]) for setting in reserved_keymaps
        }
        for setting, reserved in reserved_keymaps.items():
            plt.rcParams[setting] = [
                key for key in plt.rcParams[setting] if key not in reserved
            ]

        workspace = build_workspace(
            self.source,
            self.target,
            self.parameters.n_steps,
            self.parameters.model,
            self._dynamic_artists,
        )
        self.fig = workspace.fig
        self.axes = workspace.axes
        self.source_ax, self.current_ax, self.target_ax = self.axes
        self.source_image, self.current_image, self.target_image = workspace.images
        self.source_footer, self.current_footer, self.target_footer = (
            workspace.footers
        )
        self.parameter_button = workspace.parameter_button
        self.menu_button = workspace.menu_button
        self.image_button = workspace.image_button
        self.file_button = workspace.file_button
        self.run_button = workspace.run_button
        self.register_button = workspace.register_button
        self.clear_button = workspace.clear_button
        self.clear_all_button = workspace.clear_all_button
        self.time_slider = workspace.time_slider
        self.status_text = workspace.status_text
        self.renderer = SplineRenderer(
            self.axes,
            (self.source_image, self.current_image, self.target_image),
            (self.source_footer, self.current_footer, self.target_footer),
            self._dynamic_artists,
        )

        self._build_parameter_menu()
        self._build_overlay_menu()
        self._build_image_menu()
        self._build_file_menu()
        self._build_observation_menu()
        self.input_radio.on_clicked(self._on_input_field)
        self.input_image_toggle.on_clicked(self._on_input_image_toggle)
        self.current_image_toggle.on_clicked(self._on_current_image_toggle)
        self.current_image_radio.on_clicked(self._on_current_image_mode)
        self.current_radio.on_clicked(self._on_current_field)
        self.target_radio.on_clicked(self._on_target_mode)
        self.time_slider.on_changed(self._on_time)
        self.run_button.on_clicked(lambda _event: self.run())
        self.register_button.on_clicked(lambda _event: self.register())
        self.parameter_button.on_clicked(
            lambda _event: self.set_parameter_menu_visible(
                not self.parameter_menu_open
            )
        )
        self.parameter_menu_close_button.on_clicked(
            lambda _event: self.set_parameter_menu_visible(False)
        )
        self.menu_button.on_clicked(
            lambda _event: self.set_menu_visible(not self.menu_open)
        )
        self.menu_close_button.on_clicked(
            lambda _event: self.set_menu_visible(False)
        )
        self.file_button.on_clicked(
            lambda _event: self.set_file_menu_visible(not self.file_menu_open)
        )
        self.image_button.on_clicked(
            lambda _event: self.set_image_menu_visible(
                self.active_modal not in ("images", "observations")
            )
        )
        self.clear_button.on_clicked(lambda _event: self.clear())
        self.clear_all_button.on_clicked(lambda _event: self.clear_all())
        for widget in (
            self.rho_slider,
            self.alpha_slider,
            self.beta_slider,
            self.sigma_slider,
        ):
            widget.on_changed(self._on_parameter_change)
        self.gamma_slider.on_changed(self._on_gamma_change)
        self.cost_slider.on_changed(self._on_cost_change)
        self.lbfgs_lr_slider.on_changed(self._on_lbfgs_lr_change)
        self.model_radio.on_clicked(self._on_model_change)
        self.operator_radio.on_clicked(self._on_operator_change)
        self.optimized_fields_check.on_clicked(self._on_optimized_fields_change)
        self.steps_slider.on_changed(self._on_parameter_change)
        self.iterations_slider.on_changed(self._on_parameter_change)
        self.amplitude_slider.on_changed(self._on_amplitude_change)
        self.spacing_slider.on_changed(self._on_spacing_change)
        self.device_radio.on_clicked(self._on_device_change)

        self._workspace_widgets = [
            self.parameter_button,
            self.image_button,
            self.file_button,
            self.clear_button,
            self.clear_all_button,
            self.menu_button,
            self.run_button,
            self.register_button,
            self.time_slider,
        ]
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
        self.rho_slider = self.parameter_menu.rho_slider
        self.model_radio = self.parameter_menu.model_radio
        self.alpha_slider = self.parameter_menu.alpha_slider
        self.beta_slider = self.parameter_menu.beta_slider
        self.gamma_slider = self.parameter_menu.gamma_slider
        self.sigma_slider = self.parameter_menu.sigma_slider
        self.operator_radio = self.parameter_menu.operator_radio
        self.optimized_fields_check = self.parameter_menu.optimized_fields_check
        self.brush_slider = self.parameter_menu.brush_slider
        self.amplitude_slider = self.parameter_menu.amplitude_slider
        self.spacing_slider = self.parameter_menu.spacing_slider
        self.renderer.vector_spacing = int(self.spacing_slider.val)
        self.steps_slider = self.parameter_menu.steps_slider
        self.iterations_slider = self.parameter_menu.iterations_slider
        self.cost_slider = self.parameter_menu.cost_slider
        self.lbfgs_lr_slider = self.parameter_menu.lbfgs_lr_slider
        self.device_radio = self.parameter_menu.device_radio
        self.control_time_editor = self.parameter_menu.control_time_editor
        self.parameter_menu_close_button = self.parameter_menu.close_button

    def _build_overlay_menu(self) -> None:
        self.overlay_menu = build_overlay_menu(
            self.fig,
            self.parameters,
            on_control_select=self._select_control_time,
        )
        self.input_radio = self.overlay_menu.input_radio
        self.overlay_control_selector = self.overlay_menu.control_time_selector
        self.input_image_toggle = self.overlay_menu.input_image_toggle
        self.current_image_toggle = self.overlay_menu.current_image_toggle
        self.current_image_radio = self.overlay_menu.current_image_radio
        self.current_radio = self.overlay_menu.current_radio
        self.target_radio = self.overlay_menu.target_radio
        self.menu_close_button = self.overlay_menu.close_button

    def _build_file_menu(self) -> None:
        actions = (
            ("LOAD FIELD", lambda: self._run_file_action(self.load_field_dialog)),
            ("LOAD PROJECT", lambda: self._run_file_action(self.load_project_dialog)),
            ("SAVE FIELD", lambda: self._run_file_action(self.save_field_dialog)),
            ("SAVE PROJECT", lambda: self._run_file_action(self.save_project_dialog)),
        )
        self.file_menu = build_file_menu(self.fig, actions)
        self.file_menu.close_button.on_clicked(
            lambda _event: self.set_file_menu_visible(False)
        )

    def _build_image_menu(self) -> None:
        self.image_menu = build_image_menu(self.fig)
        self.image_menu.load_source_button.on_clicked(
            lambda _event: self._run_image_action(self.load_source_dialog)
        )
        self.image_menu.load_target_button.on_clicked(
            lambda _event: self._run_image_action(self.load_target_dialog)
        )
        self.image_menu.manage_spline_button.on_clicked(
            lambda _event: self._set_modal("observations")
        )
        self.image_menu.close_button.on_clicked(
            lambda _event: self.set_image_menu_visible(False)
        )

    def _build_observation_menu(self) -> None:
        self.observation_menu = build_observation_menu(
            self.fig,
            on_select=self._select_image,
            on_place=self._place_image,
            on_unplace=self._unplace_image,
        )
        self.observation_menu.load_directory_button.on_clicked(
            lambda _event: self.load_timed_directory_dialog()
        )
        self.observation_menu.add_images_button.on_clicked(
            lambda _event: self.add_images_dialog()
        )
        self.observation_menu.remove_button.on_clicked(
            lambda _event: self.remove_selected_image()
        )
        self.observation_menu.close_button.on_clicked(
            lambda _event: self.set_image_menu_visible(False)
        )

    def _run_file_action(self, action) -> None:
        self.set_file_menu_visible(False)
        action()

    def _run_image_action(self, action) -> None:
        self.set_image_menu_visible(False)
        action()

    def _connect_events(self) -> None:
        canvas = self.fig.canvas
        canvas.mpl_connect("key_press_event", self._on_key_press)
        canvas.mpl_connect("pick_event", self._on_pick)
        canvas.mpl_connect("button_release_event", self._on_button_release)
        canvas.mpl_connect("resize_event", lambda _event: self.editor.cancel())
        canvas.mpl_connect("close_event", self._restore_keymaps)

    def _restore_keymaps(self, _event=None) -> None:
        for setting, values in self._original_keymaps.items():
            plt.rcParams[setting] = values

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
                and bool(self.parameters.control_steps)
            ),
        )
        self.file_menu.set_visible(self.file_menu_open)
        self.image_menu.set_visible(self.image_menu_open, self.parameters.model)
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

    def _active_field_key(self) -> str:
        if self.input_kind != "control_jerk":
            return self.input_kind
        if not self.parameters.control_steps:
            return "initial_jerk"
        index = min(self.control_index, len(self.parameters.control_steps) - 1)
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
            self.amplitude_slider.set_val(self._drawing_amplitude())
        finally:
            self._syncing_widgets = syncing

    def _current_parameters(self) -> SplineParameters:
        return replace(
            self.parameters,
            alpha=float(self.alpha_slider.val),
            beta=float(self.beta_slider.val),
            gamma=10 ** float(self.gamma_slider.val),
            rho=float(self.rho_slider.val),
            n_steps=int(round(self.steps_slider.val)),
            kernel=self.operator_radio.value_selected.lower(),
            sigma=float(self.sigma_slider.val),
            model=(
                "splines"
                if self.model_radio.value_selected == "Spline"
                else "classic"
            ),
            cost_cst=10 ** float(self.cost_slider.val),
            iterations=int(round(self.iterations_slider.val)),
            lbfgs_lr=10 ** float(self.lbfgs_lr_slider.val),
            optimized_fields=tuple(
                name
                for name, active in zip(
                    OPTIMIZABLE_INITIAL_FIELDS,
                    self.optimized_fields_check.get_status(),
                )
                if active
            ),
        )

    def make_setup(
        self,
        model: str | None = None,
        *,
        preserve_targets: bool = False,
    ) -> SplineSetup:
        parameters = self._current_parameters()
        if model is not None:
            parameters = replace(parameters, model=model)
        if parameters.control_steps:
            controls = torch.stack(
                [
                    self.fields[f"control_jerk:{index}"]
                    for index in range(len(parameters.control_steps))
                ],
                dim=0,
            )
        else:
            controls = self.source.new_zeros((0,) + tuple(self.source.shape))
        if parameters.model == "classic" and not preserve_targets:
            targets = self.target
            target_times = (1.0,)
            target_paths = (self.target_path,)
        else:
            observations = [
                (float(time), index)
                for index, time in enumerate(self.target_times)
                if time is not None
            ]
            if len(observations) != len(self.target_times):
                raise ValueError("place every target image before running a spline")
            order = [index for _time, index in sorted(observations)]
            targets = torch.cat([self._targets[index:index + 1] for index in order])
            target_times = tuple(time for time, _index in sorted(observations))
            target_paths = tuple(self.target_paths[index] for index in order)
        return SplineSetup(
            source=self.source,
            target=targets,
            initial_momentum=self.fields["initial_momentum"],
            initial_acceleration=self.fields["initial_acceleration"],
            initial_jerk=self.fields["initial_jerk"],
            control_jerks=controls,
            parameters=parameters,
            source_path=self.source_path,
            target_path=target_paths[-1],
            target_times=target_times,
            target_paths=target_paths,
        )

    def _set_status(self, message: str) -> None:
        height, width = self.source.shape[-2:]
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
        self._sync_target_to_time()
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
        if selected == "control_jerk" and not self.parameters.control_steps:
            self.input_kind = "initial_jerk"
            self.editor.set_active("initial_jerk")
            self._sync_amplitude_slider()
            self._syncing_widgets = True
            self.input_radio.set_active(2)
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
            self.current_image_radio,
            index,
            "#168a8a",
        )
        if self.cache is not None:
            self._render_panels(self._render_current, self._render_target)
        self.fig.canvas.draw_idle()

    def _on_input_image_toggle(self, _label: str) -> None:
        if self._syncing_widgets:
            return
        self.show_input_image = self.input_image_toggle.get_status()[0]
        self._render_panels(self._render_source)
        self.fig.canvas.draw_idle()

    def _on_current_image_toggle(self, _label: str) -> None:
        if self._syncing_widgets:
            return
        self.show_current_image = self.current_image_toggle.get_status()[0]
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
        set_radio_active_color(self.current_radio, index, color)
        if self.cache is not None:
            self._render_panels(self._render_current)
        self.fig.canvas.draw_idle()

    def _on_target_mode(self, label: str) -> None:
        if self._syncing_widgets:
            return
        self.target_mode = label
        self._render_panels(self._render_target)
        self.fig.canvas.draw_idle()

    def _on_time(self, _value: float) -> None:
        if self._syncing_widgets or self._running:
            return
        if self.active_modal is not None:
            self._workspace_dirty = True
            self._render_metrics()
            return
        self._sync_target_to_time()
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
        previous_steps = self.parameters.n_steps
        try:
            parameters = self._current_parameters()
        except ValueError as error:
            self._syncing_widgets = True
            self.steps_slider.set_val(previous_steps)
            self._syncing_widgets = False
            self._show_message(f"Invalid step count: {error}")
            return
        for time in self.target_times:
            if time is None:
                continue
            exact_step = time * parameters.n_steps
            if abs(exact_step - round(exact_step)) > 1e-6:
                self._syncing_widgets = True
                self.steps_slider.set_val(previous_steps)
                self._syncing_widgets = False
                self._show_message(
                    f"Target time {time:.3g} is not on the {parameters.n_steps}-step mesh."
                )
                return
        self.parameters = parameters
        if self.parameters.n_steps != previous_steps:
            self.time_slider.valmax = self.parameters.n_steps
            self.time_slider.ax.set_xlim(0, self.parameters.n_steps)
            self._refresh_control_widgets()
            self._refresh_observation_widgets()
        self._invalidate("Parameters changed. Press Run.")

    def _on_gamma_change(self, value: float) -> None:
        self.gamma_slider.valtext.set_text(f"{10 ** float(value):.3g}")
        self._on_parameter_change(value)

    def _on_cost_change(self, value: float) -> None:
        self.cost_slider.valtext.set_text(f"{10 ** float(value):.3g}")
        self._on_parameter_change(value)

    def _on_lbfgs_lr_change(self, value: float) -> None:
        self.lbfgs_lr_slider.valtext.set_text(f"{10 ** float(value):.3g}")
        self._on_parameter_change(value)

    def _on_optimized_fields_change(self, _label: str) -> None:
        self._on_parameter_change(0.0)

    def _on_model_change(self, label: str) -> None:
        if self._syncing_widgets or self._running:
            return
        previous = self.parameters
        if label == "Spline" and self.operator_radio.value_selected == "Gaussian":
            self._syncing_widgets = True
            self.operator_radio.set_active(0)
            self._syncing_widgets = False
        try:
            self.parameters = self._current_parameters()
        except ValueError as error:
            self._syncing_widgets = True
            self.model_radio.set_active(0 if previous.model == "classic" else 1)
            self.operator_radio.set_active(0 if previous.kernel == "sobolev" else 1)
            self._syncing_widgets = False
            self._show_message(f"Invalid model change: {error}")
            return
        self._update_action_labels()
        model_label = "spline" if self.parameters.model == "splines" else "classic"
        self._invalidate(f"Model changed to {model_label}. Press Run.")

    def _update_action_labels(self) -> None:
        model = "SPLINE" if self.parameters.model == "splines" else "CLASSIC"
        self.run_button.label.set_text(f"RUN {model}")
        self.register_button.label.set_text(f"REGISTER {model}")

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
        if label == "Gaussian" and self.model_radio.value_selected == "Spline":
            self._syncing_widgets = True
            self.model_radio.set_active(0)
            self._syncing_widgets = False
        self._on_parameter_change(0.0)
        self._update_action_labels()

    def _on_device_change(self, label: str) -> None:
        if self._syncing_widgets or self._running:
            return
        self.device = label.lower()
        self._set_status("Computing device changed.")
        self.fig.canvas.draw_idle()

    def _target_names(self) -> tuple[str, ...]:
        return tuple(
            path or f"target_{index + 1:03d}"
            for index, path in enumerate(self.target_paths)
        )

    def _image_names(self) -> tuple[str, ...]:
        return (self.source_path or "source", *self._target_names())

    def _refresh_observation_widgets(self) -> None:
        if not hasattr(self, "observation_menu"):
            return
        self.observation_menu.editor.set_state(
            self.parameters.n_steps,
            self._image_names(),
            (0.0, *self.target_times),
            self.image_index,
        )
        self._sync_control_time_widgets()
        self._refresh_target_markers()

    def _select_target(self, index: int) -> None:
        if not len(self._targets):
            return
        self.target_index = min(max(int(index), 0), len(self._targets) - 1)
        self.image_index = self.target_index + 1
        self._refresh_observation_widgets()
        self._render_panels(self._render_target)
        self.fig.canvas.draw_idle()

    def _select_image(self, index: int) -> None:
        self.image_index = min(max(int(index), 0), len(self._targets))
        if self.image_index > 0:
            self.target_index = self.image_index - 1
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
        if not 0 <= index < len(self._targets):
            return
        old_source = self.source.clone()
        old_source_path = self.source_path
        new_source_path = self.target_paths[index]
        new_source = self._targets[index:index + 1].clone()
        if new_source_path:
            try:
                new_source, _resolved = load_image(new_source_path)
            except FileNotFoundError:
                pass
        self.source = new_source.to(dtype=self.source.dtype)
        self.source_path = new_source_path
        self._targets[index:index + 1] = old_source
        self.target_paths[index] = old_source_path
        self.image_index = 0
        self._set_native_image_size(reload_targets=True)
        self._invalidate("Source image changed. Press Run.")
        self._refresh_observation_widgets()

    def _place_target(self, index: int, time: float) -> None:
        for other_index, other_time in enumerate(self.target_times):
            if other_index != index and other_time is not None and abs(other_time - time) < 1e-8:
                self._show_message("Another target already occupies that node.")
                return
        self.target_times[index] = float(time)
        self.target_index = index
        self.image_index = index + 1
        self.last_registration = None
        self._refresh_observation_widgets()
        self._set_status(f"Placed target {index + 1} at t={time:.3g}.")
        self._render_panels(self._render_target)
        self.fig.canvas.draw_idle()

    def _unplace_target(self, index: int) -> None:
        if not 0 <= index < len(self.target_times):
            return
        self.target_times[index] = None
        self.last_registration = None
        self._refresh_observation_widgets()
        self._set_status(f"Target {index + 1} is now unplaced.")
        self.fig.canvas.draw_idle()

    def remove_selected_image(self) -> None:
        if self.image_index == 0:
            self._show_message("Replace the source before removing it.")
            return
        self.target_index = self.image_index - 1
        if len(self._targets) <= 1:
            self._show_message("At least one target image is required.")
            return
        keep = [index for index in range(len(self._targets)) if index != self.target_index]
        self._targets = torch.cat([self._targets[index:index + 1] for index in keep])
        self.target_times = [self.target_times[index] for index in keep]
        self.target_paths = [self.target_paths[index] for index in keep]
        self.target_index = min(self.target_index, len(self._targets) - 1)
        self.image_index = self.target_index + 1
        self.last_registration = None
        self._update_target_mse_cache()
        self._refresh_observation_widgets()
        self._render_target()
        self.fig.canvas.draw_idle()

    def _update_target_mse_cache(self) -> None:
        if self.cache is None:
            return
        previous_cache = self.cache
        targets = self._targets.to(dtype=self.cache.images.dtype)
        mse = torch.stack(
            [
                (self.cache.images - target).square().mean(dim=(1, 2, 3))
                for target in targets
            ]
        )
        self.cache = replace(self.cache, target_mse=mse)
        if (
            self.last_registration is not None
            and self.last_registration.trajectory is previous_cache
        ):
            self.last_registration = replace(
                self.last_registration,
                trajectory=self.cache,
            )

    def _sync_target_to_time(self) -> None:
        placed = sorted(
            (round(time * self.parameters.n_steps), index)
            for index, time in enumerate(self.target_times)
            if time is not None
        )
        if not placed:
            return
        current = self._time_index()
        target_index = next(
            (index for step, index in placed if step >= current),
            placed[-1][1],
        )
        if target_index != self.target_index:
            self.target_index = target_index
            self._refresh_observation_widgets()

    def _set_targets_from_setup(self, setup: SplineSetup) -> None:
        self._targets = setup.target.clone()
        self.target_times = list(setup.target_times)
        self.target_paths = list(setup.target_paths)
        self.target_index = min(self.target_index, len(self._targets) - 1)
        self._sync_target_to_time()
        self.image_index = self.target_index + 1
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
            control_steps=(),
            control_times=tuple(times),
        )
        self._replace_control_fields(fields, amplitudes)
        self.control_index = min(index, max(0, len(times) - 1))
        if not times and self.input_kind == "control_jerk":
            self.input_kind = "initial_jerk"
            self._syncing_widgets = True
            self.input_radio.set_active(2)
            self._syncing_widgets = False
        self.editor.set_active(self._active_field_key())
        self._sync_amplitude_slider()
        self._refresh_control_widgets()
        self._invalidate("Control time removed. Press Run.")

    def _select_control_time(self, index: int) -> None:
        if not self.parameters.control_steps:
            return
        self.control_index = min(max(index, 0), len(self.parameters.control_steps) - 1)
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
        if key == "p":
            self.set_parameter_menu_visible(not self.parameter_menu_open)
            return
        if key == "m":
            self.set_menu_visible(not self.menu_open)
            return
        if key == "l":
            self.set_file_menu_visible(not self.file_menu_open)
            return
        if key == "i":
            self.set_image_menu_visible(
                self.active_modal not in ("images", "observations")
            )
            return
        if key == "escape" and self.active_modal is not None:
            self._set_modal(None)
            return
        if self.active_modal is not None:
            return
        if key == "r":
            self.run()
        elif key == "g":
            self.register()
        elif key == "ctrl+z":
            self.undo()
        elif key == "ctrl+s":
            self.save_setup_dialog(quick=True)
        elif key == "ctrl+o":
            self.load_setup_dialog()
        elif key == "c":
            self.clear()
        elif key == "left":
            self.set_time_index(int(self.time_slider.val) - 1)
        elif key == "right":
            self.set_time_index(int(self.time_slider.val) + 1)
        elif key == "[":
            self._jump_relative_control(-1)
        elif key == "]":
            self._jump_relative_control(1)
        elif key in ("1", "2", "3", "4"):
            self.input_radio.set_active(int(key) - 1)
        elif key == "escape":
            if self.editor.cancel():
                self.fig.canvas.draw_idle()

    def _on_pick(self, event) -> None:
        if self.active_modal is not None or self._running:
            return
        target_index = self._target_markers.get(event.artist)
        if target_index is not None:
            self._select_target(target_index)
            time = self.target_times[target_index]
            if time is not None:
                self.set_time_index(round(time * self.parameters.n_steps))
            return
        step = self._control_markers.get(event.artist)
        if step is not None:
            self.set_time_index(step)

    def _jump_relative_control(self, direction: int) -> None:
        controls = self.parameters.control_steps
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
        controls = self.parameters.control_steps
        if not controls:
            return
        index = min(max(int(index), 0), len(controls) - 1)
        self.set_time_index(controls[index])

    def set_time_index(self, index: int) -> None:
        index = min(max(int(index), 0), self.parameters.n_steps)
        if int(self.time_slider.val) != index:
            self.time_slider.set_val(index)

    def _refresh_control_widgets(self) -> None:
        controls = self.parameters.control_steps
        count = len(controls)
        self.control_index = min(self.control_index, max(0, count - 1))
        self._sync_control_time_widgets()
        self._update_overlay_control_selector_visibility()
        self._refresh_control_markers()
        self._refresh_target_markers()

    def _update_overlay_control_selector_visibility(self) -> None:
        self.overlay_menu.set_control_selector_visible(
            self.menu_open
            and self.input_kind == "control_jerk"
            and bool(self.parameters.control_steps)
        )

    def _sync_control_time_widgets(self) -> None:
        image_steps = tuple(
            round(time * self.parameters.n_steps)
            for time in self.target_times
            if time is not None
        )
        self.control_time_editor.set_state(
            self.parameters.n_steps,
            self.parameters.control_steps,
            self.control_index,
            image_steps=image_steps,
        )
        self.overlay_control_selector.set_state(
            self.parameters.n_steps,
            self.parameters.control_steps,
            self.control_index,
        )

    def _refresh_control_markers(self) -> None:
        for artist in self._control_markers:
            try:
                artist.remove()
            except ValueError:
                pass
        self._control_markers.clear()
        for step in self.parameters.control_steps:
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
        if not hasattr(self, "time_slider"):
            return
        for artist in self._target_markers:
            try:
                artist.remove()
            except ValueError:
                pass
        self._target_markers.clear()
        for index, time in enumerate(self.target_times):
            if time is None:
                continue
            step = round(time * self.parameters.n_steps)
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
        if (
            self._running
            or getattr(self.fig.canvas, "mouse_grabber", None) is not None
        ):
            return
        self._running = True
        self.editor.cancel()
        self._set_workspace_active(False)
        self._set_status(
            f"Computing {label}... 0/{self.parameters.n_steps} (0%)"
        )
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        self._last_progress_draw = perf_counter()
        self.last_error = None
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
            self._running = False
            self._set_workspace_active(self.active_modal is None)
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
        if self._running or getattr(self.fig.canvas, "mouse_grabber", None) is not None:
            return
        self._running = True
        self.editor.cancel()
        self._set_workspace_active(False)
        model_label = "spline" if model == "splines" else model
        self._set_status(f"Optimizing {model_label} from the images...")
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        self._last_progress_draw = perf_counter()
        self.last_error = None
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
            self._running = False
            self._set_workspace_active(self.active_modal is None)
        self._render()

    def _apply_registration_result(self, result: RegistrationResult) -> None:
        self.parameters = result.setup.parameters
        self._set_fields_from_setup(result.setup)
        self.editor.fields = self.fields
        self.editor.clear_history()
        self.editor.set_active(self._active_field_key())
        self.cache = result.trajectory
        self.last_registration = result
        if result.model == "splines":
            self._set_targets_from_setup(result.setup)
        else:
            self._update_target_mse_cache()
        self._drawing_amplitudes = {
            key: DEFAULT_DRAWING_AMPLITUDE for key in self.fields
        }
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
            self.source,
            self.editor.field,
            self.input_kind,
            self.control_index,
            self._current_parameters(),
            self._drawing_amplitude(),
            self.show_input_image,
        )

    def _time_index(self) -> int:
        return min(int(round(self.time_slider.val)), self.parameters.n_steps)

    def _render_current(self) -> None:
        self.renderer.render_current(
            self.source,
            self.cache,
            self.current_image_mode,
            self.current_field,
            self._time_index(),
            self.show_current_image,
        )

    def _render_target(self) -> None:
        time = self.target_times[self.target_index]
        self.renderer.render_target(
            self.source,
            self.target,
            self.cache,
            self.current_image_mode,
            self.target_mode,
            self._time_index(),
            self.target_index,
            len(self._targets),
            time,
        )

    def _render_metrics(self) -> None:
        index = self._time_index()
        time = index / self.parameters.n_steps
        self.time_slider.valtext.set_text(
            f"{index}/{self.parameters.n_steps}   t={time:.3f}"
        )

    def _render(self) -> None:
        self._render_source()
        self._render_current()
        self._render_target()
        self._render_metrics()
        self.fig.canvas.draw_idle()

    def _choose_file(self, purpose: str) -> Path | None:
        try:
            return choose_file(purpose, output_path=self.output_path)
        except RuntimeError as error:
            self._set_status(str(error))
            self.fig.canvas.draw_idle()
            return None

    def _set_native_image_size(self, *, reload_targets: bool = False) -> None:
        size = tuple(self.source.shape[-2:])
        if reload_targets:
            targets = []
            for index, path in enumerate(self.target_paths):
                target = None
                if path:
                    try:
                        target, _resolved = load_image(path, size)
                    except FileNotFoundError:
                        pass
                if target is None:
                    target = resize_field(
                        self._targets[index:index + 1],
                        size,
                        scale_vector_displacement=False,
                    )
                targets.append(target.to(dtype=self.source.dtype))
            self._targets = torch.cat(targets).contiguous()
        else:
            self._targets = resize_field(
                self._targets,
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
        self.source = image.to(dtype=self.source.dtype)
        self.source_path = str(resolved)
        self._set_native_image_size()
        self._refresh_observation_widgets()
        self._invalidate(f"Loaded source from {resolved}. Press Run.")

    def load_target(self, path: str | Path) -> None:
        image, resolved = load_image(path, self.source.shape[-2:])
        self._targets = image.to(dtype=self.source.dtype)
        self.target_times = [1.0]
        self.target_paths = [str(resolved)]
        self.target_index = 0
        self.image_index = 1
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
            image, resolved = load_image(path, self.source.shape[-2:])
            images.append(image.to(dtype=self.source.dtype))
            resolved_paths.append(str(resolved))
        self._targets = torch.cat((self._targets, *images))
        self.target_times.extend([None] * len(images))
        self.target_paths.extend(resolved_paths)
        self.target_index = len(self._targets) - len(images)
        self.image_index = self.target_index + 1
        self.last_registration = None
        self._update_target_mse_cache()
        self._refresh_observation_widgets()
        self._set_status("Images added. Place every unmarked image on the timeline.")
        self._render_target()
        self.fig.canvas.draw_idle()

    def load_timed_directory(self, path: str | Path) -> None:
        batch = load_timed_image_directory(path)
        parameters = replace(self._current_parameters(), model="splines")
        setup = SplineSetup(
            source=batch.source,
            target=batch.target,
            initial_momentum=torch.zeros_like(batch.source),
            initial_acceleration=torch.zeros_like(batch.source),
            initial_jerk=torch.zeros_like(batch.source),
            control_jerks=batch.source.new_zeros(
                (len(parameters.control_steps),) + tuple(batch.source.shape)
            ),
            parameters=parameters,
            source_path=batch.source_path,
            target_path=batch.target_paths[-1],
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
            timed_indices = [
                (float(time), index)
                for index, time in enumerate(self.target_times)
                if time is not None
            ]
            order = [index for _time, index in sorted(timed_indices)]
            trajectory = replace(
                trajectory,
                target_mse=trajectory.target_mse[order],
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

    def load_field(self, path: str | Path) -> None:
        field = load_scalar_field(
            path,
            self.source.shape[-2:],
            dtype=self.source.dtype,
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
        if setup.parameters.n_steps > MAX_STEPS:
            raise ValueError(
                f"the interactive playground supports at most {MAX_STEPS} steps"
            )
        self.editor.cancel()
        self.parameters = setup.parameters
        self.source = setup.source.clone()
        self.source_path = setup.source_path
        self._targets = setup.target.clone()
        self.target_times = list(setup.target_times)
        self.target_paths = list(setup.target_paths)
        self.target_index = 0
        self.image_index = 1
        self._set_fields_from_setup(setup)
        self._drawing_amplitudes = {
            key: DEFAULT_DRAWING_AMPLITUDE for key in self.fields
        }
        self.editor.fields = self.fields
        self.editor.clear_history()
        self._set_native_image_size()
        self.control_index = 0
        if not self.parameters.control_steps and self.input_kind == "control_jerk":
            self.input_kind = "initial_jerk"

        self._syncing_widgets = True
        try:
            input_index = INPUT_FIELDS.index(self.input_kind)
            self.input_radio.set_active(input_index)
            self.rho_slider.valmax = max(
                0.95,
                self.parameters.rho,
                self.rho_slider.valmax,
            )
            self.rho_slider.ax.set_xlim(
                self.rho_slider.valmin,
                self.rho_slider.valmax,
            )
            self.rho_slider.set_val(self.parameters.rho)
            self._set_slider_value(
                self.alpha_slider,
                self.parameters.alpha,
                lower_bound=0,
            )
            self._set_slider_value(
                self.beta_slider,
                self.parameters.beta,
                lower_bound=0,
            )
            log_gamma = float(np.log10(self.parameters.gamma))
            self._set_slider_value(self.gamma_slider, log_gamma, padding=1)
            self.gamma_slider.valtext.set_text(f"{self.parameters.gamma:.3g}")
            self._set_slider_value(
                self.sigma_slider,
                self.parameters.sigma,
                padding=max(0.05, 0.5 * self.parameters.sigma),
            )
            self.operator_radio.set_active(
                0 if self.parameters.kernel == "sobolev" else 1
            )
            self.model_radio.set_active(
                0 if self.parameters.model == "classic" else 1
            )
            optimized_fields = set(self.parameters.optimized_fields)
            for index, (name, active) in enumerate(
                zip(
                    OPTIMIZABLE_INITIAL_FIELDS,
                    self.optimized_fields_check.get_status(),
                )
            ):
                if active != (name in optimized_fields):
                    self.optimized_fields_check.set_active(index)
            log_cost = float(np.log10(max(self.parameters.cost_cst, 1e-12)))
            self._set_slider_value(self.cost_slider, log_cost, padding=1)
            self.cost_slider.valtext.set_text(f"{self.parameters.cost_cst:.3g}")
            assert self.parameters.lbfgs_lr is not None
            log_lbfgs_lr = float(np.log10(self.parameters.lbfgs_lr))
            self._set_slider_value(
                self.lbfgs_lr_slider,
                log_lbfgs_lr,
                padding=1,
            )
            self.lbfgs_lr_slider.valtext.set_text(
                f"{self.parameters.lbfgs_lr:.3g}"
            )
            self.steps_slider.valmin = 1
            self.steps_slider.valmax = MAX_STEPS
            self.steps_slider.ax.set_xlim(1, MAX_STEPS)
            self.steps_slider.set_val(self.parameters.n_steps)
            self._set_slider_value(
                self.iterations_slider,
                self.parameters.iterations,
                padding=1,
                lower_bound=1,
            )
            self.time_slider.valmin = 0
            self.time_slider.valmax = self.parameters.n_steps
            self.time_slider.ax.set_xlim(0, self.parameters.n_steps)
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

    @staticmethod
    def _set_slider_value(
        slider: Slider,
        value: float,
        padding: float | None = None,
        lower_bound: float | None = None,
    ) -> None:
        padding = max(1, abs(value) * 0.5) if padding is None else padding
        slider.valmin = min(slider.valmin, value - padding)
        if lower_bound is not None:
            slider.valmin = max(lower_bound, slider.valmin)
        slider.valmax = max(slider.valmax, value + padding)
        slider.ax.set_xlim(slider.valmin, slider.valmax)
        slider.set_val(value)

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
        path = self._choose_file("source")
        if path is not None:
            self._dialog_action(self.load_source, path, "SOURCE LOAD")

    def load_target_dialog(self) -> None:
        path = self._choose_file("target")
        if path is not None:
            self._dialog_action(self.load_target, path, "TARGET LOAD")

    def add_images_dialog(self) -> None:
        try:
            paths = choose_files()
            self.add_images(paths)
        except Exception as error:
            self._set_status(f"IMAGE LOAD ERROR: {type(error).__name__}: {error}")
            self.fig.canvas.draw_idle()

    def load_timed_directory_dialog(self) -> None:
        try:
            path = choose_directory("load_timed_images")
            if path is not None:
                self.load_timed_directory(path)
        except Exception as error:
            self._set_status(f"DIRECTORY LOAD ERROR: {type(error).__name__}: {error}")
            self.fig.canvas.draw_idle()

    def load_project_dialog(self) -> None:
        try:
            path = choose_directory("load_project")
            if path is not None:
                self.load_project(path)
        except Exception as error:
            self._set_status(f"PROJECT LOAD ERROR: {type(error).__name__}: {error}")
            self.fig.canvas.draw_idle()

    def save_project_dialog(self) -> None:
        try:
            path = choose_directory("save_project")
            if path is not None:
                self.save_project(path)
        except Exception as error:
            self._set_status(f"PROJECT SAVE ERROR: {type(error).__name__}: {error}")
            self.fig.canvas.draw_idle()

    def load_field_dialog(self) -> None:
        path = self._choose_file("load_field")
        if path is not None:
            self._dialog_action(self.load_field, path, "FIELD LOAD")

    def save_field_dialog(self) -> None:
        path = self._choose_file("save_field")
        if path is not None:
            self._dialog_action(self.save_field, path, "FIELD SAVE")

    def load_setup_dialog(self) -> None:
        path = self._choose_file("load_setup")
        if path is not None:
            self._dialog_action(self.load_setup_path, path, "SETUP LOAD")

    def save_setup_dialog(self, quick: bool = False) -> None:
        path = self.output_path if quick and self.output_path else self._choose_file("save_setup")
        if path is not None:
            self._dialog_action(self.save, path, "SAVE")

    def _dialog_action(self, action, path: Path, label: str) -> None:
        try:
            action(path)
        except Exception as error:
            self._set_status(f"{label} ERROR: {type(error).__name__}: {error}")
            self.fig.canvas.draw_idle()
