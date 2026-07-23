# Metamorphosis Spline Lab

This playground edits the shooting data of the periodic 2D spline integrator and
keeps the target image comparison-only. It does not optimize the fields.

Run it from the repository root:

```bash
.venv/bin/python -m draft.playground.splines m0t m1c \
  --steps 16 --control-steps 4 8 12
```

The default device is `auto`, which selects CUDA when available and CPU
otherwise. Use `--device cpu` or `--device cuda` to override it, `--run` to
integrate immediately, or `--no-show --screenshot spline.png` for a headless
render. Passing `--control-steps` without values creates a spline without jerk
resets. The status line reports every completed integration step.

## Inputs

- **Momentum** edits the dual initial momentum `p0`.
- **Force** edits `u0`; each run computes the integrator state
  `a0 = A_I0 u0`.
- **Jerk** edits the initial jerk `r0`.
- **Control jerk** edits the absolute right limit `r(tau_c+)` at the selected mesh
  node. It is not a jerk increment.
- Left-drag paints positive values, right-drag paints negative values, and
  Shift-drag erases.
- **Clear displayed field** clears only the field named above the source image.
- **Clear all fields** clears every initial and control field while preserving
  the images and parameters.

Press `P` or click **Parameter Menu** for the categorized controls:

- **Model** contains `rho`, the Sobolev operator parameters `alpha`, `beta`,
  `gamma`, and an interactive normalized control-time line;
- **Draw** contains brush size and amplitude;
- **Numerical** contains `steps`, from 1 through 40, and the compute-device
  selector. CUDA is selected by default when available; CPU is always available.

On the control-time line, left-click an empty mesh location to add a control,
drag a marker to move it, and right-click a marker to remove it. A new control
starts with a zero field. Control times are stored as normalized values, so a
control at step 8 of 16 steps moves to step 20 when the resolution changes to
40 steps. Resolutions that would collapse controls onto the same mesh node are
rejected.

Press `M` or click **View / Overlay Menu** to open the three-column display
menu. Its source column reuses the control-time line to select which control
jerk field is displayed, without changing the control topology. That selector
is shown only while **Control jerk** is the selected editable field. The current panel
can show:

- the full metamorphosis image `I(t)`;
- the source transported without photometric change, `I_D(t)`;
- the source modified only by the accumulated photometric source,
  `I_phot(t)`, without advection.

The deformation-only image replays the same periodic semi-Lagrangian transport
as the spline integrator. The photometric-only image starts at the source and
accumulates `dt * residuals_stock[k]` at fixed pixels. Dual `p`, `u`, `r`, and
the vector momentum `m = L v` overlays are orange; primal `a = A_I u` and
`v = K m` overlays are yellow. The force is shown as `u = A_I^-1 a`. Press `M`
or `Esc` to close the menu.

Field squared norms are shown in LaTeX below the source and current images. The
target panel shows the normalized MSE for whichever full, deformation-only, or
photometric-only image is currently selected.

The time slider has one value for every state node, including both endpoints.
Orange knot markers are clickable; `[` and `]` jump to the previous or next
knot.

## Persistence

Press `L` or click the sidebar's **Load / Save** button to open the unified file
menu for source and target images, the displayed source field, and complete
setups.

**Save setup** stores source, target, editable fields, numerical parameters, and
normalized control times in one `.pt` file. Legacy setups containing only
control-step indices remain loadable. Reopen a setup in the UI or on launch:

```bash
.venv/bin/python -m draft.playground.splines --setup draft/spline_setup.pt
```

**Load field** accepts scalar `.pt`, `.pth`, `.npy`, and `.npz` files supported
by `field_playground_core.py`. Loaded fields are resized to the source image.
The same operation is available without a GUI file dialog, for example:

```bash
.venv/bin/python -m draft.playground.splines m0t m1c \
  --field draft/u0.pt --field-kind force
```

Use `--field-kind control --control-index 1` to initialize a specific control
field; control indices are zero-based.

## Files

- `core.py`: validation, setup persistence, force-to-acceleration conversion,
  integration, and detached node-aligned trajectory caches.
- `editor.py`: reusable scalar painting and per-field undo history.
- `app.py`: application state and interaction coordination.
- `images.py`: image discovery and loading.
- `rendering.py`: panel rendering, overlays, and LaTeX diagnostics.
- `styles.py`: shared colors and display metadata.
- `workspace.py`: persistent image panels, sidebar, timeline, and status layout.
- `menus/`: focused parameter, control-time, overlay, file, and dialog modules.
- `main.py`: CLI construction and launch.
