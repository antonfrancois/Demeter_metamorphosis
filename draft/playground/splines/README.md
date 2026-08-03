# Metamorphosis Spline Lab

This playground can either integrate editable shooting fields or optimize them
directly from source and target images. It supports classic endpoint
registration and periodic 2D spline regression against timed observations.

Run it from the repository root:

```bash
.venv/bin/python -m draft.playground.splines m0t m1c \
  --steps 16 --control-steps 4 8 12
```

The default device is `auto`, which selects CUDA when available and CPU
otherwise. Use `--device cpu` or `--device cuda` to override it, `--run` to
integrate immediately, `--register` to optimize, or
`--no-show --screenshot spline.png` for a headless
render. Passing `--control-steps` without values creates a spline without jerk
resets. The status line reports progress, the compute device, and image size.

## Inputs

- **Momentum** edits the dual initial momentum `p0`.
- **Acceleration** edits the actual initial acceleration `a0`; each run computes
  the displayed force `u0 = A_I0^-1 a0`.
- **Jerk** edits the initial jerk `r0`.
- **Control jerk** edits the absolute right limit `r(tau_c+)` at the selected mesh
  node. It is not a jerk increment.
- Left-drag paints positive values, right-drag paints negative values, and
  Shift-drag erases.
- **Clear displayed field** clears only the field named above the source image.
- **Clear all fields** clears every initial and control field while preserving
  the images and parameters.

Press `P` or click **Parameter Menu** for the categorized controls:

- **Model** selects Classic or Spline and contains `rho`, the operator, its
  parameters, the optimized-initial-field checkboxes, and an interactive
  normalized control-time line. Unchecked initial fields are fixed to zero;
  control jerks remain optimized. Gaussian automatically selects Classic
  because spline cometric inversion is Sobolev-only;
- **Draw** contains brush size, amplitude, and vector-arrow spacing. Each
  editable field remembers its own amplitude, shown as `[x...]` in the
  source-panel title;
- **Numerical** contains the registration cost constant, integration `steps`,
  optimization `iterations`, LBFGS learning rate, and the compute-device
  selector. CUDA is selected by default when available; CPU is always
  available. Registration defaults to 10 optimization iterations; the default
  learning rate is `0.1` for Spline and `1.0` for Classic.

On the control-time line, left-click an empty mesh location to add a control,
drag a marker to move it, and right-click a marker to remove it. A new control
starts with a zero field. Control times are stored as normalized values, so a
control at step 8 of 16 steps moves to step 30 when the resolution changes to
60 steps. Resolutions that would collapse controls onto the same mesh node are
rejected.

The two main actions follow the selected model. **Run** integrates the currently
drawn or loaded fields. **Register** starts from the images and zero fields,
optimizes with LBFGS, then loads the optimized fields and a normal replayed
trajectory into the same editor. Classic accepts Sobolev or Gaussian and uses
one endpoint target. Spline requires Sobolev and uses every timed target.

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
vector momentum `m` overlays are orange; primal `a = A_I u` and `v = K m`
overlays are yellow. The force is shown as `u = A_I^-1 a`. Press `M` or `Esc`
to close the menu. The source and current columns each have an independent
image switch; hiding an image leaves its field on a black background without
changing any image mode or field selection.

Field squared norms are shown in LaTeX below the source and current images.
The scalar primal field `a` uses `||.||_{I_t}`, while dual fields `p`, `u`, and
`r` use `||.||_{I_t^*}`; no scalar spline field uses a plain L2 norm. The target
panel shows the normalized MSE for whichever full, deformation-only, or
photometric-only image is currently selected.

The time slider has one value for every state node, including both endpoints.
Orange downward knot markers are clickable; `[` and `]` jump to the previous or
next knot. Blue/purple upward markers are image observations; the currently
displayed target is purple. The target panel shows the first observation at or
to the right of the current node, retaining an observation until the slider
moves past it.

## Persistence

Press `I` or click **Images** for model-specific image actions. Classic shows
only **Load source image** and **Load target image**. Spline shows only
the timed **Manage spline images** view directly, without an intermediate menu.
**Add images** adds unplaced rows; click any mesh node to place the selected
image. Node 0 is the source, marked `[S]`. Placing another image at node 0
promotes it to source and moves the previous source into that image's former
slot. Right-click a non-source row to unplace it.

Press `L` or click **Load / Save** for scalar-field loading, complete-setup
loading, complete-setup saving, and timed-project saving. Image loading is kept
entirely in the separate Images menu.

A timed image directory has this portable form:

```text
series/
  images.csv       # columns: filename,time
  source.png       # the unique row at time 0
  target_001.png   # rows in (0,1]
  target_002.png
```

It can be loaded by the lab, by `--timed-images`, or directly as the `source`
argument of `MetamorphosisSplines`. **Save timed project** writes this format
plus `spline_setup.pt`, and, when available, `trajectory.pt` and
`optimization.pt` in one atomic project directory.

**Save setup** stores source, all timed targets, editable fields, model and
numerical parameters, and normalized control times in one `.pt` file. Version-1
and version-2 setups remain loadable; their saved initial force is converted to
initial acceleration. Version-1 single-target setups remain endpoint
observations. Reopen a setup in the UI or on launch:

```bash
.venv/bin/python -m draft.playground.splines --setup draft/spline_setup.pt
```

**Load field** accepts scalar `.pt`, `.pth`, `.npy`, and `.npz` files supported
by `field_playground_core.py`. Loaded fields are resized to the source image.
The same operation is available without a GUI file dialog, for example:

```bash
.venv/bin/python -m draft.playground.splines m0t m1c \
  --field draft/a0.pt --field-kind acceleration
```

Use `--field-kind control --control-index 1` to initialize a specific control
field; control indices are zero-based.

## Files

- `core.py`: validation, setup persistence, legacy force-to-acceleration migration,
  integration, and detached node-aligned trajectory caches.
- `registration.py`: classic/spline optimizer adapters and normal trajectory replay.
- `project_io.py`: atomic timed-image project saving.
- `editor.py`: reusable scalar painting and per-field undo history.
- `app.py`: application state and interaction coordination.
- `images.py`: image discovery and loading.
- `rendering.py`: panel rendering, overlays, and LaTeX diagnostics.
- `styles.py`: shared colors and display metadata.
- `workspace.py`: persistent image panels, sidebar, timeline, and status layout.
- `menus/`: focused parameter, control-time, image, observation-placement,
  overlay, file, and dialog modules.
- `src/demeter/utils/spline_data.py`: public timed image directory codec.
- `main.py`: CLI construction and launch.
