# Classical registration fields

`draft/export_classic_metamorphosis_fields.py` writes one subdirectory per
registration here by default. Each run stores its playground-ready keyframes
under this hierarchy:

```text
images/
  source.png
  target.png
  final.png
vector/
  momentum/momentum_tNNN.pt
  velocity/velocity_tNNN.pt
scalar/
  momentum/image_momentum_tNNN.pt
  velocity/image_velocity_tNNN.pt
spline_setup.pt  # Sobolev with rho < 1
```

The vector files contain the classical deformation velocity
`v = -sqrt(rho) K(p grad(I))` and its vector momentum. The scalar files contain
image momentum `p` (playground dual kind `u`) and image velocity `A_I p`
(playground primal kind `a`). For Sobolev runs, `m = L v` exactly and
`spline_setup.pt` is directly loadable through the spline lab's **Load complete
setup** action. It contains the optimized `p0` with zero force, jerk, and
controls.

The nodes are uniformly spaced from `t=0` through `t=1`. `trajectory.pt` also
contains the images and all four fields for the complete trajectory.

By default, the target is resized to the source image's native dimensions. Pass
`--size H W` to resize both images to an explicit registration resolution.

Example:

```bash
.venv/bin/python draft/export_classic_metamorphosis_fields.py m0t m1c \
  --kernel sobolev --rho 0.5 --size 128 128 --integration-steps 10
```

Use `--kernel gaussian --sigma 3 3` for the periodic circular Gaussian RKHS
instead. Classical gradients, divergence, transport, and flow composition use
periodic boundaries for both kernel choices.

Load a frame in the playground:

```bash
PYTHONPATH=src:. .venv/bin/python -m draft.playground.field_playground \
  --field draft/playground/registration_fields/reg_test_m0t_to_reg_test_m1c_sobolev_rho_0.5/vector/velocity/velocity_t000.pt
```

Compare the optimized Sobolev momentum's classical geodesic with its
zero-acceleration, zero-jerk spline specialization:

```bash
PYTHONPATH=src:. .venv/bin/python draft/playground/compare_spline_geodesic.py \
  draft/playground/registration_fields/reg_test_m0t_to_reg_test_m1c_sobolev_rho_0.5/
```

The final argument may also be that run's `manifest.json`, `spline_setup.pt`, or
`scalar/momentum/image_momentum_t000.pt`.

The report includes trajectory and endpoint errors for the image, scalar
momentum, and physical velocity. With zero acceleration, zero jerk, and no
controls, the periodic classical and spline trajectories use the same
source-inside-warp update and should agree up to floating-point error.
