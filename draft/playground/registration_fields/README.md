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
```

The vector files contain the classical deformation velocity
`v = -sqrt(rho) K(p grad(I))` and its matched momentum `m = L v`. The scalar
files contain image momentum `p` (playground dual kind `u`) and image velocity
`A_I p` (playground primal kind `a`).

The nodes are uniformly spaced from `t=0` through `t=1`. `trajectory.pt` also
contains the images and all four fields for the complete trajectory.

By default, the target is resized to the source image's native dimensions. Pass
`--size H W` to resize both images to an explicit registration resolution.

Example:

```bash
.venv/bin/python draft/export_classic_metamorphosis_fields.py m0t m1c \
  --rho 0.5 --size 128 128 --integration-steps 10
```

Load a frame in the playground:

```bash
PYTHONPATH=src:. .venv/bin/python -m draft.playground.field_playground \
  --field draft/playground/registration_fields/reg_test_m0t_to_reg_test_m1c_rho_0.5/vector/velocity/velocity_t000.pt
```
