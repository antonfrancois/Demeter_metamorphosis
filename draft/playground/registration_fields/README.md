# Classical registration fields

`draft/export_classic_metamorphosis_fields.py` writes one subdirectory per
registration here by default. Each time node contains two playground-ready
files:

- `velocity_tNNN.pt`: the classical deformation velocity
  `v = -sqrt(rho) K(p grad(I))`.
- `momentum_tNNN.pt`: its matched vector momentum `m = L v`.

The nodes are uniformly spaced from `t=0` through `t=1`. `trajectory.pt` also
contains the scalar image momentum `p`, images, vector momenta, velocities, and
times for the complete trajectory.

Example:

```bash
.venv/bin/python draft/export_classic_metamorphosis_fields.py m0t m1c \
  --rho 0.5 --size 128 128 --integration-steps 10
```

Load a frame in the playground:

```bash
PYTHONPATH=src:. .venv/bin/python -m draft.playground.field_playground \
  --field draft/playground/registration_fields/reg_test_m0t_to_reg_test_m1c_rho_0.5/velocity_t000.pt
```
