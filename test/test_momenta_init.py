import torch
from torch.utils._pytree import tree_flatten, tree_unflatten

from demeter.metamorphosis.var_classes import Momenta


def test_image_rot_trans_scale_affine_none():
    shape = (1, 1, 8, 8)  # 2D
    momenta = Momenta.from_config(shape, diffeo=True, rotation=True, translation=True, scaling=True, device="cpu")

    mom_dict = momenta.as_dict()
    assert set(mom_dict.keys()) == {"momentum_I", "momentum_R", "momentum_T", "momentum_S"}
    assert "momentum_A" not in mom_dict
    assert mom_dict["momentum_I"].shape == shape
    assert mom_dict["momentum_R"].shape == (2, 2)
    assert mom_dict["momentum_T"].shape == (2,)
    assert mom_dict["momentum_S"].shape == (2,)


def test_no_image_rot_trans_scale_affine_none():
    shape = (1, 1, 8, 8)  # 2D
    momenta = Momenta.from_config(shape, diffeo=False, rotation=True, translation=True, scaling=True, device="cpu")

    mom_dict = momenta.as_dict()
    assert set(mom_dict.keys()) == {"momentum_R", "momentum_T", "momentum_S"}
    assert "momentum_I" not in mom_dict
    assert "momentum_A" not in mom_dict
    assert mom_dict["momentum_R"].shape == (2, 2)
    assert mom_dict["momentum_T"].shape == (2,)
    assert mom_dict["momentum_S"].shape == (2,)


def test_image_affine_forces_translation_only():
    shape = (1, 1, 8, 8)  # 2D
    momenta = Momenta.from_config(shape, diffeo=True, affine=True, rotation=True, translation=True, scaling=True, device="cpu")

    mom_dict = momenta.as_dict()
    assert "momentum_I" in mom_dict
    assert "momentum_A" in mom_dict
    assert "momentum_T" in mom_dict  # affine forces translation
    assert "momentum_R" not in mom_dict  # rotation disabled by affine=True
    assert "momentum_S" not in mom_dict  # scaling disabled by affine=True
    assert mom_dict["momentum_I"].shape == shape
    assert mom_dict["momentum_A"].shape == (2, 2)
    assert mom_dict["momentum_T"].shape == (2,)


def test_no_image_affine_forces_translation_only():
    shape = (1, 1, 8, 8)  # 2D
    momenta = Momenta.from_config(shape, diffeo=False, affine=True, rotation=True, translation=True, scaling=True, device="cpu")

    mom_dict = momenta.as_dict()
    assert "momentum_I" not in mom_dict
    assert "momentum_A" in mom_dict
    assert "momentum_T" in mom_dict
    assert "momentum_R" not in mom_dict
    assert "momentum_S" not in mom_dict
    assert mom_dict["momentum_A"].shape == (2, 2)
    assert mom_dict["momentum_T"].shape == (2,)


def test_pytree_registration_handles_missing_fields():
    # Only momentum_I present; others None
    momenta = Momenta(momentum_I=torch.zeros((1, 1, 4, 4), requires_grad=True))

    flat, spec = tree_flatten(momenta)
    assert len(flat) == 1  # only momentum_I is a leaf

    rebuilt = tree_unflatten(flat, spec)
    assert isinstance(rebuilt, Momenta)
    assert rebuilt.momentum_I.shape == (1, 1, 4, 4)
    assert rebuilt.momentum_R is None
    assert rebuilt.momentum_T is None
    assert rebuilt.momentum_S is None
    assert rebuilt.momentum_A is None


def test_checkpoint_with_momenta_pytree():
    # Build an input Momenta with only two tensor fields; others are None.
    mom = Momenta(
        momentum_I=torch.randn((1, 1, 4, 4), requires_grad=True),
        momentum_R=torch.randn((2, 2), requires_grad=True),
    )

    # Flatten once to reuse the same spec; checkpoint will see only leaves.
    leaves, spec = tree_flatten(mom)

    def step_fn(*flat):
        # Rebuild the structured Momenta from checkpoint's flat inputs.
        m = tree_unflatten(flat, spec)
        # Simple differentiable ops on each present tensor.
        out = Momenta(
            momentum_I=m.momentum_I * 2,
            momentum_R=m.momentum_R * 3,
            momentum_T=None,
            momentum_S=None,
            momentum_A=None,
        )
        # Return leaves so checkpoint only handles tensors.
        out_leaves, _ = tree_flatten(out)
        return tuple(out_leaves)

    # Checkpoint with pytree inputs/outputs; unflatten afterward.
    out_flat = torch.utils.checkpoint.checkpoint(step_fn, *leaves, use_reentrant=False)
    out = tree_unflatten(list(out_flat), spec)

    # Backprop to ensure gradients reach the original inputs.
    loss = out.momentum_I.sum() + out.momentum_R.sum()
    loss.backward()

    assert mom.momentum_I.grad is not None
    assert mom.momentum_R.grad is not None
