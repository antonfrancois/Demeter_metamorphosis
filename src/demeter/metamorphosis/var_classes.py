from dataclasses import dataclass
from logging import warning
from typing import Optional, Sequence, Tuple

import torch
from torch import Tensor
from torch.utils._pytree import register_pytree_node


_MOMENTA_FIELDS: Tuple[str, ...] = (
    "momentum_I",
    "momentum_R",
    "momentum_T",
    "momentum_S",
    "momentum_A",
)


@dataclass
class Momenta:
    """
    Structured container for metamorphosis momenta.

    Optional tensor fields stay compatible with checkpoint/autograd via pytree
    registration; non-present entries are stored in the pytree context.

    Covered init scenarios (see tests):
    - Diffeo + rotation + translation + scaling (no affine) -> I, R, T, S set, A absent.
    - Same but without image momentum -> R, T, S set, I and A absent.
    - Affine requested with image momentum -> I, A, T set; rotation/scaling disabled.
    - Affine requested without image momentum -> A, T set; rotation/scaling disabled.

    Example
    -------
    ```python
    from demeter.metamorphosis.var_classes import Momenta
    import torch

    # Build a pytree-compatible state from config
    mom = Momenta.from_config(
        image_shape=(1, 1, 8, 8),
        diffeo=True,
        rotation=True,
        translation=True,
        scaling=True,
        affine=False,
        device="cpu",
    )

    # Use directly with torch checkpoint
    leaves, spec = torch.utils._pytree.tree_flatten(mom)

    def step_fn(*flat):
        state = torch.utils._pytree.tree_unflatten(flat, spec)
        updated = Momenta(
            momentum_I=state.momentum_I * 0.5,
            momentum_R=state.momentum_R,
            momentum_T=state.momentum_T,
            momentum_S=state.momentum_S,
            momentum_A=state.momentum_A,
        )
        out_leaves, _ = torch.utils._pytree.tree_flatten(updated)
        return tuple(out_leaves)

    out_flat = torch.utils.checkpoint.checkpoint(step_fn, *leaves, use_reentrant=False)
    out = torch.utils._pytree.tree_unflatten(list(out_flat), spec)
    ```
    """

    momentum_I: Optional[Tensor] = None
    momentum_R: Optional[Tensor] = None
    momentum_T: Optional[Tensor] = None
    momentum_S: Optional[Tensor] = None
    momentum_A: Optional[Tensor] = None

    def __post_init__(self):
        for name in _MOMENTA_FIELDS:
            tensor = getattr(self, name)
            if not (isinstance(tensor, Tensor) or tensor is None):
                raise TypeError(f"Momentum.{name} must be a tensor, got {type(tensor)} at initialization")


    def as_dict(self) -> dict:
        """Return a dict that matches the previous API (skips None entries)."""
        return {name: getattr(self, name) for name in _MOMENTA_FIELDS if getattr(self, name) is not None}

    def requires_grad_(self, flag: bool = True) -> "Momenta":
        """Set requires_grad for all present tensor fields."""
        for name in _MOMENTA_FIELDS:
            tensor = getattr(self, name)
            if tensor is not None:
                tensor.requires_grad_(flag)
        return self

    def detach(self) -> "Momenta":
        """Detach all present tensor fields and return self."""
        for name in _MOMENTA_FIELDS:
            tensor = getattr(self, name)
            if tensor is not None:
                setattr(self, name, tensor.detach())
        return self

    def clone(self) -> "Momenta":
        """Return a deep copy of the container with cloned tensors."""
        return Momenta(
            **{
                name: (getattr(self, name).clone() if getattr(self, name) is not None else None)
                for name in _MOMENTA_FIELDS
            }
        )

    def nan_report(self):
        """
        Check NaNs field by field.

        Returns
        -------
        dict
            Keys match momentum names; values are booleans or None when the field is absent.
            Includes an extra key 'any' summarizing if any present tensor contains NaN.
        """
        report = {}
        any_nan = False
        for name in _MOMENTA_FIELDS:
            tensor = getattr(self, name)
            if tensor is None:
                report[name] = None
            else:
                has_nan = bool(tensor.isnan().any().item())
                report[name] = has_nan
                any_nan = any_nan or has_nan
        report["any"] = any_nan
        return report

    def has_nan(self) -> bool:
        """Return True if any present tensor contains NaNs."""
        return self.nan_report()["any"]

    def __repr__(self):
        # Keep a readable summary even when optional tensors are absent
        lines = ["Momenta("]
        for name in _MOMENTA_FIELDS:
            tensor = getattr(self, name)
            if tensor is None:
                lines.append(f"\t{name}=None")
            else:
                lines.append(f"\t{name}=Tensor(shape={tuple(tensor.shape)}, device={tensor.device}, dtype={tensor.dtype})")
        lines.append(")")
        return ",\n".join(lines)

    @classmethod
    def from_config(
        cls,
        image_shape: Sequence[int],
        diffeo: bool = True,
        rotation: bool = True,
        translation: bool = True,
        scaling: bool = True,
        affine: bool = False,
        rot_prior=None,
        trans_prior=None,
        scale_prior=None,
        affine_prior=None,
        device: str = "cuda:0",
        requires_grad: bool = True,
    ) -> "Momenta":
        """
        Rebuilds the old `prepare_momenta` logic but returns a structured pytree.
        """
        dim = 2 if len(image_shape) == 4 else 3
        kwargs = {"dtype": torch.float32, "device": device}

        def _ensure_tensor(value, default):
            if value is None:
                value = default
            return value if torch.is_tensor(value) else torch.tensor(value)

        def _make_skew_from_prior(prior: Tensor, *, allow_matrix: bool):
            if prior.ndim == 2 and allow_matrix:
                return prior.to(**kwargs)
            if prior.ndim <= 1:
                if dim == 2:
                    return torch.tensor([[0, prior], [-prior, 0]], **kwargs)
                r1, r2, r3 = prior
                return torch.tensor([[0, -r1, -r2], [r1, 0, -r3], [r2, r3, 0]], **kwargs)
            raise ValueError("Prior must be 2D matrix or 1D vector.")

        if affine:
            scaling = rotation = False
            translation = True
            warning("affine is true, scaling and rotation set to False, translation set to True")

            affine_prior = _ensure_tensor(affine_prior, torch.zeros((dim, dim)))
            rot_prior = _ensure_tensor(rot_prior, affine_prior)
        else:
            rot_prior = _ensure_tensor(rot_prior, torch.zeros((dim,)) if dim == 3 else torch.tensor([0.0]))
            scale_prior = _ensure_tensor(scale_prior, torch.zeros((dim,)))

        trans_prior = _ensure_tensor(trans_prior, torch.zeros((dim,)))

        momentum_I = torch.zeros(image_shape, **kwargs) if diffeo else None
        momentum_R = None
        if rotation:
            momentum_R = _make_skew_from_prior(rot_prior, allow_matrix=True)

        momentum_A = None
        if affine:
            momentum_A = _make_skew_from_prior(affine_prior, allow_matrix=True)

        momentum_T = trans_prior.to(**kwargs) if translation else None
        momentum_S = scale_prior.to(**kwargs) if scaling else None

        momenta = cls(
            momentum_I=momentum_I,
            momentum_R=momentum_R,
            momentum_T=momentum_T,
            momentum_S=momentum_S,
            momentum_A=momentum_A,
        )
        momenta.requires_grad_(requires_grad)
        return momenta


def _momenta_flatten(momenta: Momenta):
    children = []
    mask = []
    for name in _MOMENTA_FIELDS:
        tensor = getattr(momenta, name)
        present = tensor is not None
        mask.append(present)
        if present:
            children.append(tensor)
    # PyTorch expects (children, context)
    return tuple(children), tuple(mask)


def _momenta_unflatten(children: Tuple[Tensor, ...], mask: Tuple[bool, ...]):
    child_iter = iter(children)
    kwargs = {}
    for name, present in zip(_MOMENTA_FIELDS, mask):
        kwargs[name] = next(child_iter) if present else None
    return Momenta(**kwargs)


register_pytree_node(Momenta, _momenta_flatten, _momenta_unflatten)
