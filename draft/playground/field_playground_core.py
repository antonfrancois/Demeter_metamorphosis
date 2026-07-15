"""Pure tensor and file operations used by the field playground."""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ..sobolevfluid_operator import SobolevFluidOperator


VECTOR_KINDS = ("velocity", "vector_momentum")
SCALAR_KINDS = ("a", "u")
FORMAT_VERSION = 1


@dataclass
class LoadedField:
    field: torch.Tensor
    kind: str
    image: torch.Tensor | None = None
    metadata: dict[str, Any] = dataclass_field(default_factory=dict)


@dataclass
class AnalysisResult:
    counterpart: torch.Tensor
    roundtrip: torch.Tensor
    kernel_response: torch.Tensor | None
    relative_roundtrip: float
    squared_norm: float
    solver_iterations: int | None = None
    solver_time: float | None = None


def mode_for_kind(kind: str) -> str:
    if kind in VECTOR_KINDS:
        return "vector"
    if kind in SCALAR_KINDS:
        return "scalar"
    raise ValueError(f"unsupported field kind {kind!r}")


def _canonical_kind(value: Any, mode: str, *, strict: bool = False) -> str:
    text = str(value).lower().strip()
    aliases = {
        "v": "velocity",
        "momentum": "vector_momentum",
        "m": "vector_momentum",
        "acceleration": "a",
        "scalar_momentum": "u",
        "residual": "u",
        "residuals": "u",
        "primal": "velocity" if mode == "vector" else "a",
        "dual": "vector_momentum" if mode == "vector" else "u",
    }
    kind = aliases.get(text, text)
    allowed = VECTOR_KINDS if mode == "vector" else SCALAR_KINDS
    if kind in allowed:
        return kind
    if strict:
        raise ValueError(f"field kind {value!r} is incompatible with a {mode} field")
    return allowed[0]


def coerce_image(value: Any) -> torch.Tensor:
    image = torch.as_tensor(value, dtype=torch.float32).detach().cpu()
    if image.ndim == 2:
        image = image[None, None]
    elif image.ndim == 3 and image.shape[0] in (1, 3, 4):
        image = image[:3].mean(dim=0, keepdim=True)[None]
    elif image.ndim == 3 and image.shape[-1] in (1, 3, 4):
        image = image[..., :3].mean(dim=-1)[None, None]
    elif image.ndim == 4 and image.shape[0] == 1:
        if image.shape[1] > 1:
            image = image[:, :3].mean(dim=1, keepdim=True)
    else:
        raise ValueError(f"image must represent one 2D image, got {tuple(image.shape)}")
    if image.shape[1] != 1:
        raise ValueError(f"image must have one channel, got {tuple(image.shape)}")
    if not torch.isfinite(image).all():
        raise ValueError("image contains non-finite values")
    minimum, maximum = image.amin(), image.amax()
    if maximum > 1 or minimum < 0:
        span = maximum - minimum
        image = (image - minimum) / span if span > 0 else torch.zeros_like(image)
    return image.contiguous()


def coerce_field(value: Any) -> tuple[torch.Tensor, str]:
    tensor = torch.as_tensor(value, dtype=torch.float32).detach().cpu()
    if not torch.isfinite(tensor).all():
        raise ValueError("field contains non-finite values")
    if tensor.ndim == 2:
        return tensor[None, None].contiguous(), "scalar"
    if tensor.ndim == 3 and tensor.shape[0] in (1, 2):
        return tensor[None].contiguous(), "scalar" if tensor.shape[0] == 1 else "vector"
    if tensor.ndim == 3 and tensor.shape[-1] == 2:
        return tensor.permute(2, 0, 1)[None].contiguous(), "vector"
    if tensor.ndim == 4 and tensor.shape[0] == 1 and tensor.shape[1] in (1, 2):
        return tensor.contiguous(), "scalar" if tensor.shape[1] == 1 else "vector"
    if tensor.ndim == 4 and tensor.shape[0] == 1 and tensor.shape[-1] == 2:
        return tensor.permute(0, 3, 1, 2).contiguous(), "vector"
    raise ValueError(
        "field must have shape [H,W], [C,H,W], [H,W,2], "
        f"[1,C,H,W], or [1,H,W,2]; got {tuple(tensor.shape)}"
    )


def resize_field(
    field: torch.Tensor,
    size: tuple[int, int],
    *,
    scale_vector_displacement: bool = True,
) -> torch.Tensor:
    old_height, old_width = field.shape[-2:]
    if (old_height, old_width) == tuple(size):
        return field
    resized = F.interpolate(field, size=size, mode="bilinear", align_corners=False)
    if field.shape[1] == 2 and scale_vector_displacement:
        resized[:, 0] *= size[1] / old_width
        resized[:, 1] *= size[0] / old_height
    return resized


def load_field_file(path: str | Path) -> LoadedField:
    path = Path(path).expanduser().resolve()
    suffix = path.suffix.lower()
    if suffix == ".npy":
        payload: Any = np.load(path, allow_pickle=False)
    elif suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            key = "field" if "field" in archive.files else archive.files[0]
            payload = archive[key].copy()
    else:
        payload = torch.load(path, map_location="cpu", weights_only=True)

    metadata: dict[str, Any] = {}
    image = None
    kind_hint = None
    value = payload
    if isinstance(payload, dict):
        version = payload.get("format_version")
        if version is not None and int(version) > FORMAT_VERSION:
            raise ValueError(f"unsupported field format version {version}")
        convention = payload.get("dx_convention")
        if convention not in (None, "pixel"):
            raise ValueError(f"the playground cannot edit {convention!r} fields")
        metadata = {key: val for key, val in payload.items() if not torch.is_tensor(val)}
        image = coerce_image(payload["image"]) if "image" in payload else None
        kind_hint = payload.get("field_kind", payload.get("kind"))
        for key in (
            "field",
            "velocity",
            "vector_momentum",
            "momentum",
            "u",
            "a",
            "acceleration",
            "residual",
            "residuals",
        ):
            if key in payload:
                value = payload[key]
                kind_hint = kind_hint or (None if key == "field" else key)
                break
        else:
            raise ValueError(f"no supported field key found in {path}")

    field, mode = coerce_field(value)
    default_kind = "velocity" if mode == "vector" else "u"
    kind = _canonical_kind(kind_hint or default_kind, mode, strict=kind_hint is not None)
    return LoadedField(field, kind, image, metadata)


def gaussian_patch_mask(
    size: tuple[int, int], center: tuple[float, float], sigma: float
) -> torch.Tensor:
    y = torch.arange(size[0], dtype=torch.float32)[:, None]
    x = torch.arange(size[1], dtype=torch.float32)[None, :]
    distance_sq = (x - center[0]).square() + (y - center[1]).square()
    return torch.exp(-distance_sq / (2 * max(float(sigma), 0.25) ** 2))


def _sample_polyline(
    points: list[tuple[float, float]], spacing: float
) -> list[tuple[float, float]]:
    if len(points) < 2:
        return points
    spacing = max(float(spacing), 1.0)
    filtered = [points[0]]
    for point in points[1:-1]:
        if np.hypot(point[0] - filtered[-1][0], point[1] - filtered[-1][1]) >= spacing:
            filtered.append(point)
    filtered.append(points[-1])

    sampled = [filtered[0]]
    for start, end in zip(filtered[:-1], filtered[1:]):
        distance = float(np.hypot(end[0] - start[0], end[1] - start[1]))
        count = max(1, int(np.ceil(distance / spacing)))
        sampled.extend(
            (
                start[0] + (end[0] - start[0]) * index / count,
                start[1] + (end[1] - start[1]) * index / count,
            )
            for index in range(1, count + 1)
        )
    return sampled


def _stroke_mask(
    size: tuple[int, int], points: list[tuple[float, float]], sigma: float
) -> torch.Tensor:
    points = _sample_polyline(points, max(1.0, sigma / 2))
    mask = torch.zeros(size)
    if not points:
        return mask
    sigma = max(float(sigma), 0.25)
    radius = max(1, int(np.ceil(3 * sigma)))
    coordinates = torch.arange(-radius, radius + 1, dtype=torch.float32)
    y, x = torch.meshgrid(coordinates, coordinates, indexing="ij")
    kernel = torch.exp(-(x.square() + y.square()) / (2 * sigma**2))
    height, width = size
    for x_center, y_center in points:
        x_center, y_center = round(x_center), round(y_center)
        x0, x1 = max(0, x_center - radius), min(width, x_center + radius + 1)
        y0, y1 = max(0, y_center - radius), min(height, y_center + radius + 1)
        if x0 >= x1 or y0 >= y1:
            continue
        kx0, ky0 = x0 - (x_center - radius), y0 - (y_center - radius)
        patch = kernel[ky0 : ky0 + y1 - y0, kx0 : kx0 + x1 - x0]
        mask[y0:y1, x0:x1] = torch.maximum(mask[y0:y1, x0:x1], patch)
    return mask


def add_vector_arrow(
    field: torch.Tensor,
    start: tuple[float, float],
    end: tuple[float, float],
    sigma: float,
    gain: float = 1.0,
) -> torch.Tensor:
    if field.shape[1] != 2:
        raise ValueError("vector field must have shape [1,2,H,W]")
    displacement = torch.tensor((end[0] - start[0], end[1] - start[1])) * float(gain)
    mask = gaussian_patch_mask(field.shape[-2:], start, sigma)
    return field + displacement[None, :, None, None] * mask[None, None]


def paint_scalar_stroke(
    field: torch.Tensor,
    points: list[tuple[float, float]],
    sigma: float,
    amplitude: float,
) -> torch.Tensor:
    if field.shape[1] != 1:
        raise ValueError("scalar field must have shape [1,1,H,W]")
    return field + float(amplitude) * _stroke_mask(field.shape[-2:], points, sigma)[None, None]


def erase_stroke(
    field: torch.Tensor,
    points: list[tuple[float, float]],
    sigma: float,
) -> torch.Tensor:
    mask = _stroke_mask(field.shape[-2:], points, sigma).clamp(0, 1)
    return field * (1 - mask[None, None])


def _relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    denominator = expected.norm()
    return float(actual.norm()) if denominator == 0 else float((actual - expected).norm() / denominator)


def analyze_field(
    image: torch.Tensor,
    field: torch.Tensor,
    kind: str,
    *,
    alpha: float = 0.2,
    beta: float = 0.2,
    gamma: float = 0.001,
    rho: float = 0.5,
    solve_inverse: bool = True,
    cg_eps: float = 1e-5,
    device: str | torch.device = "cpu",
) -> AnalysisResult:
    mode = mode_for_kind(kind)
    image = image.to(device=device, dtype=torch.float32)
    field = field.to(device=device, dtype=torch.float32)
    operator = SobolevFluidOperator(alpha=alpha, beta=beta, gamma=gamma)

    with torch.no_grad():
        if mode == "vector":
            if field.shape[1] != 2:
                raise ValueError(f"{kind} requires a vector field")
            if kind == "velocity":
                counterpart = operator.apply_operator(field)
                roundtrip = operator.apply_inverse(counterpart)
            else:
                counterpart = operator.apply_inverse(field)
                roundtrip = operator.apply_operator(counterpart)
            return AnalysisResult(
                counterpart.cpu(),
                roundtrip.cpu(),
                None,
                _relative_error(roundtrip, field),
                float((field * counterpart).sum()),
            )

        if field.shape[1] != 1:
            raise ValueError(f"{kind} requires a scalar field")

        from ..cometric_inversion import CometricOperator

        cometric = CometricOperator(
            image, rho, operator, dx_convention="pixel"
        )
        gradient = cometric.image_gradient
        solver_iterations = solver_time = None
        if kind == "u":
            covector = field
            acceleration = cometric(covector)
            if solve_inverse and acceleration.norm() != 0:
                roundtrip = cometric.inverse(acceleration, eps=cg_eps)
            else:
                roundtrip = covector.clone()
            counterpart = acceleration
            expected = covector
        else:
            acceleration = field
            if not solve_inverse:
                raise ValueError("acceleration input requires a cometric inverse solve")
            covector, solver_iterations, solver_time = cometric.inverse(
                acceleration,
                eps=cg_eps,
                return_info=True,
            )
            counterpart = covector
            roundtrip = cometric(covector)
            expected = acceleration

        vector_momentum = (covector.unsqueeze(2) * gradient).sum(dim=1)
        kernel_response = operator.apply_inverse(vector_momentum)
        return AnalysisResult(
            counterpart.cpu(),
            roundtrip.cpu(),
            kernel_response.cpu(),
            _relative_error(roundtrip, expected),
            float((covector * acceleration).sum()),
            solver_iterations,
            solver_time,
        )
