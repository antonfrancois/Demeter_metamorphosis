"""Pure tensor and file operations used by the field playground.

Version: July 16, 2026.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ..cometric_inversion import CometricOperator
from ..sobolevfluid_operator import SobolevFluidOperator


VECTOR_KINDS = ("velocity", "vector_momentum")
SCALAR_KINDS = ("a", "u")
FORMAT_VERSION = 1
FIELD_KEYS = (
    "field",
    "velocity",
    "vector_momentum",
    "momentum",
    "u",
    "a",
    "acceleration",
    "residual",
    "residuals",
)


def _as_tensor(value: Any) -> torch.Tensor:
    tensor = torch.as_tensor(value)
    if tensor.dtype != torch.float64:
        tensor = tensor.float()
    return tensor.detach().cpu()


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
    deformation_velocity: torch.Tensor | None
    relative_roundtrip: float
    squared_norm: float
    deformation_energy_contribution: float | None = None
    operator_time: float | None = None
    solver_residual: float | None = None
    solver_iterations: int | None = None
    solver_time: float | None = None


def mode_for_kind(kind: str) -> str:
    return "vector" if kind in VECTOR_KINDS else "scalar"


def _canonical_kind(value: Any, mode: str) -> str:
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
    return aliases.get(text, text)


def coerce_image(value: Any) -> torch.Tensor:
    image = _as_tensor(value)
    if image.ndim == 2:
        image = image[None, None]
    elif image.ndim == 3:
        image = image[None] if image.shape[0] == 1 else image.movedim(-1, 0)[None]
    elif image.shape[1] != 1:
        image = image.movedim(-1, 1)
    minimum, maximum = image.amin(), image.amax()
    if maximum > 1 or minimum < 0:
        span = maximum - minimum
        image = (image - minimum) / span if span > 0 else torch.zeros_like(image)
    return image.contiguous()


def coerce_field(value: Any) -> tuple[torch.Tensor, str]:
    tensor = _as_tensor(value)
    if tensor.ndim == 2:
        tensor = tensor[None, None]
    elif tensor.ndim == 3:
        tensor = tensor[None] if tensor.shape[0] in (1, 2) else tensor.movedim(-1, 0)[None]
    elif tensor.shape[1] not in (1, 2):
        tensor = tensor.movedim(-1, 1)
    mode = "scalar" if tensor.shape[1] == 1 else "vector"
    return tensor.contiguous(), mode


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


def _metadata_scalar(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode()
    return value


def load_field_file(path: str | Path) -> LoadedField:
    path = Path(path).expanduser().resolve()
    suffix = path.suffix.lower()
    if suffix == ".npy":
        payload: Any = np.load(path, allow_pickle=False)
    elif suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            field_keys = [key for key in FIELD_KEYS if key in archive.files]
            if field_keys:
                payload = {key: archive[key].copy() for key in archive.files}
            else:
                payload = archive[archive.files[0]].copy()
    else:
        payload = torch.load(path, map_location="cpu", weights_only=True)

    metadata: dict[str, Any] = {}
    image = None
    kind_hint = None
    named_kind_hint = None
    value = payload
    if isinstance(payload, dict):
        metadata = {
            key: val
            for key, val in payload.items()
            if key not in FIELD_KEYS and key != "image" and not torch.is_tensor(val)
        }
        image = coerce_image(payload["image"]) if "image" in payload else None
        kind_hint = _metadata_scalar(payload.get("field_kind", payload.get("kind")))
        key = next(key for key in FIELD_KEYS if key in payload)
        value = payload[key]
        named_kind_hint = None if key == "field" else key

    field, mode = coerce_field(value)
    default_kind = "velocity" if mode == "vector" else "u"
    kind = _canonical_kind(kind_hint or named_kind_hint or default_kind, mode)
    return LoadedField(field, kind, image, metadata)


def gaussian_patch_mask(
    size: tuple[int, int],
    center: tuple[float, float],
    sigma: float,
    *,
    device=None,
    dtype=torch.float32,
) -> torch.Tensor:
    y = torch.arange(size[0], device=device, dtype=dtype)[:, None]
    x = torch.arange(size[1], device=device, dtype=dtype)[None, :]
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
    field: torch.Tensor, points: list[tuple[float, float]], sigma: float
) -> torch.Tensor:
    points = _sample_polyline(points, max(1.0, sigma / 2))
    size = field.shape[-2:]
    mask = field.new_zeros(size)
    if not points:
        return mask
    sigma = max(float(sigma), 0.25)
    radius = max(1, int(np.ceil(3 * sigma)))
    coordinates = torch.arange(
        -radius, radius + 1, device=field.device, dtype=field.dtype
    )
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
    displacement = field.new_tensor((end[0] - start[0], end[1] - start[1])) * float(gain)
    mask = gaussian_patch_mask(
        field.shape[-2:], start, sigma, device=field.device, dtype=field.dtype
    )
    return field + displacement[None, :, None, None] * mask[None, None]


def paint_scalar_stroke(
    field: torch.Tensor,
    points: list[tuple[float, float]],
    sigma: float,
    amplitude: float,
) -> torch.Tensor:
    return field + float(amplitude) * _stroke_mask(field, points, sigma)[None, None]


def erase_stroke(
    field: torch.Tensor,
    points: list[tuple[float, float]],
    sigma: float,
) -> torch.Tensor:
    mask = _stroke_mask(field, points, sigma)
    return field * (1 - mask[None, None])


def _relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    error_rms = (actual - expected).square().mean().sqrt()
    reference_rms = expected.square().mean().sqrt()
    if reference_rms == 0:
        return 0.0 if error_rms == 0 else float("inf")
    return float(error_rms / reference_rms)


def _timed_call(function, argument: torch.Tensor):
    if argument.is_cuda:
        with torch.cuda.device(argument.device):
            stream = torch.cuda.current_stream()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record(stream)
            result = function(argument)
            end.record(stream)
        end.synchronize()
        return result, start.elapsed_time(end) / 1000

    start = perf_counter()
    result = function(argument)
    return result, perf_counter() - start


def analyze_field(
    image: torch.Tensor,
    field: torch.Tensor,
    kind: str,
    *,
    alpha: float = 0.2,
    beta: float = 0.2,
    gamma: float = 0.001,
    rho: float = 0.5,
    cg_eps: float = 1e-5,
    device: str | torch.device = "cpu",
) -> AnalysisResult:
    mode = mode_for_kind(kind)
    dtype = torch.float64 if field.dtype == torch.float64 else torch.float32
    field = field.to(device=device, dtype=dtype)
    operator = SobolevFluidOperator(alpha=alpha, beta=beta, gamma=gamma)

    with torch.no_grad():
        if mode == "vector":
            if kind == "velocity":
                counterpart, operator_time = _timed_call(
                    operator.apply_operator, field
                )
                roundtrip = operator.apply_inverse(counterpart)
            else:
                counterpart, operator_time = _timed_call(
                    operator.apply_inverse, field
                )
                roundtrip = operator.apply_operator(counterpart)
            relative_error = _relative_error(roundtrip, field)
            squared_norm = float((field * counterpart).sum())
            return AnalysisResult(
                counterpart.cpu(),
                roundtrip.cpu(),
                None,
                relative_error,
                squared_norm,
                operator_time=operator_time,
            )

        image = image.to(device=device, dtype=dtype)

        cometric = CometricOperator(
            image, rho, operator, dx_convention="pixel"
        )
        operator_time = solver_residual = solver_iterations = solver_time = None
        if kind == "u":
            covector = field
            acceleration, operator_time = _timed_call(cometric, covector)
            roundtrip = cometric.inverse(acceleration, eps=cg_eps)
            counterpart = acceleration
            expected = covector
        else:
            acceleration = field
            (
                covector,
                solver_iterations,
                solver_time,
                solver_residual,
            ) = cometric.inverse(acceleration, eps=cg_eps, return_info=True)
            counterpart = covector
            roundtrip = cometric(covector)
            expected = acceleration

        vector_momentum = covector * cometric.image_gradient[:, 0]
        kernel_response = operator.apply_inverse(vector_momentum)
        deformation_velocity = -(cometric.rho**0.5) * kernel_response
        relative_error = _relative_error(roundtrip, expected)
        squared_norm = float((covector * acceleration).sum())
        deformation_energy_contribution = float(
            cometric.rho * (vector_momentum * kernel_response).sum()
        )
        return AnalysisResult(
            counterpart.cpu(),
            roundtrip.cpu(),
            deformation_velocity.cpu(),
            relative_error,
            squared_norm,
            deformation_energy_contribution=deformation_energy_contribution,
            operator_time=operator_time,
            solver_residual=solver_residual,
            solver_iterations=solver_iterations,
            solver_time=solver_time,
        )
