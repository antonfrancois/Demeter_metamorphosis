"""Register two images with classical metamorphosis and export its fields.

The per-time-step files are directly loadable by the field playground. The
exported vector momentum and velocity form a matched pair for the configured
Sobolev fluid operator, and the scalar image momentum is paired with the image
velocity produced by the metamorphosis cometric.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import sys
from typing import Any

import matplotlib.image as mpimg
import numpy as np
import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import demeter.metamorphosis as mt

from demeter.utils.cometric_inversion import CometricOperator
from demeter.utils.reproducing_kernels import SobolevFluidOperator


IMAGE_BANK = PROJECT_ROOT / "examples" / "im2Dbank"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "draft" / "playground" / "registration_fields"
FORMAT_VERSION = 1


def resolve_image_path(value: str | Path) -> Path:
    """Resolve a path, an image-bank filename, or a Demeter shorthand."""
    candidate = Path(value).expanduser()
    options = (
        candidate,
        IMAGE_BANK / candidate,
        IMAGE_BANK / f"reg_test_{value}.png",
    )
    for path in options:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(
        f"could not find image {value!r}; pass a path, an im2Dbank filename, "
        "or a shorthand such as '01'"
    )


def load_image(path: str | Path, size: tuple[int, int] | None = None) -> torch.Tensor:
    """Load a 2D image as a normalized ``[1, 1, H, W]`` float tensor."""
    array = np.asarray(mpimg.imread(path))
    if array.ndim == 3:
        if array.shape[-1] >= 3:
            array = np.dot(array[..., :3], [0.2989, 0.5870, 0.1140])
        else:
            array = array[..., 0]
    if array.ndim != 2:
        raise ValueError(f"expected a 2D image at {path}, got shape {array.shape}")

    image = torch.as_tensor(np.asarray(array).copy(), dtype=torch.float32)[None, None]
    if not torch.isfinite(image).all():
        raise ValueError(f"image contains non-finite values: {path}")
    minimum, maximum = image.amin(), image.amax()
    if minimum < 0 or maximum > 1:
        span = maximum - minimum
        image = (image - minimum) / span if span > 0 else torch.zeros_like(image)
    if size is not None and tuple(image.shape[-2:]) != tuple(size):
        image = F.interpolate(image, size=size, mode="bilinear", align_corners=False)
    return image.contiguous()


def resize_target_to_source(
    source: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Resize only the target's spatial dimensions to match the source."""
    if source.shape[:2] != target.shape[:2]:
        raise ValueError(
            "source and target batch/channel dimensions differ "
            f"({source.shape[:2]} != {target.shape[:2]})"
        )
    if source.shape[-2:] == target.shape[-2:]:
        return target
    return F.interpolate(
        target,
        size=source.shape[-2:],
        mode="bilinear",
        align_corners=False,
    ).contiguous()


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("a CUDA device was requested, but CUDA is unavailable")
    return device


def run_registration(
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    rho: float,
    operator: SobolevFluidOperator,
    integration_steps: int,
    iterations: int,
    cost_cst: float,
    grad_coef: float,
    device: torch.device,
) -> Any:
    """Run Demeter's classical balanced metamorphosis registration."""
    source = source.to(device)
    target = target.to(device)
    initial_momentum = torch.zeros_like(source, requires_grad=True)
    return mt.metamorphosis(
        source,
        target,
        initial_momentum,
        rho=rho,
        cost_cst=cost_cst,
        integration_steps=integration_steps,
        n_iter=iterations,
        grad_coef=grad_coef,
        kernelOperator=operator,
        safe_mode=False,
        integration_method="semiLagrangian",
        optimizer_method="LBFGS_torch",
        dx_convention="pixel",
        hamiltonian_integration=False,
        save_gpu_memory=False,
    )


def extract_trajectory(
    registration: Any,
    operator: SobolevFluidOperator,
    rho: float,
) -> dict[str, torch.Tensor]:
    """Return states at the uniform nodes ``0, 1/N, ..., 1``.

    Demeter stores images and scalar image momenta after every integration
    step. Its optimized momentum supplies the missing state at ``t=0``.
    Deformation velocity is recomputed at every node from the classical
    Hamiltonian relation ``v = -sqrt(rho) K(p grad(I))``. Scalar image
    velocity is evaluated with the prototype cometric as ``A_I p``.
    """
    initial_momentum = registration.to_analyse[0].detach().cpu()
    images = torch.cat(
        (registration.source.detach().cpu(), registration.mp.image_stock.detach().cpu()),
        dim=0,
    )
    image_momenta = torch.cat(
        (initial_momentum, registration.mp.momentum_stock.detach().cpu()),
        dim=0,
    )
    times = torch.linspace(0, 1, registration.mp.n_step + 1)

    with torch.no_grad():
        cometric = CometricOperator(
            images, rho, operator, dx_convention="pixel"
        )
        image_gradients = cometric.image_gradient
        vector_momenta = -math.sqrt(rho) * (
            image_gradients * image_momenta.unsqueeze(2)
        ).sum(dim=1)
        velocities = operator.apply_inverse(vector_momenta)
        image_velocities = cometric(image_momenta)

    return {
        "times": times,
        "images": images,
        "image_momenta": image_momenta,
        "image_velocities": image_velocities,
        "vector_momenta": vector_momenta,
        "velocities": velocities,
    }


def _field_payload(
    field: torch.Tensor,
    field_kind: str,
    image: torch.Tensor,
    *,
    time: float,
    time_index: int,
    source_path: Path,
    target_path: Path,
    parameters: dict[str, Any],
    field_role: str | None = None,
) -> dict[str, Any]:
    payload = {
        "format_version": FORMAT_VERSION,
        "field": field.detach().cpu(),
        "field_kind": field_kind,
        "image": image.detach().cpu(),
        "image_path": str(source_path),
        "target_path": str(target_path),
        "time": time,
        "time_index": time_index,
        "dx_convention": "pixel",
        "parameters": parameters,
    }
    if field_role is not None:
        payload["field_role"] = field_role
    return payload


def _save_image(image: torch.Tensor, path: Path) -> None:
    if image.ndim != 4 or image.shape[:2] != (1, 1):
        raise ValueError(f"expected image shape [1, 1, H, W], got {image.shape}")
    mpimg.imsave(
        path,
        image.detach().cpu()[0, 0].numpy(),
        cmap="gray",
        vmin=0,
        vmax=1,
    )


def save_trajectory(
    trajectory: dict[str, torch.Tensor],
    output_dir: Path,
    *,
    source_path: Path,
    target_path: Path,
    target_image: torch.Tensor,
    parameters: dict[str, Any],
) -> Path:
    """Save paired playground files and one complete trajectory archive."""
    output_dir.mkdir(parents=True, exist_ok=True)
    field_directories = {
        "velocity": Path("vector") / "velocity",
        "vector_momentum": Path("vector") / "momentum",
        "image_velocity": Path("scalar") / "velocity",
        "image_momentum": Path("scalar") / "momentum",
    }
    for directory in field_directories.values():
        (output_dir / directory).mkdir(parents=True, exist_ok=True)
    image_paths = {
        "source": Path("images/source.png"),
        "target": Path("images/target.png"),
        "final": Path("images/final.png"),
    }
    (output_dir / "images").mkdir(parents=True, exist_ok=True)
    _save_image(trajectory["images"][0:1], output_dir / image_paths["source"])
    _save_image(target_image, output_dir / image_paths["target"])
    _save_image(trajectory["images"][-1:], output_dir / image_paths["final"])

    frame_records = []
    frame_count = len(trajectory["times"])
    digits = max(3, len(str(frame_count - 1)))

    for index, time_tensor in enumerate(trajectory["times"]):
        time = float(time_tensor)
        suffix = f"t{index:0{digits}d}"
        velocity_name = f"velocity_{suffix}.pt"
        momentum_name = f"momentum_{suffix}.pt"
        image_momentum_name = f"image_momentum_{suffix}.pt"
        image_velocity_name = f"image_velocity_{suffix}.pt"
        velocity_path = field_directories["velocity"] / velocity_name
        momentum_path = field_directories["vector_momentum"] / momentum_name
        image_momentum_path = (
            field_directories["image_momentum"] / image_momentum_name
        )
        image_velocity_path = (
            field_directories["image_velocity"] / image_velocity_name
        )
        image = trajectory["images"][index : index + 1]

        torch.save(
            _field_payload(
                trajectory["velocities"][index : index + 1],
                "velocity",
                image,
                time=time,
                time_index=index,
                source_path=source_path,
                target_path=target_path,
                parameters=parameters,
            ),
            output_dir / velocity_path,
        )
        torch.save(
            _field_payload(
                trajectory["vector_momenta"][index : index + 1],
                "vector_momentum",
                image,
                time=time,
                time_index=index,
                source_path=source_path,
                target_path=target_path,
                parameters=parameters,
            ),
            output_dir / momentum_path,
        )
        torch.save(
            _field_payload(
                trajectory["image_momenta"][index : index + 1],
                "u",
                image,
                time=time,
                time_index=index,
                source_path=source_path,
                target_path=target_path,
                parameters=parameters,
                field_role="image_momentum",
            ),
            output_dir / image_momentum_path,
        )
        torch.save(
            _field_payload(
                trajectory["image_velocities"][index : index + 1],
                "a",
                image,
                time=time,
                time_index=index,
                source_path=source_path,
                target_path=target_path,
                parameters=parameters,
                field_role="image_velocity",
            ),
            output_dir / image_velocity_path,
        )
        frame_records.append(
            {
                "index": index,
                "time": time,
                "velocity": velocity_path.as_posix(),
                "vector_momentum": momentum_path.as_posix(),
                "image_momentum": image_momentum_path.as_posix(),
                "image_velocity": image_velocity_path.as_posix(),
            }
        )

    archive_path = output_dir / "trajectory.pt"
    torch.save(
        {
            "format_version": FORMAT_VERSION,
            **trajectory,
            "source_path": str(source_path),
            "target_path": str(target_path),
            "dx_convention": "pixel",
            "parameters": parameters,
        },
        archive_path,
    )
    manifest = {
        "format_version": FORMAT_VERSION,
        "source_path": str(source_path),
        "target_path": str(target_path),
        "trajectory": archive_path.name,
        "images": {name: path.as_posix() for name, path in image_paths.items()},
        "parameters": parameters,
        "frames": frame_records,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return output_dir


def _default_run_name(source: Path, target: Path, rho: float) -> str:
    text = f"{source.stem}_to_{target.stem}_rho_{rho:g}"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="Source image path, im2Dbank filename, or shorthand")
    parser.add_argument("target", help="Target image path, im2Dbank filename, or shorthand")
    parser.add_argument("--rho", type=float, default=0.5, help="Metamorphosis balance in [0, 1]")
    parser.add_argument(
        "--size",
        nargs=2,
        type=int,
        metavar=("H", "W"),
        help="Resize both images; by default only the target is resized to the source",
    )
    parser.add_argument("--integration-steps", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--cost-cst", type=float, default=0.001)
    parser.add_argument("--grad-coef", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--beta", type=float, default=0.2)
    parser.add_argument("--gamma", type=float, default=0.001)
    parser.add_argument("--device", default="auto", help="Torch device, or 'auto'")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Root directory for exported registration runs",
    )
    parser.add_argument("--name", help="Output subdirectory name")
    return parser


def main(argv: list[str] | None = None) -> Path:
    args = build_parser().parse_args(argv)
    if not 0 <= args.rho <= 1:
        raise ValueError("rho must be in [0, 1]")
    if args.integration_steps < 1:
        raise ValueError("integration_steps must be at least 1")
    if args.iterations < 1:
        raise ValueError("iterations must be at least 1")

    source_path = resolve_image_path(args.source)
    target_path = resolve_image_path(args.target)
    size = tuple(args.size) if args.size else None
    source = load_image(source_path, size)
    target = load_image(target_path, size)
    if source.shape != target.shape:
        print(
            f"Resizing target from {tuple(target.shape[-2:])} "
            f"to source size {tuple(source.shape[-2:])}"
        )
        target = resize_target_to_source(source, target)

    device = resolve_device(args.device)
    operator = SobolevFluidOperator(args.alpha, args.beta, args.gamma)
    registration = run_registration(
        source,
        target,
        rho=args.rho,
        operator=operator,
        integration_steps=args.integration_steps,
        iterations=args.iterations,
        cost_cst=args.cost_cst,
        grad_coef=args.grad_coef,
        device=device,
    )
    trajectory = extract_trajectory(registration, operator, args.rho)
    parameters = {
        "rho": args.rho,
        "alpha": args.alpha,
        "beta": args.beta,
        "gamma": args.gamma,
        "integration_steps": args.integration_steps,
        "iterations": args.iterations,
        "cost_cst": args.cost_cst,
        "grad_coef": args.grad_coef,
        "device": str(device),
    }
    run_name = args.name or _default_run_name(source_path, target_path, args.rho)
    output_dir = save_trajectory(
        trajectory,
        args.output_dir.expanduser() / run_name,
        source_path=source_path,
        target_path=target_path,
        target_image=target,
        parameters=parameters,
    )
    print(f"Saved {len(trajectory['times'])} time nodes to {output_dir}")
    return output_dir


if __name__ == "__main__":
    main()
