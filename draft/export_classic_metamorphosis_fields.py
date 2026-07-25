"""Register two images with periodic classical metamorphosis and export fields.

The registration kernel can be a circular Gaussian RKHS or the matched
periodic Sobolev fluid inverse ``K=L^-1`` used by the spline lab. Per-frame
fields are loadable by the field playground; Sobolev runs with ``rho < 1`` also
emit a complete zero-force, zero-jerk spline-lab setup containing the optimized
momentum.
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
from demeter.utils.reproducing_kernels import GaussianRKHS, SobolevFluidOperator
from draft.playground.splines.core import (
    SplineParameters,
    save_setup,
    zero_setup,
)


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


def build_kernel_operator(
    kind: str,
    *,
    alpha: float = 0.2,
    beta: float = 0.2,
    gamma: float = 0.001,
    sigma: tuple[float, float] = (3.0, 3.0),
    kernel_reach: int = 3,
):
    """Build a periodic classical metamorphosis kernel operator."""
    if kind == "sobolev":
        return SobolevFluidOperator(
            alpha,
            beta,
            gamma,
            boundary="periodic",
        )
    if kind == "gaussian":
        return GaussianRKHS(
            tuple(float(value) for value in sigma),
            border_type="circular",
            normalized=True,
            kernel_reach=kernel_reach,
        )
    raise ValueError("kernel kind must be 'gaussian' or 'sobolev'")


def run_registration(
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    rho: float,
    operator: Any,
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
        boundary="periodic",
    )


def _image_momentum(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    momentum = getattr(value, "momentum_I", None)
    if isinstance(momentum, torch.Tensor):
        return momentum
    raise TypeError(f"expected a tensor or image Momenta, got {type(value)}")


def extract_trajectory(
    registration: Any,
    operator: Any,
    rho: float,
) -> dict[str, torch.Tensor]:
    """Return states at the uniform nodes ``0, 1/N, ..., 1``.

    Demeter stores images and scalar image momenta after every integration
    step. Its optimized momentum supplies the missing state at ``t=0``.
    Deformation velocity is recomputed at every node from the classical
    Hamiltonian relation ``v = -sqrt(rho) K(p grad(I))``. Scalar image
    velocity is evaluated with the prototype cometric as ``A_I p``.
    """
    optimized = getattr(registration, "optimized_momenta", None)
    if optimized is None:
        optimized = registration.to_analyse[0]
    initial_momentum = _image_momentum(optimized).detach().cpu()
    images = torch.cat(
        (registration.source.detach().cpu(), registration.mp.image_stock.detach().cpu()),
        dim=0,
    )
    momentum_stock = registration.mp.momentum_stock
    if isinstance(momentum_stock, torch.Tensor):
        momentum_nodes = momentum_stock.detach().cpu()
    else:
        momentum_nodes = torch.cat(
            [_image_momentum(momentum).detach().cpu() for momentum in momentum_stock],
            dim=0,
        )
    image_momenta = torch.cat((initial_momentum, momentum_nodes), dim=0)
    times = torch.linspace(0, 1, registration.mp.n_step + 1)

    with torch.no_grad():
        cometric = CometricOperator(
            images,
            rho,
            operator,
            dx_convention="pixel",
            gradient_boundary="periodic",
        )
        image_gradients = cometric.image_gradient
        vector_momenta = -math.sqrt(rho) * (
            image_gradients * image_momenta.unsqueeze(2)
        ).sum(dim=1)
        velocities = operator(vector_momenta)
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
    spline_parameters: SplineParameters | None = None,
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
        "initial_momentum": frame_records[0]["image_momentum"],
        "frames": frame_records,
    }
    if spline_parameters is not None:
        setup = zero_setup(
            trajectory["images"][0:1],
            target_image,
            spline_parameters,
            source_path=str(source_path),
            target_path=str(target_path),
        )
        setup.initial_momentum.copy_(trajectory["image_momenta"][0:1])
        setup_path = save_setup(setup, output_dir / "spline_setup.pt")
        manifest["spline_setup"] = setup_path.name
    else:
        (output_dir / "spline_setup.pt").unlink(missing_ok=True)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return output_dir


def _default_run_name(
    source: Path,
    target: Path,
    rho: float,
    kernel: str,
) -> str:
    text = f"{source.stem}_to_{target.stem}_{kernel}_rho_{rho:g}"
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
    parser.add_argument(
        "--kernel",
        choices=("gaussian", "sobolev"),
        default="sobolev",
        help="Periodic deformation kernel (default: sobolev)",
    )
    parser.add_argument("--sigma", nargs=2, type=float, default=(3.0, 3.0))
    parser.add_argument("--kernel-reach", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.2, help="Sobolev L coefficient")
    parser.add_argument("--beta", type=float, default=0.2, help="Sobolev L coefficient")
    parser.add_argument("--gamma", type=float, default=0.001, help="Sobolev L coefficient")
    parser.add_argument("--cg-eps", type=float, default=1e-5)
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
    if args.kernel_reach < 1:
        raise ValueError("kernel_reach must be at least 1")

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
    operator = build_kernel_operator(
        args.kernel,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        sigma=tuple(args.sigma),
        kernel_reach=args.kernel_reach,
    )
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
        "kernel": args.kernel,
        "boundary": "periodic",
        "rho": args.rho,
        "integration_steps": args.integration_steps,
        "iterations": args.iterations,
        "cost_cst": args.cost_cst,
        "grad_coef": args.grad_coef,
        "device": str(device),
    }
    if args.kernel == "sobolev":
        parameters.update(
            alpha=args.alpha,
            beta=args.beta,
            gamma=args.gamma,
            cg_eps=args.cg_eps,
        )
        spline_parameters = (
            SplineParameters(
                alpha=args.alpha,
                beta=args.beta,
                gamma=args.gamma,
                rho=args.rho,
                cg_eps=args.cg_eps,
                n_steps=args.integration_steps,
                control_steps=(),
            )
            if args.rho < 1
            else None
        )
    else:
        parameters.update(
            sigma=tuple(args.sigma),
            kernel_reach=args.kernel_reach,
        )
        isotropic_sigma = math.isclose(args.sigma[0], args.sigma[1])
        if args.rho < 1 and isotropic_sigma and args.kernel_reach == 3:
            spline_parameters = SplineParameters(
                alpha=args.alpha,
                beta=args.beta,
                gamma=args.gamma,
                rho=args.rho,
                cg_eps=args.cg_eps,
                n_steps=args.integration_steps,
                control_steps=(),
                kernel="gaussian",
                sigma=args.sigma[0],
            )
        else:
            spline_parameters = None
            if args.rho < 1:
                print(
                    "Not saving spline_setup.pt: the spline lab only represents "
                    "isotropic Gaussian kernels with kernel reach 3"
                )
    run_name = args.name or _default_run_name(
        source_path,
        target_path,
        args.rho,
        args.kernel,
    )
    output_dir = save_trajectory(
        trajectory,
        args.output_dir.expanduser() / run_name,
        source_path=source_path,
        target_path=target_path,
        target_image=target,
        parameters=parameters,
        spline_parameters=spline_parameters,
    )
    print(f"Saved {len(trajectory['times'])} time nodes to {output_dir}")
    return output_dir


if __name__ == "__main__":
    main()
