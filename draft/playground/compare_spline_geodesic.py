"""Compare a periodic geodesic with its zero-acceleration, zero-jerk spline.

Pass the ``image_momentum_t000.pt`` field exported by
``draft/export_classic_metamorphosis_fields.py --kernel sobolev``. The embedded
source image and registration parameters are used unless explicitly overridden.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
for path in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from draft.export_classic_metamorphosis_fields import load_image
from draft.playground.field_playground_core import load_field_file
from draft.playground.splines.core import (
    SplineParameters,
    load_setup,
    run_classic,
    run_spline,
    zero_setup,
)


@torch.inference_mode()
def run_classical_geodesic(
    source: torch.Tensor,
    initial_momentum: torch.Tensor,
    parameters: SplineParameters,
    *,
    device: str | torch.device,
) -> dict[str, torch.Tensor]:
    setup = zero_setup(source, source, parameters)
    setup.initial_momentum.copy_(initial_momentum)
    trajectory = run_classic(setup, device=device)
    return {
        "images": trajectory.images,
        "momentum": trajectory.momentum,
        "velocity": trajectory.velocity,
    }


@torch.inference_mode()
def run_geodesic_spline(
    source: torch.Tensor,
    initial_momentum: torch.Tensor,
    parameters: SplineParameters,
    *,
    device: str | torch.device,
):
    setup = zero_setup(source, source, parameters)
    setup.initial_momentum.copy_(initial_momentum)
    return run_spline(setup, device=device)


def _error_metrics(
    reference: torch.Tensor,
    candidate: torch.Tensor,
) -> dict[str, float | int]:
    difference = candidate - reference
    absolute_rms = float(difference.square().mean().sqrt())
    reference_rms = float(reference.square().mean().sqrt())
    per_node_difference = difference.flatten(1).square().mean(1).sqrt()
    per_node_reference = reference.flatten(1).square().mean(1).sqrt()
    per_node_relative = per_node_difference / per_node_reference.clamp_min(1e-12)
    worst_node = int(per_node_relative.argmax())
    endpoint_denominator = max(float(per_node_reference[-1]), 1e-12)
    return {
        "absolute_rms": absolute_rms,
        "relative_rms": absolute_rms / max(reference_rms, 1e-12),
        "endpoint_relative_rms": float(per_node_difference[-1])
        / endpoint_denominator,
        "maximum_absolute": float(difference.abs().max()),
        "worst_node": worst_node,
        "worst_node_relative_rms": float(per_node_relative[worst_node]),
    }


def compare_geodesics(
    source: torch.Tensor,
    initial_momentum: torch.Tensor,
    parameters: SplineParameters,
    *,
    device: str | torch.device = "auto",
) -> dict[str, Any]:
    if source.shape != initial_momentum.shape:
        raise ValueError(
            "source and initial momentum must have the same shape, got "
            f"{source.shape} and {initial_momentum.shape}"
        )
    if parameters.control_steps:
        raise ValueError("geodesic comparison requires no spline control times")

    classical = run_classical_geodesic(
        source,
        initial_momentum,
        parameters,
        device=device,
    )
    spline = run_geodesic_spline(
        source,
        initial_momentum,
        parameters,
        device=device,
    )
    metrics = {
        "image": _error_metrics(classical["images"], spline.images),
        "momentum": _error_metrics(classical["momentum"], spline.momentum),
        "velocity": _error_metrics(classical["velocity"], spline.velocity),
    }
    metrics["geodesic_invariant"] = {
        "maximum_force": float(spline.force.abs().max()),
        "maximum_acceleration": float(spline.acceleration.abs().max()),
        "maximum_jerk": float(spline.jerk.abs().max()),
    }
    return {
        "parameters": parameters.as_dict(),
        "metrics": metrics,
        "classical": classical,
        "spline": spline,
    }


def _parameter_value(
    override: float | int | None,
    metadata: dict[str, Any],
    name: str,
    default: float | int,
):
    return override if override is not None else metadata.get(name, default)


def load_comparison_input(
    momentum_path: str | Path,
    *,
    source_path: str | Path | None,
    rho: float | None,
    alpha: float | None,
    beta: float | None,
    gamma: float | None,
    cg_eps: float | None,
    steps: int | None,
) -> tuple[torch.Tensor, torch.Tensor, SplineParameters]:
    input_path = Path(momentum_path).expanduser()
    if input_path.is_dir():
        input_path = (
            input_path / "spline_setup.pt"
            if (input_path / "spline_setup.pt").is_file()
            else input_path / "manifest.json"
        )
    if input_path.name == "manifest.json" and input_path.is_file():
        manifest = json.loads(input_path.read_text(encoding="utf-8"))
        relative = manifest.get("spline_setup") or manifest.get("initial_momentum")
        if relative is None:
            raise ValueError(
                f"manifest has no spline_setup or initial_momentum: {input_path}"
            )
        input_path = input_path.parent / relative
    if not input_path.is_file():
        raise FileNotFoundError(
            f"comparison input not found: {input_path}. Pass an exported run "
            "directory, manifest.json, spline_setup.pt, or image_momentum_t000.pt."
        )

    if input_path.name == "spline_setup.pt":
        setup = load_setup(input_path)
        source = load_image(source_path) if source_path is not None else setup.source
        momentum = setup.initial_momentum
        saved = setup.parameters.as_dict()
    else:
        loaded = load_field_file(input_path)
        time_index = loaded.metadata.get("time_index")
        if time_index is not None and int(time_index) != 0:
            raise ValueError(
                f"expected an initial momentum at time_index 0, got {time_index}"
            )
        momentum = loaded.field
        if source_path is not None:
            source = load_image(source_path)
        elif loaded.image is not None:
            source = loaded.image
        elif loaded.metadata.get("image_path"):
            source = load_image(loaded.metadata["image_path"])
        else:
            raise ValueError("the momentum has no embedded source; pass --source")
        saved = loaded.metadata.get("parameters", {})

    if momentum.shape[1] != 1:
        raise ValueError(f"initial momentum must be scalar, got {momentum.shape}")
    if source.shape != momentum.shape:
        raise ValueError(
            "source and momentum shapes differ; numerical comparison never resizes "
            f"fields ({source.shape} != {momentum.shape})"
        )

    if saved.get("kernel") not in (None, "sobolev"):
        raise ValueError("spline comparison requires a Sobolev-exported momentum")
    parameters = SplineParameters(
        rho=float(_parameter_value(rho, saved, "rho", 0.5)),
        alpha=float(_parameter_value(alpha, saved, "alpha", 0.2)),
        beta=float(_parameter_value(beta, saved, "beta", 0.2)),
        gamma=float(_parameter_value(gamma, saved, "gamma", 0.001)),
        cg_eps=float(_parameter_value(cg_eps, saved, "cg_eps", 1e-5)),
        n_steps=int(
            steps
            if steps is not None
            else saved.get("integration_steps", saved.get("n_steps", 16))
        ),
        control_steps=(),
    )
    return source, momentum, parameters


def _serializable_report(comparison: dict[str, Any]) -> dict[str, Any]:
    return {
        "parameters": comparison["parameters"],
        "metrics": comparison["metrics"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "momentum",
        help=(
            "Exported run directory, manifest, spline_setup.pt, or scalar "
            "momentum at time_index 0"
        ),
    )
    parser.add_argument("--source", help="Override the embedded source image")
    parser.add_argument("--rho", type=float)
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--beta", type=float)
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--cg-eps", type=float)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--json", type=Path, help="Optional JSON report path")
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    source, momentum, parameters = load_comparison_input(
        args.momentum,
        source_path=args.source,
        rho=args.rho,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        cg_eps=args.cg_eps,
        steps=args.steps,
    )
    comparison = compare_geodesics(
        source,
        momentum,
        parameters,
        device=args.device,
    )
    report = _serializable_report(comparison)
    print(json.dumps(report, indent=2))
    print(
        "With zero acceleration, zero jerk, and no controls, the classical and "
        "spline trajectories use the same source-inside-warp update."
    )
    if args.json is not None:
        args.json.expanduser().write_text(
            json.dumps(report, indent=2) + "\n",
            encoding="utf-8",
        )
    return comparison


if __name__ == "__main__":
    main()
