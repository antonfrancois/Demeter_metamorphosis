"""Benchmark warm CPU/CUDA latency of the draft metric operators.

Run from the repository root with::

    PYTHONPATH=src .venv/bin/python -m draft.benchmark_cometric
"""

from argparse import ArgumentParser
from math import ceil
from statistics import median
from time import perf_counter

import torch

from demeter.utils.cometric_inversion import CometricOperator
from demeter.utils.reproducing_kernels import SobolevFluidOperator


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _time_call(function, device: torch.device) -> float:
    _synchronize(device)
    start = perf_counter()
    function()
    _synchronize(device)
    return perf_counter() - start


def _summary(samples: list[float]) -> tuple[float, float, float, float]:
    ordered = sorted(samples)
    p95 = ordered[ceil(0.95 * len(ordered)) - 1]
    return median(ordered), p95, ordered[0], ordered[-1]


def _inputs(size: int, dtype: torch.dtype) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(2026 + size)
    coordinates = torch.linspace(-1, 1, size, dtype=dtype)
    y, x = torch.meshgrid(coordinates, coordinates, indexing="ij")
    image = (
        torch.exp(-5 * (x.square() + y.square()))
        + 0.2 * torch.sin(4 * x) * torch.cos(3 * y)
    )[None, None]
    scalar = torch.randn((1, 1, size, size), generator=generator, dtype=dtype)
    vector = torch.randn((1, 2, size, size), generator=generator, dtype=dtype)
    return image, scalar, vector


def _benchmark(
    size: int,
    device: torch.device,
    dtype: torch.dtype,
    repeats: int,
    warmup: int,
    tolerance: float,
):
    image, scalar, vector = (value.to(device) for value in _inputs(size, dtype))
    operator = SobolevFluidOperator(alpha=0.2, beta=0.2, gamma=0.001)
    cometric = CometricOperator(image, rho=0.5, kernel_operator=operator)

    functions = {
        "L": lambda: operator.apply_operator(vector),
        "K": lambda: operator.apply_inverse(vector),
        "A": lambda: cometric(scalar),
        "A^-1": lambda: cometric.inverse(
            scalar, eps=tolerance, return_info=True
        ),
    }

    with torch.inference_mode():
        cold_inverse = _time_call(functions["A^-1"], device)
        for _ in range(warmup):
            for function in functions.values():
                function()
        _synchronize(device)

        timings = {
            name: _summary(
                [_time_call(function, device) for _ in range(repeats)]
            )
            for name, function in functions.items()
        }
        _, iterations, _, residual = functions["A^-1"]()
        _synchronize(device)

    return cold_inverse, timings, iterations, residual


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--sizes", nargs="+", type=int, default=[64, 128, 240, 512])
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    args = parser.parse_args()

    if args.repeats < 1 or args.warmup < 0:
        parser.error("repeats must be positive and warmup must be non-negative")
    if args.threads is not None:
        torch.set_num_threads(args.threads)

    dtype = getattr(torch, args.dtype)
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))

    print(f"PyTorch {torch.__version__}; dtype={args.dtype}; CPU threads={torch.get_num_threads()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
    print("Synchronized wall time in ms: median / p95 [min, max]")

    for size in args.sizes:
        results = {}
        for device in devices:
            cold, timings, iterations, residual = _benchmark(
                size,
                device,
                dtype,
                args.repeats,
                args.warmup,
                args.tolerance,
            )
            results[device.type] = timings
            print(
                f"\n{size}x{size} {device.type}: cold A^-1={1e3 * cold:.3f}; "
                f"iterations={iterations}; residual={residual:.3g}"
            )
            for name, values in timings.items():
                med, p95, low, high = (1e3 * value for value in values)
                print(
                    f"  {name:4}: {med:8.3f} / {p95:8.3f} "
                    f"[{low:8.3f}, {high:8.3f}]"
                )

        if "cuda" in results:
            speedups = []
            for name in results["cpu"]:
                cpu_median = results["cpu"][name][0]
                cuda_median = results["cuda"][name][0]
                speedups.append(f"{name}={cpu_median / cuda_median:.2f}x")
            print("  CPU/CUDA warm speedup: " + ", ".join(speedups))


if __name__ == "__main__":
    main()
