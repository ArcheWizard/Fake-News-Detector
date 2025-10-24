"""Benchmark inference latency for baseline and optimized models.

Usage:
  python scripts/benchmark_inference.py \
    --model_dir runs/roberta-kfr/model \
    --iters 50 --warmup 5 --device cpu

Optional comparisons:
  python scripts/benchmark_inference.py \
    --baseline_dir runs/roberta-kfr/model \
    --optimized_dir runs/roberta-kfr/model-quant \
    --iters 50 --warmup 5 --device cpu

Notes:
- Uses the centralized classification pipeline utility for consistency.
- Focuses on latency (mean/median/p95). Throughput can be inferred.
"""

from __future__ import annotations

import argparse
import statistics
import time

import torch

from fnd.models.utils import create_classification_pipeline


def _now() -> float:
    return time.perf_counter()


def _run_benchmark(
    model_dir: str, text: str, iters: int, warmup: int, device_arg: str | None
) -> dict:
    # Select device index for pipeline utility
    device: int | None
    if device_arg == "cpu":
        device = -1
    elif device_arg == "gpu":
        device = 0 if torch.cuda.is_available() else None
    else:
        device = None

    pipe = create_classification_pipeline(
        model_dir=model_dir, max_length=256, device=device, return_all_scores=True
    )

    # Warmup
    for _ in range(warmup):
        _ = pipe(text)

    times: list[float] = []
    for _ in range(iters):
        t0 = _now()
        _ = pipe(text)
        times.append(_now() - t0)

    mean = sum(times) / len(times)
    med = statistics.median(times)
    p95 = statistics.quantiles(times, n=20)[18] if len(times) >= 20 else max(times)

    return {
        "model_dir": model_dir,
        "iters": iters,
        "warmup": warmup,
        "device": device_arg or "auto",
        "mean_ms": round(mean * 1000, 3),
        "median_ms": round(med * 1000, 3),
        "p95_ms": round(p95 * 1000, 3),
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark inference latency")
    parser.add_argument(
        "--model_dir", help="Model directory (used if baseline/optimized not provided)"
    )
    parser.add_argument("--baseline_dir", help="Baseline model directory", default=None)
    parser.add_argument(
        "--optimized_dir",
        help="Optimized model directory (quantized or pruned)",
        default=None,
    )
    parser.add_argument(
        "--iters", type=int, default=50, help="Number of measured iterations"
    )
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
    parser.add_argument(
        "--device",
        choices=["cpu", "gpu", "auto"],
        default="auto",
        help="Force device selection (auto prefers GPU if available)",
    )
    parser.add_argument(
        "--text",
        default=(
            "Breaking: Government announces new policy changes today that could"
            " impact the economy and public services. Experts are divided."
        ),
        help="Text to classify for benchmarking",
    )
    args = parser.parse_args()

    runs = []

    if args.baseline_dir or args.optimized_dir:
        if args.baseline_dir:
            runs.append(("baseline", args.baseline_dir))
        if args.optimized_dir:
            runs.append(("optimized", args.optimized_dir))
    else:
        if not args.model_dir:
            parser.error("--model_dir is required when baseline/optimized not provided")
        runs.append(("single", args.model_dir))

    print(
        f"Inference benchmark (iters={args.iters}, "
        f"warmup={args.warmup}, device={args.device})"
    )
    print("-" * 60)

    results = []
    for label, mdir in runs:
        res = _run_benchmark(mdir, args.text, args.iters, args.warmup, args.device)
        print(
            f"{label:9s} | mean={res['mean_ms']:7.2f} ms | median={res['median_ms']:7.2f} ms | p95={res['p95_ms']:7.2f} ms | {mdir}"
        )
        results.append((label, res))

    # Optional comparison summary
    if len(results) == 2:
        (_, base), (_, opt) = results
        if base and opt:
            speedup = (
                base["mean_ms"] / opt["mean_ms"] if opt["mean_ms"] > 0 else float("inf")
            )
            print("-" * 60)
            print(f"Speedup (baseline/optimized): {speedup:.2f}x (mean latency)")


if __name__ == "__main__":
    main()
