"""FRACTAL command-line interface for benchmarks, environment diagnostics, and serving."""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List, Optional

import numpy as np

from fractal_quant.codec import (
    ACTIVATION_FRAME_LIMIT_BYTES,
    Int4BlockQuantizer,
    cosine_similarity,
    dequantize_activation_int8,
    quantize_activation_int8,
    quantize_tensor_int4,
)


def _global_int4_baseline(weights: np.ndarray) -> np.ndarray:
    """Single global-scale uniform 4-bit baseline for accuracy comparison."""
    max_val = float(np.max(np.abs(weights)))
    if max_val == 0.0:
        return np.zeros_like(weights)
    scale = max_val / 7.0
    return (np.clip(np.round(weights / scale), -8.0, 7.0) * scale).astype(np.float32)


def run_benchmark(num_layers: int = 22, hidden_size: int = 2048, intermediate_size: int = 5632) -> int:
    """Reproduce FRACTAL micro-block INT4 vs global INT4 quantization across transformer layers."""
    rng = np.random.default_rng(seed=42)
    fractal_cos_list: List[float] = []
    global_cos_list: List[float] = []
    total_fp32 = 0
    total_frq4 = 0

    t0 = time.perf_counter()
    print(f"FRACTAL Micro-Block INT4 Benchmark ({num_layers} layers, {intermediate_size}x{hidden_size})")
    for layer_idx in range(1, num_layers + 1):
        weights = rng.normal(0.0, 0.02, (intermediate_size, hidden_size)).astype(np.float32)
        outlier_prob = 0.0005 + 0.0025 * (layer_idx / num_layers)
        outlier_mask = rng.random(weights.shape) < outlier_prob
        count = int(np.sum(outlier_mask))
        if count > 0:
            outlier_scale = 0.5 + 1.5 * (layer_idx / num_layers)
            weights[outlier_mask] = rng.uniform(0.3, outlier_scale, count) * rng.choice(
                [-1.0, 1.0], count
            )

        global_rec = _global_int4_baseline(weights)
        cos_g = cosine_similarity(weights, global_rec)

        pkt, fractal_rec = quantize_tensor_int4(weights, block_size=64)
        cos_f = cosine_similarity(weights, fractal_rec)

        total_fp32 += weights.nbytes
        total_frq4 += len(pkt)
        fractal_cos_list.append(cos_f)
        global_cos_list.append(cos_g)

    elapsed = time.perf_counter() - t0
    ratio = total_fp32 / total_frq4
    mean_f = float(np.mean(fractal_cos_list))
    mean_g = float(np.mean(global_cos_list))

    # Verify 4096-dim INT8 activation contract
    act = rng.normal(0.0, 1.0, (1, 1, 4096)).astype(np.float32)
    frame = quantize_activation_int8(act)
    _ = dequantize_activation_int8(frame, (1, 1, 4096))

    print(f"  Mean Cosine Sim (Global INT4)  : {mean_g:.4f} (min={min(global_cos_list):.4f})")
    print(f"  Mean Cosine Sim (FRACTAL INT4) : {mean_f:.4f} (min={min(fractal_cos_list):.4f})")
    print(f"  Wire Compression Ratio         : {ratio:.2f}x ({total_fp32 / 1e6:.1f} MB -> {total_frq4 / 1e6:.1f} MB)")
    print(f"  Activation Frame Size          : {len(frame)} / {ACTIVATION_FRAME_LIMIT_BYTES} B limit")
    print(f"  Completed in {elapsed:.2f}s")
    return 0


def run_doctor() -> int:
    """Inspect local runtime capabilities and required dependencies."""
    print(f"Python: {sys.version.split()[0]}")
    for mod in ("numpy", "flask", "torch", "transformers", "tensorflow", "torchao", "executorch"):
        try:
            __import__(mod)
            status = "installed"
        except ImportError:
            status = "not installed"
        print(f"  {mod:<14}: {status}")
    quantizer = Int4BlockQuantizer(block_size=64)
    pkt, stats = quantizer.compress("smoke", np.random.randn(1024).astype(np.float32))
    print(f"  FRQ4 codec    : OK (ratio={stats.compression_ratio:.2f}x, cos={stats.cosine_similarity:.4f}, bytes={len(pkt)})")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point for the fractal CLI."""
    parser = argparse.ArgumentParser(prog="fractal", description="FRACTAL Compute Orchestration CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    bench_p = sub.add_parser("bench", help="Run reproducible INT4 block quantization benchmark")
    bench_p.add_argument("--layers", type=int, default=6, help="Number of transformer blocks to evaluate")
    bench_p.add_argument("--hidden-size", type=int, default=2048)
    bench_p.add_argument("--intermediate-size", type=int, default=5632)

    sub.add_parser("doctor", help="Check installed dependencies and codec self-test")

    serve_p = sub.add_parser("serve", help="Start the local orchestration server")
    serve_p.add_argument("--host", default="127.0.0.1")
    serve_p.add_argument("--port", type=int, default=5000)

    args = parser.parse_args(argv)

    if args.command == "bench":
        return run_benchmark(args.layers, args.hidden_size, args.intermediate_size)
    if args.command == "doctor":
        return run_doctor()
    if args.command == "serve":
        server_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "fractal_server"))
        if server_dir not in sys.path:
            sys.path.insert(0, server_dir)
        from fractal_server import server as srv

        srv._ensure_initialized()
        srv.app.run(host=args.host, port=args.port, debug=False, use_reloader=False)
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
