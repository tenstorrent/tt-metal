# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Benchmark script for matmul auto-config calibration.

Sweeps representative shapes through matmul_auto and default ttnn.matmul,
producing a CSV that can be used to calibrate heuristic weights or retrain
the DNN scorer via DNNConfigGenerator.train_from_csv().
"""

from __future__ import annotations

import csv
import logging
import os
import sys
import time

logger = logging.getLogger(__name__)

# Representative shapes: (batch, M, K, N)
BENCHMARK_SHAPES = [
    (1, 32, 32, 32),
    (1, 128, 128, 128),
    (1, 512, 512, 512),
    (1, 1024, 1024, 1024),
    (1, 1024, 4096, 1024),
    (1, 2048, 2048, 2048),
    # Falcon-7B shapes
    (1, 32, 4544, 4672),
    (1, 32, 4544, 18176),
    (1, 128, 4544, 4544),
    (1, 1024, 4544, 4672),
]

WARMUP_RUNS = 5
TIMED_RUNS = 20


def _measure(fn, device, warmup=WARMUP_RUNS, runs=TIMED_RUNS):
    """Measure average latency in microseconds."""
    import ttnn

    for _ in range(warmup):
        fn()
    ttnn.synchronize_device(device)
    start = time.perf_counter()
    for _ in range(runs):
        fn()
    ttnn.synchronize_device(device)
    return (time.perf_counter() - start) / runs * 1e6


def run_benchmark(output_csv: str = "matmul_auto_benchmark.csv"):
    """Run the full benchmark sweep and write results to CSV."""
    import torch
    import ttnn
    from ttnn._experimental.auto_config import matmul_auto
    from ttnn._experimental.auto_config.matmul_auto import MatmulAutoConfig

    device = ttnn.open_device(device_id=0)
    grid = device.compute_with_storage_grid_size()
    gx, gy = grid.x, grid.y

    results = []
    for batch, M, K, N in BENCHMARK_SHAPES:
        a = torch.randn(batch, M, K)
        b = torch.randn(K, N)
        ta = ttnn.from_torch(a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        tb = ttnn.from_torch(b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        t_default = _measure(lambda: ttnn.matmul(ta, tb), device)
        t_auto = _measure(lambda: matmul_auto(ta, tb), device)

        selector = MatmulAutoConfig()
        result = selector.select(ta, tb)
        sel = result.selected_config

        row = {
            "M": M,
            "K": K,
            "N": N,
            "batch_size": batch,
            "grid_x": gx,
            "grid_y": gy,
            "config_family": sel.config_family if sel else "fallback",
            "in0_block_w": sel.params.get("in0_block_w", 0) if sel else 0,
            "per_core_M": sel.params.get("per_core_M", 0) if sel else 0,
            "per_core_N": sel.params.get("per_core_N", 0) if sel else 0,
            "default_us": f"{t_default:.0f}",
            "auto_us": f"{t_auto:.0f}",
            "speedup": f"{t_default / t_auto:.3f}" if t_auto > 0 else "N/A",
        }
        results.append(row)
        print(
            f"  {M:5d}x{K:5d}x{N:5d}  default={t_default:8.0f}µs  auto={t_auto:8.0f}µs  "
            f"speedup={t_default / t_auto:.3f}x  config={row['config_family']}"
        )

    ttnn.close_device(device)

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults written to {output_csv} ({len(results)} shapes)")
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    out = sys.argv[1] if len(sys.argv) > 1 else "matmul_auto_benchmark.csv"
    run_benchmark(out)
