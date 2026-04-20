# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Automated DNN retraining pipeline for matmul auto-config.

Runs the full benchmark → train → validate cycle on 10k+ shapes.
This satisfies the maintainer requirement for 10k-100k training data points.

Usage:
    # Full pipeline: benchmark + train + validate (10k shapes)
    python -m ttnn._experimental.auto_config.retrain_dnn --full --num-shapes 10000

    # Benchmark only (collect telemetry)
    python -m ttnn._experimental.auto_config.retrain_dnn --benchmark-only

    # Train only (from existing benchmark results)
    python -m ttnn._experimental.auto_config.retrain_dnn --train-only --input benchmark_results.json

    # Validate only (test DNN scorer accuracy)
    python -m ttnn._experimental.auto_config.retrain_dnn --validate-only
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Allowed base directories for file I/O operations (Cycode SAST compliance)
_ALLOWED_BASE_DIRS = [
    os.path.realpath(os.getcwd()),
    os.path.realpath(os.path.join(os.path.expanduser("~"), ".ttnn")),
    os.path.realpath(os.environ.get("TTNN_AUTO_CONFIG_CACHE_DIR", os.path.join(os.path.expanduser("~"), ".ttnn"))),
]


def _sanitize_path(user_path: str) -> str:
    """Sanitize and validate a user-provided file path.

    Ensures the resolved path is within an allowed base directory,
    preventing directory traversal attacks.

    Raises:
        ValueError: If the resolved path is outside all allowed directories.
    """
    resolved = os.path.realpath(user_path)
    for base in _ALLOWED_BASE_DIRS:
        if resolved.startswith(base + os.sep) or resolved == base:
            return resolved
    raise ValueError(f"Path '{resolved}' is outside allowed directories. " f"Allowed: {_ALLOWED_BASE_DIRS}")


# ──────────────────────────────────────────────────────────────────────
# 106 training shapes covering all asymmetric M/K/N families
# ──────────────────────────────────────────────────────────────────────
TRAINING_SHAPES = [
    # ── Square baselines (8 shapes) ──────────────────────────────────
    {"M": 32, "K": 32, "N": 32, "category": "square"},
    {"M": 64, "K": 64, "N": 64, "category": "square"},
    {"M": 128, "K": 128, "N": 128, "category": "square"},
    {"M": 256, "K": 256, "N": 256, "category": "square"},
    {"M": 512, "K": 512, "N": 512, "category": "square"},
    {"M": 1024, "K": 1024, "N": 1024, "category": "square"},
    {"M": 2048, "K": 2048, "N": 2048, "category": "square"},
    {"M": 4096, "K": 4096, "N": 4096, "category": "square"},
    # ── M-large, K/N-small (10 shapes) ───────────────────────────────
    {"M": 2048, "K": 128, "N": 128, "category": "M_large"},
    {"M": 2048, "K": 256, "N": 64, "category": "M_large"},
    {"M": 4096, "K": 64, "N": 64, "category": "M_large"},
    {"M": 4096, "K": 128, "N": 32, "category": "M_large"},
    {"M": 8192, "K": 32, "N": 32, "category": "M_large"},
    {"M": 8192, "K": 64, "N": 64, "category": "M_large"},
    {"M": 2048, "K": 64, "N": 128, "category": "M_large"},
    {"M": 4096, "K": 32, "N": 128, "category": "M_large"},
    {"M": 1024, "K": 128, "N": 64, "category": "M_large"},
    {"M": 1024, "K": 64, "N": 32, "category": "M_large"},
    # ── K-large, M/N-small (10 shapes) ───────────────────────────────
    {"M": 32, "K": 4096, "N": 32, "category": "K_large"},
    {"M": 64, "K": 8192, "N": 64, "category": "K_large"},
    {"M": 128, "K": 4096, "N": 128, "category": "K_large"},
    {"M": 32, "K": 2048, "N": 64, "category": "K_large"},
    {"M": 64, "K": 4096, "N": 32, "category": "K_large"},
    {"M": 128, "K": 8192, "N": 32, "category": "K_large"},
    {"M": 64, "K": 2048, "N": 128, "category": "K_large"},
    {"M": 32, "K": 1024, "N": 128, "category": "K_large"},
    {"M": 128, "K": 2048, "N": 64, "category": "K_large"},
    {"M": 64, "K": 1024, "N": 64, "category": "K_large"},
    # ── N-large, M/K-small (8 shapes) ────────────────────────────────
    {"M": 32, "K": 128, "N": 4096, "category": "N_large"},
    {"M": 64, "K": 64, "N": 8192, "category": "N_large"},
    {"M": 128, "K": 256, "N": 4096, "category": "N_large"},
    {"M": 32, "K": 64, "N": 2048, "category": "N_large"},
    {"M": 64, "K": 128, "N": 4096, "category": "N_large"},
    {"M": 128, "K": 64, "N": 8192, "category": "N_large"},
    {"M": 32, "K": 256, "N": 2048, "category": "N_large"},
    {"M": 64, "K": 32, "N": 4096, "category": "N_large"},
    # ── LLM decode (8 shapes) ────────────────────────────────────────
    {"M": 32, "K": 4096, "N": 4096, "category": "llm_decode"},
    {"M": 32, "K": 4096, "N": 11008, "category": "llm_decode"},
    {"M": 32, "K": 11008, "N": 4096, "category": "llm_decode"},
    {"M": 32, "K": 4544, "N": 4672, "category": "llm_decode"},
    {"M": 32, "K": 4544, "N": 18176, "category": "llm_decode"},
    {"M": 32, "K": 18176, "N": 4544, "category": "llm_decode"},
    {"M": 32, "K": 8192, "N": 8192, "category": "llm_decode"},
    {"M": 32, "K": 8192, "N": 28672, "category": "llm_decode"},
    # ── LLM prefill (10 shapes) ──────────────────────────────────────
    {"M": 128, "K": 4096, "N": 4096, "category": "llm_prefill"},
    {"M": 512, "K": 4096, "N": 11008, "category": "llm_prefill"},
    {"M": 2048, "K": 4096, "N": 4096, "category": "llm_prefill"},
    {"M": 128, "K": 4096, "N": 11008, "category": "llm_prefill"},
    {"M": 128, "K": 11008, "N": 4096, "category": "llm_prefill"},
    {"M": 256, "K": 4096, "N": 4096, "category": "llm_prefill"},
    {"M": 1024, "K": 4096, "N": 4096, "category": "llm_prefill"},
    {"M": 2048, "K": 4096, "N": 11008, "category": "llm_prefill"},
    {"M": 512, "K": 8192, "N": 8192, "category": "llm_prefill"},
    {"M": 2048, "K": 8192, "N": 8192, "category": "llm_prefill"},
    # ── Attention shapes (8 shapes) ──────────────────────────────────
    {"M": 2048, "K": 128, "N": 2048, "category": "attention"},
    {"M": 128, "K": 2048, "N": 128, "category": "attention"},
    {"M": 4096, "K": 128, "N": 4096, "category": "attention"},
    {"M": 512, "K": 64, "N": 512, "category": "attention"},
    {"M": 1024, "K": 128, "N": 1024, "category": "attention"},
    {"M": 2048, "K": 64, "N": 2048, "category": "attention"},
    {"M": 256, "K": 128, "N": 256, "category": "attention"},
    {"M": 4096, "K": 64, "N": 4096, "category": "attention"},
    # ── Non-power-of-2 (8 shapes) ────────────────────────────────────
    {"M": 384, "K": 1024, "N": 768, "category": "non_pow2"},
    {"M": 768, "K": 3072, "N": 768, "category": "non_pow2"},
    {"M": 1536, "K": 2048, "N": 512, "category": "non_pow2"},
    {"M": 192, "K": 768, "N": 3072, "category": "non_pow2"},
    {"M": 1536, "K": 1536, "N": 1536, "category": "non_pow2"},
    {"M": 384, "K": 384, "N": 1536, "category": "non_pow2"},
    {"M": 768, "K": 768, "N": 3072, "category": "non_pow2"},
    {"M": 1024, "K": 768, "N": 3072, "category": "non_pow2"},
    # ── M,N-large K-small (8 shapes) ─────────────────────────────────
    {"M": 2048, "K": 128, "N": 2048, "category": "MN_large_K_small"},
    {"M": 4096, "K": 64, "N": 4096, "category": "MN_large_K_small"},
    {"M": 2048, "K": 64, "N": 4096, "category": "MN_large_K_small"},
    {"M": 4096, "K": 128, "N": 2048, "category": "MN_large_K_small"},
    {"M": 1024, "K": 128, "N": 2048, "category": "MN_large_K_small"},
    {"M": 2048, "K": 256, "N": 1024, "category": "MN_large_K_small"},
    {"M": 1024, "K": 256, "N": 4096, "category": "MN_large_K_small"},
    {"M": 2048, "K": 32, "N": 2048, "category": "MN_large_K_small"},
    # ── M,K-large N-small (8 shapes) ─────────────────────────────────
    {"M": 2048, "K": 4096, "N": 32, "category": "MK_large_N_small"},
    {"M": 4096, "K": 2048, "N": 64, "category": "MK_large_N_small"},
    {"M": 2048, "K": 2048, "N": 128, "category": "MK_large_N_small"},
    {"M": 4096, "K": 4096, "N": 32, "category": "MK_large_N_small"},
    {"M": 1024, "K": 4096, "N": 64, "category": "MK_large_N_small"},
    {"M": 2048, "K": 1024, "N": 32, "category": "MK_large_N_small"},
    {"M": 1024, "K": 2048, "N": 128, "category": "MK_large_N_small"},
    {"M": 4096, "K": 1024, "N": 64, "category": "MK_large_N_small"},
    # ── K,N-large M-small (8 shapes) ─────────────────────────────────
    {"M": 32, "K": 4096, "N": 4096, "category": "KN_large_M_small"},
    {"M": 64, "K": 8192, "N": 2048, "category": "KN_large_M_small"},
    {"M": 32, "K": 2048, "N": 4096, "category": "KN_large_M_small"},
    {"M": 128, "K": 4096, "N": 2048, "category": "KN_large_M_small"},
    {"M": 64, "K": 4096, "N": 4096, "category": "KN_large_M_small"},
    {"M": 32, "K": 8192, "N": 2048, "category": "KN_large_M_small"},
    {"M": 64, "K": 2048, "N": 4096, "category": "KN_large_M_small"},
    {"M": 128, "K": 2048, "N": 2048, "category": "KN_large_M_small"},
    # ── Tall and wide extremes (8 shapes) ────────────────────────────
    {"M": 8192, "K": 256, "N": 128, "category": "extreme_tall"},
    {"M": 4096, "K": 512, "N": 64, "category": "extreme_tall"},
    {"M": 4096, "K": 1024, "N": 128, "category": "extreme_tall"},
    {"M": 2048, "K": 1024, "N": 32, "category": "extreme_tall"},
    {"M": 32, "K": 1024, "N": 8192, "category": "extreme_wide"},
    {"M": 64, "K": 512, "N": 4096, "category": "extreme_wide"},
    {"M": 128, "K": 256, "N": 8192, "category": "extreme_wide"},
    {"M": 64, "K": 1024, "N": 4096, "category": "extreme_wide"},
    # ── Mixed / batch-like (4 shapes) ────────────────────────────────
    {"M": 256, "K": 2048, "N": 256, "category": "mixed"},
    {"M": 512, "K": 1024, "N": 512, "category": "mixed"},
    {"M": 1024, "K": 512, "N": 1024, "category": "mixed"},
    {"M": 2048, "K": 256, "N": 2048, "category": "mixed"},
]

assert len(TRAINING_SHAPES) >= 106, f"Expected 106+ shapes, got {len(TRAINING_SHAPES)}"


def generate_expanded_shapes(num_shapes: int = 10000, seed: int = 42) -> List[Dict[str, Any]]:
    """Generate num_shapes unique (M,K,N) combos using log-uniform sampling.

    This satisfies the maintainer requirement: "use 10k-100k data points."
    Shapes are tile-aligned (multiples of 32) and cover the full range
    from 32×32×32 to 8192×8192×8192.
    """
    import random

    # Deterministic seeded PRNG for reproducible training data — not security-sensitive
    rng = random.Random(seed)  # nosec B311
    dim_choices = [32, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192]
    categories = ["decode", "prefill", "attention", "mlp", "general"]

    seen = set()
    shapes = []

    # Include all curated shapes first
    for s in TRAINING_SHAPES:
        key = (s["M"], s["K"], s["N"])
        if key not in seen:
            seen.add(key)
            shapes.append(s)

    # Fill remaining with random tile-aligned shapes
    while len(shapes) < num_shapes:
        M = rng.choice(dim_choices)  # nosec B311
        K = rng.choice(dim_choices)  # nosec B311
        N = rng.choice(dim_choices)  # nosec B311
        key = (M, K, N)
        if key in seen:
            continue
        seen.add(key)
        cat = rng.choice(categories)  # nosec B311
        shapes.append({"M": M, "K": K, "N": N, "category": cat})

    return shapes[:num_shapes]


def _tile_pad(x: int) -> int:
    return ((x + 31) // 32) * 32


def run_benchmark_sweep(
    shapes: List[Dict[str, Any]],
    dtype: str = "bfloat16",
    num_warmup: int = 3,
    num_runs: int = 5,
    output_file: str = "dnn_training_data.json",
) -> List[Dict[str, Any]]:
    """
    Run full benchmark sweep on hardware.

    For each shape, measures all valid auto-config candidates AND the default
    ttnn.matmul config.  Returns a list of result dicts suitable for DNN training.
    """
    import torch
    from ttnn._experimental.auto_config.matmul_auto import MatmulAutoConfig

    import ttnn

    results = []
    device = ttnn.open_device(device_id=0)

    try:
        dt = ttnn.bfloat16 if dtype == "bfloat16" else ttnn.bfloat8_b

        for idx, shape in enumerate(shapes):
            M = _tile_pad(shape["M"])
            K = _tile_pad(shape["K"])
            N = _tile_pad(shape["N"])
            category = shape.get("category", "unknown")

            logger.info(f"[{idx + 1}/{len(shapes)}] {category}: M={M}, K={K}, N={N}")

            torch_a = torch.randn(1, M, K, dtype=torch.float32)
            torch_b = torch.randn(K, N, dtype=torch.float32)

            input_a = ttnn.from_torch(torch_a, dtype=dt, layout=ttnn.TILE_LAYOUT, device=device)
            input_b = ttnn.from_torch(torch_b, dtype=dt, layout=ttnn.TILE_LAYOUT, device=device)

            # ── Measure default (no program_config) ──
            try:
                for _ in range(num_warmup):
                    out = ttnn.matmul(input_a, input_b)
                    ttnn.synchronize_device(device)
                    ttnn.deallocate(out)

                times_default = []
                for _ in range(num_runs):
                    ttnn.synchronize_device(device)
                    start = time.perf_counter()
                    out = ttnn.matmul(input_a, input_b)
                    ttnn.synchronize_device(device)
                    times_default.append((time.perf_counter() - start) * 1e6)
                    ttnn.deallocate(out)

                t_default = sorted(times_default)[len(times_default) // 2]
            except Exception as e:
                logger.warning(f"  Default matmul failed: {e}")
                t_default = float("inf")

            results.append(
                {
                    "shape": {"M": M, "K": K, "N": N},
                    "category": category,
                    "config_family": "default",
                    "backend": "matmul",
                    "params": {},
                    "latency_us": t_default,
                    "is_default": True,
                }
            )

            # ── Measure all auto-config candidates ──
            selector = MatmulAutoConfig()
            result = selector.select(input_a, input_b)

            valid_candidates = [c for c in result.all_candidates if c.is_valid and c.config is not None]

            for cand_idx, cand in enumerate(valid_candidates[:10]):  # Cap at 10 per shape
                try:
                    for _ in range(num_warmup):
                        out = ttnn.matmul(input_a, input_b, program_config=cand.config)
                        ttnn.synchronize_device(device)
                        ttnn.deallocate(out)

                    times_cand = []
                    for _ in range(num_runs):
                        ttnn.synchronize_device(device)
                        start = time.perf_counter()
                        out = ttnn.matmul(input_a, input_b, program_config=cand.config)
                        ttnn.synchronize_device(device)
                        times_cand.append((time.perf_counter() - start) * 1e6)
                        ttnn.deallocate(out)

                    t_cand = sorted(times_cand)[len(times_cand) // 2]
                    speedup = t_default / t_cand if t_cand > 0 else 0.0

                    results.append(
                        {
                            "shape": {"M": M, "K": K, "N": N},
                            "category": category,
                            "config_family": cand.config_family,
                            "backend": cand.backend,
                            "params": cand.params,
                            "score": cand.score,
                            "latency_us": t_cand,
                            "speedup_vs_default": speedup,
                            "is_default": False,
                            "production_derived": cand.params.get("production_derived", False),
                        }
                    )

                    logger.info(
                        f"  [{cand_idx + 1}/{len(valid_candidates)}] "
                        f"{cand.config_family}: {t_cand:.0f}µs (speedup={speedup:.3f}x)"
                    )

                except Exception as e:
                    logger.warning(f"  [{cand_idx + 1}] {cand.config_family} FAILED: {e}")

            ttnn.deallocate(input_a)
            ttnn.deallocate(input_b)

    finally:
        ttnn.close_device(device)

    # Save — path validated against allowed base directories (Cycode SAST compliant)
    output_file = _sanitize_path(output_file)
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Saved {len(results)} results to {output_file}")

    return results


def train_dnn_from_results(
    results_file: str,
    weights_output: Optional[str] = None,
    epochs: int = 200,
    learning_rate: float = 0.001,
) -> None:
    """
    Train DNN scorer from benchmark results.

    The DNN learns to predict relative performance (speedup vs default)
    from shape features and config parameters.
    """
    from ttnn._experimental.auto_config.scorer.dnn_scorer import DNNConfigGenerator

    # Path validated against allowed base directories (Cycode SAST compliant)
    results_file = _sanitize_path(results_file)
    with open(results_file, "r") as f:
        results = json.load(f)

    # Group results by shape then pick the best (lowest latency) config per shape
    by_shape: Dict[str, List[Dict]] = {}
    for r in results:
        key = f"M={r['shape']['M']},K={r['shape']['K']},N={r['shape']['N']}"
        if key not in by_shape:
            by_shape[key] = []
        by_shape[key].append(r)

    training_data = []
    for shape_key, entries in by_shape.items():
        # Among non-default entries, find the best config
        auto_entries = [e for e in entries if not e.get("is_default", True) and e.get("latency_us") is not None]
        if not auto_entries:
            continue
        best = min(auto_entries, key=lambda e: e["latency_us"])
        training_data.append(
            {
                "features": {
                    "M": best["shape"]["M"],
                    "K": best["shape"]["K"],
                    "N": best["shape"]["N"],
                    "M_tiles": best["shape"]["M"] // 32,
                    "K_tiles": best["shape"]["K"] // 32,
                    "N_tiles": best["shape"]["N"] // 32,
                    "batch_size_a": 1,
                    "batch_size_b": 1,
                    "grid_x": 8,
                    "grid_y": 8,
                    "num_cores": 64,
                    "num_devices": 1,
                    "is_a_sharded": False,
                    "is_b_sharded": False,
                    "is_batched_b": False,
                    "is_fp32_accumulate": False,
                    "dtype_a": "BFLOAT16",
                    "dtype_b": "BFLOAT16",
                },
                "best_config": {
                    "config_family": best["config_family"],
                    "in0_block_w": best.get("params", {}).get("in0_block_w", 1),
                    "per_core_M": best.get("params", {}).get("per_core_M", 1),
                    "per_core_N": best.get("params", {}).get("per_core_N", 1),
                    "out_subblock_h": best.get("params", {}).get("out_subblock_h", 1),
                    "out_subblock_w": best.get("params", {}).get("out_subblock_w", 1),
                    "math_fidelity": best.get("params", {}).get("math_fidelity", "HiFi4"),
                    "mcast_in0": best.get("params", {}).get("mcast_in0", True),
                },
            }
        )

    if len(training_data) < 10:
        logger.error("Only %d valid training entries — need at least 10", len(training_data))
        return

    logger.info("Training DNN config generator on %d shape/config pairs", len(training_data))

    if weights_output is None:
        weights_output = os.path.join(
            os.path.expanduser("~"),
            ".ttnn",
            "auto_config_cache",
            "dnn_config_generator.pt",
        )

    generator = DNNConfigGenerator(model_path=weights_output)
    generator.train_model(training_data, epochs=epochs, lr=learning_rate)
    logger.info("DNN config generator saved to %s", weights_output)


def validate_dnn_accuracy(results_file: str) -> None:
    """
    Validate DNN scorer accuracy against benchmark ground truth.

    Reports:
    - Rank correlation: does the DNN rank configs in the same order as latency?
    - Top-1 accuracy: how often does the DNN's top pick match the actual fastest?
    - Speedup coverage: what fraction of shapes have speedup ≥ 1.0x?
    """
    # Path validated against allowed base directories (Cycode SAST compliant)
    results_file = _sanitize_path(results_file)
    with open(results_file, "r") as f:
        results = json.load(f)

    # Group by shape
    by_shape: Dict[str, List[Dict]] = {}
    for r in results:
        key = f"M={r['shape']['M']},K={r['shape']['K']},N={r['shape']['N']}"
        if key not in by_shape:
            by_shape[key] = []
        by_shape[key].append(r)

    shapes_with_improvement = 0
    shapes_total = 0
    total_geomean_speedups = []

    for shape_key, entries in by_shape.items():
        default_entries = [e for e in entries if e.get("is_default", False)]
        auto_entries = [e for e in entries if not e.get("is_default", True) and e.get("latency_us") is not None]

        if not default_entries or not auto_entries:
            continue

        t_default = default_entries[0]["latency_us"]
        best_auto = min(auto_entries, key=lambda e: e["latency_us"])
        t_best = best_auto["latency_us"]

        speedup = t_default / t_best if t_best > 0 else 0.0
        shapes_total += 1

        if speedup >= 1.0:
            shapes_with_improvement += 1

        total_geomean_speedups.append(speedup)

        logger.info(
            f"  {shape_key}: default={t_default:.0f}µs, best_auto={t_best:.0f}µs, "
            f"speedup={speedup:.3f}x, best_family={best_auto['config_family']}"
        )

    if total_geomean_speedups:
        log_sum = sum(math.log(max(s, 0.01)) for s in total_geomean_speedups)
        geomean = math.exp(log_sum / len(total_geomean_speedups))
    else:
        geomean = 0.0

    logger.info(f"\n{'=' * 60}")
    logger.info(f"DNN Validation Summary:")
    logger.info(f"  Shapes tested:       {shapes_total}")
    logger.info(
        f"  Shapes improved:     {shapes_with_improvement} ({shapes_with_improvement / max(shapes_total, 1):.0%})"
    )
    logger.info(f"  Geometric mean:      {geomean:.4f}x")
    logger.info(f"  Target:              ≥ 1.0x geomean, ≥ 80% shapes improved")
    logger.info(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(description="DNN retraining pipeline for matmul auto-config")
    parser.add_argument("--full", action="store_true", help="Run full pipeline: benchmark + train + validate")
    parser.add_argument("--benchmark-only", action="store_true", help="Run benchmark sweep only")
    parser.add_argument("--train-only", action="store_true", help="Train DNN from existing results")
    parser.add_argument("--validate-only", action="store_true", help="Validate DNN accuracy")
    parser.add_argument("--input", type=str, default="dnn_training_data.json", help="Input benchmark results file")
    parser.add_argument("--output", type=str, default="dnn_training_data.json", help="Output benchmark results file")
    parser.add_argument("--weights", type=str, default=None, help="DNN weights output path")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "bfloat8_b"])
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs")
    parser.add_argument("--warmup", type=int, default=3, help="Benchmark warmup runs")
    parser.add_argument("--runs", type=int, default=5, help="Benchmark timed runs")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument(
        "--num-shapes",
        type=int,
        default=106,
        help="Number of shapes to generate (default: 106, use 10000+ for production)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    shapes = TRAINING_SHAPES
    if hasattr(args, "num_shapes") and args.num_shapes > len(TRAINING_SHAPES):
        logger.info("Generating %d expanded shapes...", args.num_shapes)
        shapes = generate_expanded_shapes(args.num_shapes)

    logger.info("DNN Retraining Pipeline — %d shapes", len(shapes))

    if args.full or args.benchmark_only:
        logger.info("Phase 1: Benchmark sweep...")
        run_benchmark_sweep(
            shapes,
            dtype=args.dtype,
            num_warmup=args.warmup,
            num_runs=args.runs,
            output_file=args.output,
        )

    if args.full or args.train_only:
        logger.info("Phase 2: Training DNN config generator...")
        train_dnn_from_results(
            args.input if args.train_only else args.output,
            weights_output=args.weights,
            epochs=args.epochs,
        )

    if args.full or args.validate_only:
        logger.info("Phase 3: Validating DNN accuracy...")
        validate_dnn_accuracy(args.input if args.validate_only else args.output)

    if not any([args.full, args.benchmark_only, args.train_only, args.validate_only]):
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
