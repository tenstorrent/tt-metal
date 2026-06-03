# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Candidate configuration generator for matmul operations."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from ttnn._experimental.auto_config.base import ConfigCandidate
from ttnn._experimental.auto_config.math_fidelity import MathFidelity

import ttnn

logger = logging.getLogger(__name__)

TILE_SIZE = 32

IN0_BLOCK_W_CHOICES = [1, 2, 4, 8]

# Subblock choices ordered by priority (largest to smallest product).
SUBBLOCK_HW_CHOICES = [
    (4, 2),
    (2, 4),
    (8, 1),
    (1, 8),
    (7, 1),
    (1, 7),
    (3, 2),
    (2, 3),
    (6, 1),
    (1, 6),
    (5, 1),
    (1, 5),
    (2, 2),
    (4, 1),
    (1, 4),
    (3, 1),
    (1, 3),
    (2, 1),
    (1, 2),
    (1, 1),
]


def _quick_l1_check(
    in0_block_w: int,
    per_core_M: int,
    per_core_N: int,
    features: Dict[str, Any],
) -> bool:
    """Fast L1 budget pre-check to skip obviously-invalid candidates."""
    dtype_a = features.get("dtype_a", "DataType.BFLOAT16")
    tile_bytes = 2048 if dtype_a == "DataType.BFLOAT16" else 1088

    cb_in0_tiles = in0_block_w * per_core_M * 2
    cb_in1_tiles = in0_block_w * per_core_N * 2
    cb_out_tiles = per_core_M * per_core_N
    fp32_accum_bytes = 1 * 1 * 32 * 32 * 4

    total_cb_bytes = (cb_in0_tiles + cb_in1_tiles + cb_out_tiles) * tile_bytes + fp32_accum_bytes

    DEFAULT_L1_USABLE = 1_258_291
    max_l1 = features.get("l1_usable_bytes", DEFAULT_L1_USABLE)
    if max_l1 is None:
        max_l1 = DEFAULT_L1_USABLE

    return total_cb_bytes <= max_l1


def _find_largest_divisor(n: int, max_divisor: int = 8) -> int:
    """Find the largest divisor of n that is <= max_divisor."""
    for i in range(max_divisor, 0, -1):
        if n % i == 0:
            return i
    return 1


def _find_grid(total_tiles: int, max_rows: int = 8, max_cols: int = 8) -> tuple:
    """Find an (rows, cols) grid such that total_tiles divides evenly."""
    max_cores = max_rows * max_cols
    target = 32
    possible = [k for k in range(1, max_cores + 1) if total_tiles % k == 0]
    possible.sort(key=lambda x: abs(x - target))
    for cores in possible:
        for rows in range(1, max_rows + 1):
            if cores % rows == 0:
                cols = cores // rows
                if cols <= max_cols:
                    return rows, cols
    return 1, 1


def _get_out_subblock_w(per_core_N: int, out_subblock_h: int = 1, max_hw: int = 8) -> int:
    """Find the largest out_subblock_w that divides per_core_N and satisfies constraints."""
    for w in range(min(max_hw // max(out_subblock_h, 1), per_core_N), 0, -1):
        if per_core_N % w == 0 and out_subblock_h * w <= max_hw:
            return w
    return 1


def _get_subblock_sizes(m_tiles_per_core: int, n_tiles_per_core: int, fp32_dest_acc_en: bool = False) -> tuple:
    """Find the best (out_subblock_h, out_subblock_w) for given per_core dimensions."""
    for w, h in SUBBLOCK_HW_CHOICES:
        if fp32_dest_acc_en and (h * w) > 4:
            continue
        if m_tiles_per_core % h == 0 and n_tiles_per_core % w == 0:
            return h, w
    return 1, 1


def _div_up(a: int, b: int) -> int:
    return (a + b - 1) // b


def _generate_multicast_1d_candidates(features: Dict[str, Any]) -> List[ConfigCandidate]:
    """Generate MatmulMultiCoreReuseMultiCast1DProgramConfig candidates."""
    candidates = []
    M = features["M"]
    K = features["K"]
    N = features["N"]
    batch_size = features["batch_size_a"]
    grid_x = features["grid_x"]
    grid_y = features["grid_y"]
    num_cores = grid_x * grid_y

    batch_and_m_tiles = (batch_size * M) // TILE_SIZE
    k_tiles = K // TILE_SIZE
    n_tiles = N // TILE_SIZE

    if batch_and_m_tiles <= 0 or k_tiles <= 0 or n_tiles <= 0:
        return candidates

    for mcast_in0 in [True, False]:
        if mcast_in0:
            per_core_M = batch_and_m_tiles
            per_core_N = _div_up(n_tiles, num_cores)
        else:
            per_core_M = _div_up(batch_and_m_tiles, num_cores)
            per_core_N = n_tiles

        if per_core_M <= 0 or per_core_N <= 0:
            continue

        for in0_block_w in IN0_BLOCK_W_CHOICES:
            in0_block_w = min(in0_block_w, k_tiles)
            if k_tiles % in0_block_w != 0:
                continue

            if not _quick_l1_check(in0_block_w, per_core_M, per_core_N, features):
                continue

            out_subblock_h, out_subblock_w = _get_subblock_sizes(per_core_M, per_core_N)

            try:
                config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                    compute_with_storage_grid_size=ttnn.CoreCoord(grid_x, grid_y),
                    in0_block_w=in0_block_w,
                    out_subblock_h=out_subblock_h,
                    out_subblock_w=out_subblock_w,
                    out_block_h=per_core_M,
                    out_block_w=per_core_N,
                    per_core_M=per_core_M,
                    per_core_N=per_core_N,
                    fuse_batch=True,
                    mcast_in0=mcast_in0,
                )
                candidates.append(
                    ConfigCandidate(
                        config=config,
                        config_family="MultiCast1D",
                        backend="matmul",
                        params={
                            "mcast_in0": mcast_in0,
                            "in0_block_w": in0_block_w,
                            "per_core_M": per_core_M,
                            "per_core_N": per_core_N,
                            "out_subblock_h": out_subblock_h,
                            "out_subblock_w": out_subblock_w,
                        },
                    )
                )
            except Exception as e:
                logger.debug(f"Failed to create MultiCast1D config: {e}")

    return candidates


def _generate_multicast_2d_candidates(features: Dict[str, Any]) -> List[ConfigCandidate]:
    """Generate MatmulMultiCoreReuseMultiCastProgramConfig candidates."""
    candidates = []
    M = features["M"]
    K = features["K"]
    N = features["N"]
    batch_size = features["batch_size_a"]
    grid_x = features["grid_x"]
    grid_y = features["grid_y"]

    m_tiles = (batch_size * M) // TILE_SIZE
    k_tiles = K // TILE_SIZE
    n_tiles = N // TILE_SIZE

    if m_tiles <= 0 or k_tiles <= 0 or n_tiles <= 0:
        return candidates

    per_core_M = _div_up(m_tiles, grid_y)
    per_core_N = _div_up(n_tiles, grid_x)

    if per_core_M <= 0 or per_core_N <= 0:
        return candidates

    for transpose_mcast in [False, True]:
        if transpose_mcast:
            per_core_M_eff = _div_up(m_tiles, grid_x)
            per_core_N_eff = _div_up(n_tiles, grid_y)
        else:
            per_core_M_eff = per_core_M
            per_core_N_eff = per_core_N

        if per_core_M_eff <= 0 or per_core_N_eff <= 0:
            continue

        for in0_block_w in IN0_BLOCK_W_CHOICES:
            if k_tiles % in0_block_w != 0:
                if in0_block_w > 1:
                    continue

            if not _quick_l1_check(in0_block_w, per_core_M_eff, per_core_N_eff, features):
                continue

            out_subblock_h, out_subblock_w = _get_subblock_sizes(per_core_M_eff, per_core_N_eff)

            try:
                config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=ttnn.CoreCoord(grid_x, grid_y),
                    in0_block_w=in0_block_w,
                    out_subblock_h=out_subblock_h,
                    out_subblock_w=out_subblock_w,
                    out_block_h=per_core_M_eff,
                    out_block_w=per_core_N_eff,
                    per_core_M=per_core_M_eff,
                    per_core_N=per_core_N_eff,
                    transpose_mcast=transpose_mcast,
                    fuse_batch=True,
                )
                candidates.append(
                    ConfigCandidate(
                        config=config,
                        config_family="MultiCast2D",
                        backend="matmul",
                        params={
                            "transpose_mcast": transpose_mcast,
                            "in0_block_w": in0_block_w,
                            "per_core_M": per_core_M_eff,
                            "per_core_N": per_core_N_eff,
                            "out_subblock_h": out_subblock_h,
                            "out_subblock_w": out_subblock_w,
                        },
                    )
                )
            except Exception as e:
                logger.debug(f"Failed to create MultiCast2D config: {e}")

    return candidates


def _generate_reuse_candidates(features: Dict[str, Any]) -> List[ConfigCandidate]:
    """Generate MatmulMultiCoreReuseProgramConfig candidates (for batched B)."""
    candidates = []

    if not features["is_batched_b"]:
        return candidates

    M = features["M"]
    K = features["K"]
    N = features["N"]
    grid_x = features["grid_x"]
    grid_y = features["grid_y"]

    m_tiles = M // TILE_SIZE
    k_tiles = K // TILE_SIZE
    n_tiles = N // TILE_SIZE

    if m_tiles <= 0 or k_tiles <= 0 or n_tiles <= 0:
        return candidates

    per_core_M = m_tiles
    per_core_N = n_tiles

    for in0_block_w in [1, 2, 4]:
        if k_tiles % in0_block_w != 0:
            continue

        out_subblock_h, out_subblock_w = _get_subblock_sizes(per_core_M, per_core_N)

        try:
            config = ttnn.MatmulMultiCoreReuseProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(grid_x, grid_y),
                in0_block_w=in0_block_w,
                out_subblock_h=out_subblock_h,
                out_subblock_w=out_subblock_w,
                per_core_M=per_core_M,
                per_core_N=per_core_N,
            )
            candidates.append(
                ConfigCandidate(
                    config=config,
                    config_family="Reuse",
                    backend="matmul",
                    params={
                        "in0_block_w": in0_block_w,
                        "per_core_M": per_core_M,
                        "per_core_N": per_core_N,
                        "out_subblock_h": out_subblock_h,
                        "out_subblock_w": out_subblock_w,
                    },
                )
            )
        except Exception as e:
            logger.debug(f"Failed to create Reuse config: {e}")

    return candidates


def _generate_dram_sharded_candidates(features: Dict[str, Any]) -> List[ConfigCandidate]:
    """Generate DRAM-sharded config candidates."""
    candidates = []

    if not features["is_a_sharded"]:
        return candidates

    M = features["M"]
    K = features["K"]
    N = features["N"]
    batch_size = features["batch_size_a"]

    m_tiles = (batch_size * M) // TILE_SIZE
    k_tiles = K // TILE_SIZE
    n_tiles = N // TILE_SIZE

    if m_tiles <= 0 or k_tiles <= 0 or n_tiles <= 0:
        return candidates

    per_core_M = m_tiles
    per_core_N = n_tiles

    for in0_block_w in IN0_BLOCK_W_CHOICES:
        if k_tiles % in0_block_w != 0:
            continue

        try:
            config = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                in0_block_w=in0_block_w,
                per_core_M=per_core_M,
                per_core_N=per_core_N,
            )
            candidates.append(
                ConfigCandidate(
                    config=config,
                    config_family="DRAMSharded",
                    backend="matmul",
                    params={
                        "in0_block_w": in0_block_w,
                        "per_core_M": per_core_M,
                        "per_core_N": per_core_N,
                    },
                )
            )
        except Exception as e:
            logger.debug(f"Failed to create DRAMSharded config: {e}")

    return candidates


def generate_matmul_candidates(features: Dict[str, Any]) -> List[ConfigCandidate]:
    """Generate all valid matmul config candidates for the given features."""
    base_candidates = []

    base_candidates.extend(_generate_multicast_1d_candidates(features))
    base_candidates.extend(_generate_multicast_2d_candidates(features))
    base_candidates.extend(_generate_reuse_candidates(features))
    base_candidates.extend(_generate_dram_sharded_candidates(features))

    # Math fidelity expansion: duplicate each candidate across valid fidelities
    fidelities = features.get("math_fidelity_valid")
    if fidelities and len(fidelities) > 0:
        candidates = []
        for cand in base_candidates:
            for fid in fidelities:
                fid_cand = ConfigCandidate(
                    config=cand.config,
                    config_family=cand.config_family,
                    backend=cand.backend,
                    params={**cand.params, "math_fidelity": fid.name},
                    math_fidelity=fid,
                )
                candidates.append(fid_cand)
        logger.debug(
            f"Fidelity expansion: {len(base_candidates)} base × "
            f"{len(fidelities)} fidelities = {len(candidates)} total"
        )
    else:
        candidates = base_candidates
        for cand in candidates:
            cand.math_fidelity = MathFidelity.HiFi4

    logger.debug(f"Total candidates generated: {len(candidates)}")
    return candidates


def _build_config_from_params(
    params: Dict[str, Any],
    features: Dict[str, Any],
) -> Optional[ConfigCandidate]:
    """Build a ConfigCandidate from predicted or cached config parameters."""
    family = params.get("config_family", "MultiCast1D")
    grid_x = features.get("grid_x", 8)
    grid_y = features.get("grid_y", 8)
    in0_block_w = params.get("in0_block_w", 1)
    per_core_M = params.get("per_core_M", 1)
    per_core_N = params.get("per_core_N", 1)
    out_subblock_h = params.get("out_subblock_h", 1)
    out_subblock_w = params.get("out_subblock_w", 1)
    mcast_in0 = params.get("mcast_in0", True)

    config = None
    backend = "matmul"

    try:
        if family == "MultiCast1D":
            config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(grid_x, grid_y),
                in0_block_w=in0_block_w,
                out_subblock_h=out_subblock_h,
                out_subblock_w=out_subblock_w,
                out_block_h=per_core_M,
                out_block_w=per_core_N,
                per_core_M=per_core_M,
                per_core_N=per_core_N,
                fuse_batch=True,
                mcast_in0=mcast_in0,
            )
        elif family == "MultiCast2D":
            config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(grid_x, grid_y),
                in0_block_w=in0_block_w,
                out_subblock_h=out_subblock_h,
                out_subblock_w=out_subblock_w,
                out_block_h=per_core_M,
                out_block_w=per_core_N,
                per_core_M=per_core_M,
                per_core_N=per_core_N,
                transpose_mcast=params.get("transpose_mcast", False),
                fuse_batch=True,
            )
        elif family == "DRAMSharded":
            config = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                in0_block_w=in0_block_w,
                per_core_M=per_core_M,
                per_core_N=per_core_N,
            )
        elif family == "Reuse":
            config = ttnn.MatmulMultiCoreReuseProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(grid_x, grid_y),
                in0_block_w=in0_block_w,
                out_subblock_h=out_subblock_h,
                out_subblock_w=out_subblock_w,
                per_core_M=per_core_M,
                per_core_N=per_core_N,
            )
        else:
            logger.debug("Cannot build config for family: %s", family)
            return None
    except Exception as e:
        logger.debug("Failed to build config from params: %s", e)
        return None

    fidelity_name = params.get("math_fidelity", "HiFi4")
    try:
        fidelity = MathFidelity[fidelity_name]
    except (KeyError, TypeError):
        fidelity = MathFidelity.HiFi4

    return ConfigCandidate(
        config=config,
        config_family=family,
        backend=backend,
        params=params,
        math_fidelity=fidelity,
    )
