# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Candidate configuration generator for matmul operations.

Generates config candidates by exhaustively exploring the existing decision
tree in matmul_program_config.cpp, producing multiple alternatives instead
of a single choice. Each candidate is a valid MatmulProgramConfig variant.

Sweep bounds are capped to prevent combinatorial explosion (MAX_CANDIDATES_PER_FAMILY).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import ttnn

from ttnn.operations.auto_config.base import ConfigCandidate

logger = logging.getLogger(__name__)

TILE_SIZE = 32

# Explicit sweep bounds — prevents combinatorial explosion
IN0_BLOCK_W_CHOICES = [1, 2, 4, 8]
MAX_CANDIDATES_PER_FAMILY = 8

# Subblock choices ordered by priority (largest to smallest product).
# Mirrors SUBBLOCK_HW_CHOICES from matmul_program_config.cpp (line 16-25).
SUBBLOCK_HW_CHOICES = [
    (4, 2), (2, 4),          # subblock_hw = 8
    (7, 1), (1, 7),          # subblock_hw = 7
    (3, 2), (2, 3),          # subblock_hw = 6
    (6, 1), (1, 6),          # subblock_hw = 6
    (5, 1), (1, 5),          # subblock_hw = 5
    (2, 2),                  # subblock_hw = 4
    (4, 1), (1, 4),          # subblock_hw = 4
    (3, 1), (1, 3),          # subblock_hw = 3
    (2, 1), (1, 2),          # subblock_hw = 2
    (1, 1),                  # subblock_hw = 1
]


def _get_subblock_sizes(
    m_tiles_per_core: int, n_tiles_per_core: int, fp32_dest_acc_en: bool = False
) -> tuple:
    """Find the best (out_subblock_h, out_subblock_w) for given per_core dimensions."""
    for w, h in SUBBLOCK_HW_CHOICES:
        if fp32_dest_acc_en and (h * w) > 4:
            continue
        if m_tiles_per_core % h == 0 and n_tiles_per_core % w == 0:
            return h, w
    return 1, 1


def _div_up(a: int, b: int) -> int:
    return (a + b - 1) // b


def _generate_multicast_1d_candidates(
    features: Dict[str, Any],
) -> List[ConfigCandidate]:
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

    # Generate candidates for both mcast_in0=True (wide) and mcast_in0=False (tall)
    for mcast_in0 in [True, False]:
        if mcast_in0:
            # Wide: mcast in0, each core gets a slice of N
            per_core_M = batch_and_m_tiles
            per_core_N = _div_up(n_tiles, num_cores)
        else:
            # Tall: mcast in1, each core gets a slice of M
            per_core_M = _div_up(batch_and_m_tiles, num_cores)
            per_core_N = n_tiles

        if per_core_M <= 0 or per_core_N <= 0:
            continue

        # Sweep in0_block_w with bounded choices
        for in0_block_w in IN0_BLOCK_W_CHOICES:
            if len(candidates) >= MAX_CANDIDATES_PER_FAMILY:
                break

            in0_block_w = min(in0_block_w, k_tiles)
            if k_tiles % in0_block_w != 0:
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

    return candidates[:MAX_CANDIDATES_PER_FAMILY]


def _generate_multicast_2d_candidates(
    features: Dict[str, Any],
) -> List[ConfigCandidate]:
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
            if len(candidates) >= MAX_CANDIDATES_PER_FAMILY:
                break

            if k_tiles % in0_block_w != 0:
                if in0_block_w > 1:
                    continue
                # in0_block_w=1 always works

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

    return candidates[:MAX_CANDIDATES_PER_FAMILY]


def _generate_reuse_candidates(
    features: Dict[str, Any],
) -> List[ConfigCandidate]:
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

    return candidates[:MAX_CANDIDATES_PER_FAMILY]


def _generate_dram_sharded_candidates(
    features: Dict[str, Any],
) -> List[ConfigCandidate]:
    """Generate DRAM-sharded config candidates."""
    candidates = []

    # DRAM-sharded only valid for non-sharded, DRAM-interleaved inputs
    if features["is_a_sharded"] or "DRAM" not in features["buffer_type_a"]:
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

    return candidates[:MAX_CANDIDATES_PER_FAMILY]


def _generate_minimal_matmul_candidates(
    features: Dict[str, Any],
) -> List[ConfigCandidate]:
    """Generate MinimalMatmulConfig candidates (experimental API)."""
    candidates = []

    # minimal_matmul only works with bfloat16/bfloat8_b in TILE layout
    valid_dtypes = {"DataType.BFLOAT16", "DataType.BFLOAT8_B"}
    if features["dtype_a"] not in valid_dtypes:
        return candidates
    if features["layout_a"] != "Layout.TILE":
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

    # Try a few block size configurations — bounded sweep
    for m_block in [1, 2, 4]:
        for k_block in [1, 2, 4]:
            for n_block in [1, 2, 4]:
                if len(candidates) >= MAX_CANDIDATES_PER_FAMILY:
                    break
                if m_tiles % m_block != 0:
                    continue
                if k_tiles % k_block != 0:
                    continue
                if n_tiles % n_block != 0:
                    continue

                out_subblock_h, out_subblock_w = _get_subblock_sizes(m_block, n_block)

                try:
                    config = ttnn.experimental.MinimalMatmulConfig(
                        M_block_size=m_block,
                        K_block_size=k_block,
                        N_block_size=n_block,
                        subblock_h=out_subblock_h,
                        subblock_w=out_subblock_w,
                        grid_size=ttnn.CoreCoord(grid_x, grid_y),
                    )
                    candidates.append(
                        ConfigCandidate(
                            config=config,
                            config_family="MinimalMatmul",
                            backend="minimal_matmul",
                            params={
                                "M_block_size": m_block,
                                "K_block_size": k_block,
                                "N_block_size": n_block,
                                "subblock_h": out_subblock_h,
                                "subblock_w": out_subblock_w,
                            },
                        )
                    )
                except Exception:
                    # MinimalMatmulConfig may not be available in all builds
                    pass

    return candidates[:MAX_CANDIDATES_PER_FAMILY]


def generate_matmul_candidates(features: Dict[str, Any]) -> List[ConfigCandidate]:
    """
    Generate all valid matmul config candidates for the given features.

    This exhaustively explores the decision tree from matmul_program_config.cpp,
    producing multiple alternatives instead of a single choice. Each family
    is capped at MAX_CANDIDATES_PER_FAMILY to prevent combinatorial explosion.

    Returns:
        List of ConfigCandidate objects.
    """
    candidates = []

    # Path 1: MultiCast 1D (tall and wide variants)
    candidates.extend(_generate_multicast_1d_candidates(features))
    logger.debug(f"MultiCast1D: {len(candidates)} candidates so far")

    # Path 2: MultiCast 2D
    candidates.extend(_generate_multicast_2d_candidates(features))
    logger.debug(f"After MultiCast2D: {len(candidates)} candidates")

    # Path 3: Reuse (for batched B)
    candidates.extend(_generate_reuse_candidates(features))

    # Path 4: DRAM-sharded
    candidates.extend(_generate_dram_sharded_candidates(features))

    # Path 5: Minimal matmul (experimental)
    try:
        candidates.extend(_generate_minimal_matmul_candidates(features))
    except Exception:
        pass  # MinimalMatmul not available

    logger.debug(f"Total candidates generated: {len(candidates)}")
    return candidates
