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

from ttnn.operations.auto_config.base import ConfigCandidate
from ttnn.operations.auto_config.math_fidelity import MathFidelity

import ttnn

logger = logging.getLogger(__name__)

TILE_SIZE = 32

# Explicit sweep bounds — prevents combinatorial explosion
IN0_BLOCK_W_CHOICES = [1, 2, 4, 8]
MAX_CANDIDATES_PER_FAMILY = 8

# Subblock choices ordered by priority (largest to smallest product).
# Mirrors SUBBLOCK_HW_CHOICES from matmul_program_config.cpp (line 16-25).
SUBBLOCK_HW_CHOICES = [
    (4, 2),
    (2, 4),
    (8, 1),
    (1, 8),  # subblock_hw = 8
    (7, 1),
    (1, 7),  # subblock_hw = 7
    (3, 2),
    (2, 3),  # subblock_hw = 6
    (6, 1),
    (1, 6),  # subblock_hw = 6
    (5, 1),
    (1, 5),  # subblock_hw = 5
    (2, 2),  # subblock_hw = 4
    (4, 1),
    (1, 4),  # subblock_hw = 4
    (3, 1),
    (1, 3),  # subblock_hw = 3
    (2, 1),
    (1, 2),  # subblock_hw = 2
    (1, 1),  # subblock_hw = 1
]

# ── Production helpers (ported from models/tt_transformers/tt/model_config.py) ──


def _find_largest_divisor(n: int, max_divisor: int = 8) -> int:
    """Find the largest divisor of n that is <= max_divisor.

    Mirrors ``ModelArgs.find_largest_divisor`` used by every production
    model (Llama, Falcon, Stable Diffusion, etc.) to select ``in0_block_w``.
    """
    for i in range(max_divisor, 0, -1):
        if n % i == 0:
            return i
    return 1


def _find_grid(total_tiles: int, max_rows: int = 8, max_cols: int = 8) -> tuple:
    """Find an (rows, cols) grid such that total_tiles divides evenly.

    Mirrors ``ModelArgs.find_grid`` — targets ~32 cores for best
    utilisation on Wormhole (8×8 grid).
    """
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
    return 1, 1  # Fallback


def _get_out_subblock_w(per_core_N: int, out_subblock_h: int = 1, max_hw: int = 8) -> int:
    """Find the largest out_subblock_w that divides per_core_N and satisfies
    out_subblock_h * out_subblock_w <= max_hw."""
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


# ── Production-derived candidate generators ──
# These replicate the EXACT algorithms used by every production Tenstorrent
# model (Llama, Falcon, Stable Diffusion, DeepSeek, etc.) and are inserted
# at the TOP of the candidate list so the heuristic scorer sees them first.


def _generate_production_1d_candidate(
    features: Dict[str, Any],
) -> List[ConfigCandidate]:
    """Generate a production-quality 1D MultiCast config.

    Replicates ``ModelArgs.matmul_1d_config()`` from model_config.py (line 3309).
    This is the config used by decode/attention paths in Llama, Falcon, etc.
    """
    candidates = []
    M = features["M"]
    K = features["K"]
    N = features["N"]
    batch_size = features["batch_size_a"]
    grid_x = features["grid_x"]
    grid_y = features["grid_y"]
    num_cores = grid_x * grid_y

    if num_cores <= 0:
        return candidates

    m = batch_size * M  # fuse_batch=True
    per_core_m = m // TILE_SIZE
    if per_core_m <= 0:
        return candidates

    n_tiles = N // TILE_SIZE
    k_tiles = K // TILE_SIZE
    if n_tiles <= 0 or k_tiles <= 0:
        return candidates

    # Adjust grid if N is too small for all cores
    effective_grid_x = grid_x
    effective_grid_y = grid_y
    effective_num_cores = num_cores
    if n_tiles < num_cores:
        # Use fewer cores — match production logic
        effective_grid_y = max(1, n_tiles // effective_grid_x)
        effective_num_cores = effective_grid_x * effective_grid_y

    if effective_num_cores <= 0 or k_tiles % effective_num_cores != 0:
        # Cannot evenly divide K across cores — try with fewer cores
        for nc in range(effective_num_cores, 0, -1):
            if k_tiles % nc == 0:
                effective_num_cores = nc
                effective_grid_y = max(1, nc // effective_grid_x)
                effective_grid_x = min(effective_grid_x, nc)
                break

    if effective_num_cores <= 0:
        return candidates

    per_core_k = _find_largest_divisor(k_tiles // effective_num_cores) if k_tiles >= effective_num_cores else 1
    per_core_n = _div_up(n_tiles, effective_num_cores)
    if per_core_n <= 0:
        return candidates

    # Subblock calculation — matches production exactly
    is_fp32 = features.get("is_fp32_accumulate", False)
    max_sub = 4 if is_fp32 else 8

    out_subblock_w = max([i for i in range(1, max_sub + 1) if per_core_n % i == 0], default=1)
    out_subblock_h = max(
        [i for i in range(1, max_sub + 1) if per_core_m % i == 0 and i * out_subblock_w <= max_sub],
        default=1,
    )

    try:
        config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(effective_grid_x, effective_grid_y),
            in0_block_w=per_core_k,
            out_subblock_h=out_subblock_h,
            out_subblock_w=out_subblock_w,
            out_block_h=per_core_m,
            out_block_w=per_core_n,
            per_core_M=per_core_m,
            per_core_N=per_core_n,
            fuse_batch=True,
            mcast_in0=True,
        )
        candidates.append(
            ConfigCandidate(
                config=config,
                config_family="MultiCast1D",
                backend="matmul",
                params={
                    "mcast_in0": True,
                    "in0_block_w": per_core_k,
                    "per_core_M": per_core_m,
                    "per_core_N": per_core_n,
                    "out_subblock_h": out_subblock_h,
                    "out_subblock_w": out_subblock_w,
                    "production_derived": True,
                },
            )
        )
    except Exception as e:
        logger.debug(f"Production 1D config failed: {e}")

    return candidates


def _generate_production_2d_candidate(
    features: Dict[str, Any],
) -> List[ConfigCandidate]:
    """Generate a production-quality 2D MultiCast config.

    Replicates ``ModelArgs.matmul_config()`` from model_config.py (line 3076).
    This is the config used by prefill/large-matrix paths.
    """
    candidates = []
    M = features["M"]
    K = features["K"]
    N = features["N"]
    batch_size = features["batch_size_a"]
    grid_x = features["grid_x"]
    grid_y = features["grid_y"]

    m = batch_size * M
    m_tiles = m // TILE_SIZE
    k_tiles = K // TILE_SIZE
    n_tiles = N // TILE_SIZE

    if m_tiles <= 0 or k_tiles <= 0 or n_tiles <= 0:
        return candidates

    per_core_M = _div_up(m_tiles, grid_y)
    per_core_N = _div_up(n_tiles, grid_x)

    if per_core_M <= 0 or per_core_N <= 0:
        return candidates

    # in0_block_w — production formula
    k_per_grid = k_tiles // grid_y if k_tiles % grid_y == 0 else k_tiles
    in0_block_w = _find_largest_divisor(k_per_grid) if k_per_grid > 0 else 1

    out_subblock_h = 1
    out_subblock_w = _get_out_subblock_w(per_core_N, out_subblock_h)

    try:
        config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(grid_x, grid_y),
            in0_block_w=in0_block_w,
            out_subblock_h=out_subblock_h,
            out_subblock_w=out_subblock_w,
            out_block_h=per_core_M,
            out_block_w=per_core_N,
            per_core_M=per_core_M,
            per_core_N=per_core_N,
            transpose_mcast=False,
            fuse_batch=True,
        )
        candidates.append(
            ConfigCandidate(
                config=config,
                config_family="MultiCast2D",
                backend="matmul",
                params={
                    "transpose_mcast": False,
                    "in0_block_w": in0_block_w,
                    "per_core_M": per_core_M,
                    "per_core_N": per_core_N,
                    "out_subblock_h": out_subblock_h,
                    "out_subblock_w": out_subblock_w,
                    "production_derived": True,
                },
            )
        )
    except Exception as e:
        logger.debug(f"Production 2D config failed: {e}")

    return candidates


def _generate_production_dram_candidate(
    features: Dict[str, Any],
) -> List[ConfigCandidate]:
    """Generate a production-quality DRAM-sharded config.

    Replicates ``ModelArgs.dram_matmul_config()`` from model_config.py (line 3230).
    Uses ``find_grid`` to determine optimal core count.
    """
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

    # Use find_grid to get optimal core count — mirrors production
    try:
        rows, cols = _find_grid(k_tiles)
        num_cores = rows * cols
    except Exception:
        num_cores = 1

    if num_cores <= 0 or k_tiles % num_cores != 0:
        return candidates

    in0_block_w = _find_largest_divisor(k_tiles // num_cores)
    per_core_M = m_tiles
    per_core_N = _div_up(n_tiles, num_cores)

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
                    "production_derived": True,
                },
            )
        )
    except Exception as e:
        logger.debug(f"Production DRAM config failed: {e}")

    return candidates


def generate_matmul_candidates(features: Dict[str, Any]) -> List[ConfigCandidate]:
    """
    Generate all valid matmul config candidates for the given features.

    This exhaustively explores the decision tree from matmul_program_config.cpp,
    producing multiple alternatives instead of a single choice. Each family
    is capped at MAX_CANDIDATES_PER_FAMILY to prevent combinatorial explosion.

    Production-derived candidates (from model_config.py) are inserted FIRST,
    giving them priority during scoring and selection.

    Math fidelity is iterated as a first-class dimension: each program config
    candidate is duplicated across all valid fidelities for the input dtype pair.

    Returns:
        List of ConfigCandidate objects, each carrying a math_fidelity field.
    """
    base_candidates = []

    # ── Production-derived candidates (highest priority) ──
    # These replicate the exact algorithms used by Llama, Falcon, etc.
    base_candidates.extend(_generate_production_1d_candidate(features))
    base_candidates.extend(_generate_production_2d_candidate(features))
    base_candidates.extend(_generate_production_dram_candidate(features))
    logger.debug(f"Production candidates: {len(base_candidates)}")

    # ── Swept candidates (additional exploration) ──
    # Path 1: MultiCast 1D (tall and wide variants)
    base_candidates.extend(_generate_multicast_1d_candidates(features))
    logger.debug(f"After MultiCast1D: {len(base_candidates)} candidates")

    # Path 2: MultiCast 2D
    base_candidates.extend(_generate_multicast_2d_candidates(features))
    logger.debug(f"After MultiCast2D: {len(base_candidates)} candidates")

    # Path 3: Reuse (for batched B)
    base_candidates.extend(_generate_reuse_candidates(features))

    # Path 4: DRAM-sharded
    base_candidates.extend(_generate_dram_sharded_candidates(features))

    # Path 5: Minimal matmul (experimental)
    try:
        base_candidates.extend(_generate_minimal_matmul_candidates(features))
    except Exception:
        pass  # MinimalMatmul not available

    # --- Math fidelity expansion ---
    # Get valid fidelities for the dtype pair from features.
    # Duplicate each base candidate across all valid fidelities.
    fidelities = features.get("math_fidelity_valid")
    if fidelities and len(fidelities) > 0:
        candidates = []
        for cand in base_candidates:
            for fid in fidelities:
                # Create a new candidate with this fidelity attached
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
        # No fidelity info — use default (HiFi4) for all candidates
        candidates = base_candidates
        for cand in candidates:
            cand.math_fidelity = MathFidelity.HiFi4

    logger.debug(f"Total candidates generated: {len(candidates)}")
    return candidates
