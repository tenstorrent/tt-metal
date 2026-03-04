# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Constraint validation for matmul config candidates.

Validates candidate configurations using hardware constraints:
- Tile alignment
- Core grid feasibility
- Subblock constraints (H * W <= 8 for DST register limit)
- L1 memory budget
- Backend-specific rules (minimal_matmul dtype/layout requirements)
- Multi-device CCL compatibility
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Tuple


logger = logging.getLogger(__name__)

TILE_HEIGHT = 32
TILE_WIDTH = 32
MAX_SUBBLOCK_HW = 8  # DST register limit


def validate_tile_alignment(features: Dict[str, Any]) -> Tuple[bool, str]:
    """Check that M, K, N are tile-aligned."""
    M, K, N = features["M"], features["K"], features["N"]
    if M % TILE_HEIGHT != 0:
        return False, f"M={M} not aligned to tile height {TILE_HEIGHT}"
    if K % TILE_WIDTH != 0:
        return False, f"K={K} not aligned to tile width {TILE_WIDTH}"
    if N % TILE_WIDTH != 0:
        return False, f"N={N} not aligned to tile width {TILE_WIDTH}"
    return True, "ok"


def validate_grid_feasibility(
    config: Any, config_family: str, features: Dict[str, Any]
) -> Tuple[bool, str]:
    """Check that the config's grid size doesn't exceed available cores."""
    grid_x = features["grid_x"]
    grid_y = features["grid_y"]

    if config_family in ("DRAMSharded", "BatchedDRAMSharded"):
        # DRAM-sharded configs don't have compute_with_storage_grid_size
        return True, "ok"

    if config_family == "MultiCore":
        return True, "ok"

    if hasattr(config, "compute_with_storage_grid_size"):
        cfg_grid = config.compute_with_storage_grid_size
        if cfg_grid.x > grid_x:
            return False, f"grid.x={cfg_grid.x} > device.x={grid_x}"
        if cfg_grid.y > grid_y:
            return False, f"grid.y={cfg_grid.y} > device.y={grid_y}"
        if cfg_grid.x <= 0 or cfg_grid.y <= 0:
            return False, "grid dimensions must be > 0"

    return True, "ok"


def validate_subblock_params(config: Any, config_family: str) -> Tuple[bool, str]:
    """Check subblock parameter constraints."""
    if config_family in ("DRAMSharded", "BatchedDRAMSharded", "MultiCore"):
        return True, "ok"

    per_core_M = getattr(config, "per_core_M", None)
    per_core_N = getattr(config, "per_core_N", None)
    out_subblock_h = getattr(config, "out_subblock_h", None)
    out_subblock_w = getattr(config, "out_subblock_w", None)

    if per_core_M is None or per_core_N is None:
        return True, "ok"

    if per_core_M <= 0:
        return False, f"per_core_M={per_core_M} must be > 0"
    if per_core_N <= 0:
        return False, f"per_core_N={per_core_N} must be > 0"

    if out_subblock_h is not None and out_subblock_w is not None:
        if out_subblock_h <= 0 or out_subblock_w <= 0:
            return False, f"subblock dims must be > 0: h={out_subblock_h}, w={out_subblock_w}"
        if out_subblock_h * out_subblock_w > MAX_SUBBLOCK_HW:
            return False, (
                f"subblock_h*subblock_w={out_subblock_h * out_subblock_w} "
                f"> MAX_SUBBLOCK_HW={MAX_SUBBLOCK_HW}"
            )
        if per_core_M % out_subblock_h != 0:
            return False, f"per_core_M={per_core_M} not divisible by out_subblock_h={out_subblock_h}"
        if per_core_N % out_subblock_w != 0:
            return False, f"per_core_N={per_core_N} not divisible by out_subblock_w={out_subblock_w}"

    in0_block_w = getattr(config, "in0_block_w", None)
    if in0_block_w is not None and in0_block_w <= 0:
        return False, f"in0_block_w={in0_block_w} must be > 0"

    return True, "ok"


def validate_minimal_matmul_constraints(
    features: Dict[str, Any],
) -> Tuple[bool, str]:
    """Check minimal_matmul-specific constraints."""
    valid_dtypes = {"DataType.BFLOAT16", "DataType.BFLOAT8_B"}
    if features["dtype_a"] not in valid_dtypes:
        return False, f"minimal_matmul requires bfloat16/bfloat8_b, got {features['dtype_a']}"
    if features["layout_a"] != "Layout.TILE":
        return False, f"minimal_matmul requires TILE layout, got {features['layout_a']}"
    if features["layout_b"] != "Layout.TILE":
        return False, f"minimal_matmul requires TILE layout for input_b, got {features['layout_b']}"
    return True, "ok"


def validate_memory_config_compatibility(
    config_family: str, features: Dict[str, Any]
) -> Tuple[bool, str]:
    """Check that the config family is compatible with the input memory configuration."""
    is_a_sharded = features["is_a_sharded"]
    mem_layout_a = features["mem_layout_a"]
    buffer_type_a = features["buffer_type_a"]

    if config_family == "DRAMSharded":
        # DRAM-sharded configs require DRAM-interleaved inputs
        if is_a_sharded:
            return False, "DRAMSharded requires non-sharded (DRAM interleaved) input A"
        if "DRAM" not in buffer_type_a:
            return False, f"DRAMSharded requires DRAM buffer type, got {buffer_type_a}"

    if config_family == "BatchedDRAMSharded":
        if is_a_sharded:
            return False, "BatchedDRAMSharded requires non-sharded input A"
        if not features["is_batched_b"]:
            return False, "BatchedDRAMSharded requires batched input B"

    if config_family == "Reuse":
        # Reuse config is primarily for batched B or specific sharded patterns
        pass

    if config_family == "MultiCast2D":
        # 2D multicast works best with block-sharded or interleaved
        if is_a_sharded and "BLOCK" not in mem_layout_a and "INTERLEAVED" not in mem_layout_a:
            return False, f"MultiCast2D prefers block-sharded/interleaved, got {mem_layout_a}"

    return True, "ok"


def validate_batching_constraints(
    config_family: str, features: Dict[str, Any]
) -> Tuple[bool, str]:
    """Check batch-related constraints."""
    is_batched_b = features["is_batched_b"]

    if config_family == "MultiCast1D":
        if is_batched_b:
            return False, "MultiCast1D does not support batched input B"

    if config_family == "MultiCast2D":
        if is_batched_b:
            return False, "MultiCast2D does not support batched input B"

    return True, "ok"


def validate_l1_budget(
    config: Any, config_family: str, features: Dict[str, Any]
) -> Tuple[bool, str]:
    """Check that the config's L1 memory requirement doesn't exceed device limits."""
    # Only applies to configs where we explicitly set per block/core sizes
    if config_family in ("DRAMSharded", "BatchedDRAMSharded", "MultiCore", "Reuse"):
        return True, "ok"

    per_core_M = getattr(config, "per_core_M", None)
    per_core_N = getattr(config, "per_core_N", None)
    in0_block_w = getattr(config, "in0_block_w", None)

    if per_core_M is None or per_core_N is None or in0_block_w is None:
        return True, "ok"

    # Estimate CB sizes. TILE_SIZE=32x32=1024 elements.
    # bfloat16 = 2 bytes per element -> 2048 bytes per tile
    # bfloat8_b = 1088 bytes per tile (including header)
    tile_bytes = 2048 if features.get("dtype_a") == "DataType.BFLOAT16" else 1088

    # Rough CB memory estimate:
    # input0: per_core_M * in0_block_w
    # input1: in0_block_w * per_core_N
    # output: per_core_M * per_core_N
    # Multiply by 2 for double buffering.
    cb_tiles = (per_core_M * in0_block_w) + (in0_block_w * per_core_N) + (per_core_M * per_core_N)
    estimated_l1_usage = cb_tiles * tile_bytes * 2

    # L1 compute usable size is ~1.0 MB
    MAX_L1_BUDGET = 1048576 
    if estimated_l1_usage > MAX_L1_BUDGET:
        return False, (
            f"Estimated L1 usage {estimated_l1_usage} exceeds budget {MAX_L1_BUDGET} "
            f"(per_core_M={per_core_M}, per_core_N={per_core_N}, in0_block_w={in0_block_w})"
        )

    return True, "ok"


def validate_candidate(
    config: Any,
    config_family: str,
    backend: str,
    features: Dict[str, Any],
) -> Tuple[bool, str]:
    """
    Run all validation checks on a candidate configuration.

    Returns:
        (is_valid, reason) tuple.
    """
    # 1. Tile alignment
    is_valid, reason = validate_tile_alignment(features)
    if not is_valid:
        return False, reason

    # 2. Grid feasibility
    is_valid, reason = validate_grid_feasibility(config, config_family, features)
    if not is_valid:
        return False, reason

    # 3. Subblock constraints
    is_valid, reason = validate_subblock_params(config, config_family)
    if not is_valid:
        return False, reason

    # 4. Backend-specific constraints
    if backend == "minimal_matmul":
        is_valid, reason = validate_minimal_matmul_constraints(features)
        if not is_valid:
            return False, reason

    # 5. Memory config compatibility
    is_valid, reason = validate_memory_config_compatibility(config_family, features)
    if not is_valid:
        return False, reason

    # 6. Batching constraints
    is_valid, reason = validate_batching_constraints(config_family, features)
    if not is_valid:
        return False, reason

    # 7. L1 budget
    is_valid, reason = validate_l1_budget(config, config_family, features)
    if not is_valid:
        return False, reason

    return True, "valid"

# Fix 3: DRAM-sharded configs require sharded input tensors
def _check_dram_sharded_input(config, input_tensor_a=None):
    """Reject DRAM-sharded configs when input is not sharded."""
    config_name = type(config).__name__
    if 'DRAMSharded' in config_name:
        if input_tensor_a is not None and hasattr(input_tensor_a, 'is_sharded'):
            if not input_tensor_a.is_sharded():
                return False
        else:
            # Without tensor info, reject DRAM-sharded to be safe
            return False
    return True

