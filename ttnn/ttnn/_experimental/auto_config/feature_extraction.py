# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Feature extraction for matmul auto-config selection."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from ttnn._experimental.auto_config.math_fidelity import default_fidelity, fidelity_cycle_cost, valid_fidelities

import ttnn

logger = logging.getLogger(__name__)

TILE_HEIGHT = 32
TILE_WIDTH = 32


def _get_attr_value(obj: Any, name: str, default: Any = None) -> Any:
    value = getattr(obj, name, default)
    if callable(value):
        return value()
    return value


def extract_matmul_features(
    input_tensor_a: ttnn.Tensor,
    input_tensor_b: ttnn.Tensor,
    *,
    bias: Optional[ttnn.Tensor] = None,
    transpose_a: bool = False,
    transpose_b: bool = False,
    activation: Optional[str] = None,
    dtype: Optional[ttnn.DataType] = None,
    memory_config: Optional[ttnn.MemoryConfig] = None,
) -> Dict[str, Any]:
    """Extract normalized features from matmul inputs."""
    a_shape = list(input_tensor_a.shape)
    b_shape = list(input_tensor_b.shape)

    if transpose_a:
        a_shape[-2], a_shape[-1] = a_shape[-1], a_shape[-2]
    if transpose_b:
        b_shape[-2], b_shape[-1] = b_shape[-1], b_shape[-2]

    M = a_shape[-2]
    K = a_shape[-1]
    N = b_shape[-1]

    batch_dims_a = a_shape[:-2] if len(a_shape) > 2 else []
    batch_dims_b = b_shape[:-2] if len(b_shape) > 2 else []
    batch_size_a = 1
    for d in batch_dims_a:
        batch_size_a *= d
    batch_size_b = 1
    for d in batch_dims_b:
        batch_size_b *= d

    M_tiles = (M + TILE_HEIGHT - 1) // TILE_HEIGHT
    K_tiles = (K + TILE_WIDTH - 1) // TILE_WIDTH
    N_tiles = (N + TILE_WIDTH - 1) // TILE_WIDTH

    dtype_a = input_tensor_a.dtype
    dtype_b = input_tensor_b.dtype
    layout_a = input_tensor_a.layout
    layout_b = input_tensor_b.layout

    mem_config_a = input_tensor_a.memory_config()
    mem_config_b = input_tensor_b.memory_config()
    is_a_sharded = input_tensor_a.is_sharded()
    is_b_sharded = input_tensor_b.is_sharded()
    mem_layout_a = str(mem_config_a.memory_layout) if is_a_sharded else "INTERLEAVED"
    mem_layout_b = str(mem_config_b.memory_layout) if is_b_sharded else "INTERLEAVED"
    buffer_type_a = str(mem_config_a.buffer_type)
    buffer_type_b = str(mem_config_b.buffer_type)

    device = input_tensor_a.device()
    grid_size = device.compute_with_storage_grid_size()
    num_cores = grid_size.x * grid_size.y

    try:
        arch = str(device.arch())
    except Exception:
        arch = "unknown"

    # L1 memory budget with ~300KB kernel overhead reserve
    KERNEL_OVERHEAD_BYTES = 314_573
    l1_usable_bytes = None
    try:
        l1_total = device.l1_size_per_core()
        l1_usable_bytes = max(0, l1_total - KERNEL_OVERHEAD_BYTES)
    except (AttributeError, Exception):
        pass
    if l1_usable_bytes is None:
        _ARCH_L1_TOTAL = {"grayskull": 1_048_576, "wormhole_b0": 1_572_864}
        arch_lower = arch.lower().replace(" ", "_")
        l1_total = _ARCH_L1_TOTAL.get(arch_lower, 1_572_864)
        l1_usable_bytes = max(0, l1_total - KERNEL_OVERHEAD_BYTES)

    shard_shape_a = None
    shard_orientation_a = None
    if is_a_sharded and input_tensor_a.shard_spec() is not None:
        shard_spec = input_tensor_a.shard_spec()
        shard_shape_a = list(shard_spec.shape)
        shard_orientation_a = str(shard_spec.orientation)

    num_devices = _get_attr_value(device, "get_num_devices", None)
    if num_devices is None:
        num_devices = _get_attr_value(device, "num_devices", 1)
    is_multi_device = num_devices > 1
    mesh_shape = None
    if is_multi_device and hasattr(device, "shape"):
        mesh_shape = list(_get_attr_value(device, "shape"))

    height = batch_size_a * M
    width = N

    return {
        "M": M,
        "K": K,
        "N": N,
        "batch_size_a": batch_size_a,
        "batch_size_b": batch_size_b,
        "M_tiles": M_tiles,
        "K_tiles": K_tiles,
        "N_tiles": N_tiles,
        "height": height,
        "width": width,
        "is_tall": height > width,
        "is_wide": width > height,
        "is_square": height == width,
        "is_batched_b": batch_size_b > 1,
        "dtype_a": str(dtype_a),
        "dtype_b": str(dtype_b),
        "output_dtype": str(dtype) if dtype else str(dtype_a),
        "layout_a": str(layout_a),
        "layout_b": str(layout_b),
        "is_a_sharded": is_a_sharded,
        "is_b_sharded": is_b_sharded,
        "mem_layout_a": mem_layout_a,
        "mem_layout_b": mem_layout_b,
        "buffer_type_a": buffer_type_a,
        "buffer_type_b": buffer_type_b,
        "shard_shape_a": shard_shape_a,
        "shard_orientation_a": shard_orientation_a,
        "arch": arch,
        "grid_x": grid_size.x,
        "grid_y": grid_size.y,
        "num_cores": num_cores,
        "is_multi_device": is_multi_device,
        "num_devices": num_devices,
        "mesh_shape": mesh_shape,
        "transpose_a": transpose_a,
        "transpose_b": transpose_b,
        "has_bias": bias is not None,
        "has_activation": activation is not None,
        "activation": activation,
        "output_memory_config": str(memory_config) if memory_config else "default",
        "l1_usable_bytes": l1_usable_bytes,
        "math_fidelity_default": default_fidelity(str(dtype_a), str(dtype_b)),
        "math_fidelity_valid": valid_fidelities(str(dtype_a), str(dtype_b)),
        "math_fidelity_cycle_cost": fidelity_cycle_cost(default_fidelity(str(dtype_a), str(dtype_b))),
    }


def get_cache_key_from_features(features: Dict[str, Any]) -> str:
    """Generate a deterministic cache key from features."""
    key_parts = [
        f"arch={features.get('arch', 'unknown')}",
        f"M={features['M']}",
        f"K={features['K']}",
        f"N={features['N']}",
        f"Ba={features['batch_size_a']}",
        f"Bb={features['batch_size_b']}",
        f"da={features['dtype_a']}",
        f"db={features['dtype_b']}",
        f"la={features['layout_a']}",
        f"lb={features['layout_b']}",
        f"ma={features['mem_layout_a']}",
        f"mb={features['mem_layout_b']}",
        f"ba={features['buffer_type_a']}",
        f"bb={features['buffer_type_b']}",
        f"gx={features['grid_x']}",
        f"gy={features['grid_y']}",
        f"nd={features['num_devices']}",
        f"ta={features['transpose_a']}",
        f"tb={features['transpose_b']}",
        f"bias={features['has_bias']}",
        f"act={features['activation']}",
        f"mf={features.get('math_fidelity_default', 'unknown')}",
    ]
    if features.get("shard_shape_a"):
        key_parts.append(f"ss={features['shard_shape_a']}")
    return "|".join(key_parts)
