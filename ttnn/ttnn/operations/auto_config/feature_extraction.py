# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Feature extraction for matmul auto-config selection.

Extracts a normalized feature dictionary from input tensors and operation
parameters. This feature dict is used for cache keying, candidate generation,
and performance scoring.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import ttnn

logger = logging.getLogger(__name__)

# Standard tile dimensions
TILE_HEIGHT = 32
TILE_WIDTH = 32


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
    """
    Extract normalized features from matmul inputs.

    Returns a dict with all features needed for config selection and cache keying.
    """
    # Get shapes
    a_shape = list(input_tensor_a.shape)
    b_shape = list(input_tensor_b.shape)

    # Handle transpose
    if transpose_a:
        a_shape[-2], a_shape[-1] = a_shape[-1], a_shape[-2]
    if transpose_b:
        b_shape[-2], b_shape[-1] = b_shape[-1], b_shape[-2]

    # Extract M, K, N
    M = a_shape[-2]
    K = a_shape[-1]
    N = b_shape[-1]

    # Batch dimensions
    batch_dims_a = a_shape[:-2] if len(a_shape) > 2 else []
    batch_dims_b = b_shape[:-2] if len(b_shape) > 2 else []
    batch_size_a = 1
    for d in batch_dims_a:
        batch_size_a *= d
    batch_size_b = 1
    for d in batch_dims_b:
        batch_size_b *= d

    # Tile dimensions
    M_tiles = (M + TILE_HEIGHT - 1) // TILE_HEIGHT
    K_tiles = (K + TILE_WIDTH - 1) // TILE_WIDTH
    N_tiles = (N + TILE_WIDTH - 1) // TILE_WIDTH

    # Input metadata
    dtype_a = input_tensor_a.dtype
    dtype_b = input_tensor_b.dtype
    layout_a = input_tensor_a.layout
    layout_b = input_tensor_b.layout

    # Memory config
    mem_config_a = input_tensor_a.memory_config()
    mem_config_b = input_tensor_b.memory_config()

    is_a_sharded = input_tensor_a.is_sharded()
    is_b_sharded = input_tensor_b.is_sharded()

    # Memory layout type
    mem_layout_a = str(mem_config_a.memory_layout) if is_a_sharded else "INTERLEAVED"
    mem_layout_b = str(mem_config_b.memory_layout) if is_b_sharded else "INTERLEAVED"

    # Buffer type
    buffer_type_a = str(mem_config_a.buffer_type)
    buffer_type_b = str(mem_config_b.buffer_type)

    # Device info
    device = input_tensor_a.device()
    grid_size = device.compute_with_storage_grid_size()
    num_cores = grid_size.x * grid_size.y

    # Architecture info — critical for cache key to prevent cross-hardware collisions
    try:
        arch = str(device.arch())
    except Exception:
        arch = "unknown"

    # Shard info
    shard_shape_a = None
    shard_orientation_a = None
    if is_a_sharded and input_tensor_a.shard_spec() is not None:
        shard_spec = input_tensor_a.shard_spec()
        shard_shape_a = list(shard_spec.shape)
        shard_orientation_a = str(shard_spec.orientation)

    # Multi-device detection
    is_multi_device = hasattr(device, "num_devices") and device.num_devices() > 1
    num_devices = device.num_devices() if is_multi_device else 1
    mesh_shape = None
    if is_multi_device and hasattr(device, "shape"):
        mesh_shape = list(device.shape())

    # Aspect ratios for shape classification
    height = batch_size_a * M
    width = N

    features = {
        # Shape dimensions
        "M": M,
        "K": K,
        "N": N,
        "batch_size_a": batch_size_a,
        "batch_size_b": batch_size_b,
        "M_tiles": M_tiles,
        "K_tiles": K_tiles,
        "N_tiles": N_tiles,
        # Shape classification
        "height": height,
        "width": width,
        "is_tall": height > width,
        "is_wide": width > height,
        "is_square": height == width,
        "is_batched_b": batch_size_b > 1,
        # Dtypes
        "dtype_a": str(dtype_a),
        "dtype_b": str(dtype_b),
        "output_dtype": str(dtype) if dtype else str(dtype_a),
        # Layouts
        "layout_a": str(layout_a),
        "layout_b": str(layout_b),
        # Memory
        "is_a_sharded": is_a_sharded,
        "is_b_sharded": is_b_sharded,
        "mem_layout_a": mem_layout_a,
        "mem_layout_b": mem_layout_b,
        "buffer_type_a": buffer_type_a,
        "buffer_type_b": buffer_type_b,
        "shard_shape_a": shard_shape_a,
        "shard_orientation_a": shard_orientation_a,
        # Device
        "arch": arch,
        "grid_x": grid_size.x,
        "grid_y": grid_size.y,
        "num_cores": num_cores,
        "is_multi_device": is_multi_device,
        "num_devices": num_devices,
        "mesh_shape": mesh_shape,
        # Operation params
        "transpose_a": transpose_a,
        "transpose_b": transpose_b,
        "has_bias": bias is not None,
        "has_activation": activation is not None,
        "activation": activation,
        # Output config
        "output_memory_config": str(memory_config) if memory_config else "default",
    }

    return features


def get_cache_key_from_features(features: Dict[str, Any]) -> str:
    """
    Generate a deterministic cache key from features.

    The key includes all parameters that affect config selection.
    """
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
    ]
    if features.get("shard_shape_a"):
        key_parts.append(f"ss={features['shard_shape_a']}")
    return "|".join(key_parts)
