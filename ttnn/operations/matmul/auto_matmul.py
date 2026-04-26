# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os
import json
import hashlib
from typing import Optional, Tuple, Dict, Any, List
import torch

import ttnn
from ttnn.tensor import Tensor, DataType, Layout, MemoryConfig, BufferType
from ttnn.device import Device, MeshDevice
from ttnn.operations.core import tilize, untilize
from ttnn.operations.matmul import (
    MatmulMultiCoreReuseProgramConfig,
    MatmulMultiCoreReuseMultiCastProgramConfig,
    MatmulMultiCoreReuseMultiCast1DProgramConfig,
    MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig,
    MatmulMultiCoreReuseMultiCastBatchedDRAMShardedProgramConfig,
)
from ttnn.operations.matmul import experimental as matmul_experimental
from ttnn.operations.ccl import all_gather, reduce_scatter
from ttnn.types import Shape

_AUTOMATMUL_CACHE_PATH = "/tmp/ttnn_automatmul_cache.json"
_DEFAULT_CONFIG_TIMEOUT_S = 60.0

# Supported matmul backends
class _MatmulBackend:
    TTNN = "ttnn.matmul"
    MINIMAL = "ttnn.experimental.minimal_matmul"

# Supported CCL operations for multi-device
class _CCLPath:
    NONE = "none"
    ALL_GATHER = "all_gather"
    REDUCE_SCATTER = "reduce_scatter"
    FUSED_RS_ASYNC = "fused_reduce_scatter_async"
    FUSED_AG_MINIMAL_ASYNC = "fused_all_gather_minimal_async"

# Configuration candidate
class _MatmulConfigCandidate:
    def __init__(
        self,
        backend: str,
        program_config: Optional[Any] = None,
        ccl_path: str = _CCLPath.NONE,
        fused_activation: Optional[Any] = None,
        dtype: Optional[DataType] = None,
        layout: Optional[Layout] = None,
        memory_config: Optional[MemoryConfig] = None,
    ):
        self.backend = backend
        self.program_config = program_config
        self.ccl_path = ccl_path
        self.fused_activation = fused_activation
        self.dtype = dtype
        self.layout = layout
        self.memory_config = memory_config

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "backend": self.backend,
            "ccl_path": self.ccl_path,
        }
        if self.program_config is not None:
            result["program_config"] = str(self.program_config)
        if self.fused_activation is not None:
            result["fused_activation"] = str(self.fused_activation)
        if self.dtype is not None:
            result["dtype"] = str(self.dtype)
        if self.layout is not None:
            result["layout"] = str(self.layout)
        if self.memory_config is not None:
            result["memory_config"] = {
                "buffer_type": str(self.memory_config.buffer_type),
            }
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "_MatmulConfigCandidate":
        config = cls(
            backend=data["backend"],
            ccl_path=data["ccl_path"],
        )
        # Note: Reconstructing exact program_config objects is complex; we assume they are reconstructable from string repr
        # In practice, we'd store structured config params instead of string reprs
        return config


def _get_signature(
    input_tensor: Tensor,
    weight_tensor: Tensor,
    bias_tensor: Optional[Tensor] = None,
    mesh_device: Optional[MeshDevice] = None,
    dtype: Optional[DataType] = None,
    layout: Optional[Layout] = None,
    memory_config: Optional[MemoryConfig] = None,
    fused_activation: Optional[Any] = None,
    arch: Optional[str] = None,
) -> str:
    """
    Generate a unique signature for the matmul operation based on input characteristics.
    This signature is used to cache performance results.
    """
    if mesh_device is not None:
        if not isinstance(mesh_device, MeshDevice) or len(mesh_device.get_devices()) == 0:
            raise ValueError("Invalid or empty mesh_device provided")

        mesh_shape = mesh_device.get_mesh_shape()
        device_ids = tuple(sorted([dev.id() for dev in mesh_device.get_devices()]))
        is_multidevice = True
    else:
        mesh_shape = (1, 1)
        device_ids = tuple()
        is_multidevice = False

    # Get tensor properties
    def tensor_props(tensor: Tensor) -> Tuple:
        if tensor is None:
            return None
        return (
            list(tensor.shape),
            str(tensor.dtype()),
            str(tensor.layout()),
            str(tensor.memory_config().buffer_type),
            tensor.storage_type(),
        )

    input_props = tensor_props(input_tensor)
    weight_props = tensor_props(weight_tensor)
    bias_props = tensor_props(bias_tensor)

    # Get arch if not provided
    if arch is None and mesh_device is not None:
        first_device = mesh_device.get_devices()[0]
        if hasattr(first_device, "arch"):
            arch = str(first_device.arch)
        else:
            arch = "unknown"

    # Create signature dict
    signature_dict = {
        "input_shape": input_props[0] if input_props else None,
        "input_dtype": input_props[1] if input_props else None,
        "input_layout": input_props[2] if input_props else None,
        "input_buffer_type": input_props[3] if input_props else None,
        "weight_shape": weight_props[0] if weight_props else None,
        "weight_dtype": weight_props[1] if weight_props else None,
        "weight_layout": weight_props[2] if weight_props else None,
        "weight_buffer_type": weight_props[3] if weight_props else None,
        "bias_shape": bias_props[0] if bias_props else None,
        "bias_dtype": bias_props[1] if bias_props else None,
        "bias_layout": bias_props[2] if bias_props else None,
        "bias_buffer_type": bias_props[3] if bias_props else None,
        "mesh_shape": mesh_shape,
        "device_ids": device_ids,
        "is_multidevice": is_multidevice,
        "user_dtype": str(dtype) if dtype else None,
        "user_layout": str(layout) if layout else None,
        "user_buffer_type": str(memory_config.buffer_type) if memory_config else None,
        "fused_activation": str(fused_activation) if fused_activation else None,
        "arch": arch,
    }

    # Convert to JSON and hash
    signature_json = json.dumps(signature_dict, sort_keys=True)
    return hashlib.sha256(signature_json.encode("utf-8")).hexdigest()


def _load_cache() -> Dict[str, Dict]:
    """Load the automata cache from disk."""
    if not os.path.exists(_AUTOMATMUL_CACHE_PATH):
        return {}
    try:
        with open(_AUTOMATMUL_CACHE_PATH, "r") as f:
            data = json.load(f)
            # Convert list-based program configs back to objects if needed
            return data
    except (json.JSONDecodeError, IOError):
        return {}


def _save_cache(cache: Dict[str, Dict]) -> None:
    """Save the automata cache to disk."""
    try:
        with open(_AUTOMATMUL_CACHE_PATH, "w") as f:
            json.dump(cache, f, indent=2)
    except IOError:
        # Best effort - don't fail the operation if cache can't be saved
        pass


def _is_valid_config_for_inputs(
    candidate: _MatmulConfigCandidate,
    input_tensor: Tensor,
    weight_tensor: Tensor,
    mesh_device: Optional[MeshDevice] = None,
) -> bool:
    """Check if the candidate configuration is valid for the given inputs."""
    # Check minimal_matmul constraints
    if candidate.backend == _MatmulBackend.MINIMAL:
        # minimal_matmul has specific constraints
        if weight_tensor.layout() != ttnn.TILE_LAYOUT:
            return False
        if input_tensor.layout() != ttnn.TILE_LAYOUT:
            return False
        if mesh_device is not None:
            # For multi-device, check if fused async path is available
            if candidate.ccl_path == _CCLPath.FUSED_RS_ASYNC:
                if not hasattr(matmul_experimental, "matmul_reduce_scatter_async"):
                    return False
            elif candidate.ccl_path == _CCLPath.FUSED_AG_MINIMAL_ASYNC:
                if not hasattr(matmul_experimental, "strided_all_gather_minimal_matmul_async"):
                    return False

    # Check program config compatibility
    if candidate.program_config is not None:
        pc = candidate.program_config
        if isinstance(
            pc,
            (
                MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig,
                MatmulMultiCoreReuseMultiCastBatchedDRAMShardedProgramConfig,
            ),
        ):
            if weight_tensor.memory_config().buffer_type != BufferType.DRAM:
                return False

    return True


def _generate_candidates(
    input_tensor: Tensor,
    weight_tensor: Tensor,
    bias_tensor: Optional[Tensor] = None,
    mesh_device: Optional[MeshDevice] = None,
    fused_activation: Optional[Any] = None,
) -> List[_MatmulConfigCandidate]:
    """Generate all valid configuration candidates for the given inputs."""
    candidates = []

    # Determine if we can use minimal_matmul
    can_use_minimal = True
    if weight_tensor.layout() != ttnn.TILE_LAYOUT:
        can_use_minimal = False
    if input_tensor.layout() != ttnn.TILE_LAYOUT:
        can_use_minimal = False
    if bias_tensor is not None:
        can_use_minimal = False
    if fused_activation is not None:
        can_use_minimal = False

    # Add ttnn.matmul candidates with various program configs
    base_configs = [
        MatmulMultiCoreReuseProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            in0_block_w=1,
            out_subblock_h=1,
            out_subblock_w=1,
            per_core_M=1,
            per_core_N=1,
        ),
        MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            in0_block_w=1,
            out_subblock_h=1,
            out_subblock_w=1,
            per_core_M=1,
            per_core_N=1,
        ),
        MatmulMultiCoreReuseMultiCast1DProgramConfig(
            shard_grid=(1, 8),
            in0_block_w=1,
            per_core_M=1,
            per_core_N=1,
            fuse_batch=True,
        ),
    ]

    # Add DRAM-sharded configs if weight is in DRAM
    if weight_tensor.memory_config().buffer_type == BufferType.DRAM:
        dram_configs = [
            MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                in0_halo=False,
                compute_with_storage_grid_size=(8, 8),
            ),
            MatmulMultiCoreReuseMultiCastBatchedDRAMShardedProgramConfig(
                in0_halo=False,
                num_cores_x=8,
            ),
        ]
        base_configs.extend(dram_configs)

    for pc in base_configs:
        candidate = _MatmulConfigCandidate(
            backend=_MatmulBackend.TTNN,
            program_config=pc,
            fused_activation=fused_activation,
            dtype=weight_tensor.dtype(),
            layout=weight_tensor.layout(),
            memory_config=weight_tensor.memory_config(),
        )
        if _is_valid_config_for_inputs(candidate, input_tensor, weight_tensor, mesh_device):
            candidates.append(candidate)

    # Add minimal_matmul candidate if valid
    if can_use_minimal:
        candidate = _MatmulConfigCandidate(
            backend=_MatmulBackend.MINIMAL,
            fused_activation=None,
            dtype=weight_tensor.dtype(),
            layout=weight_tensor.layout(),
            memory_config=weight_tensor.memory_config(),
        )
        if _is_valid_config_for_inputs(candidate, input_tensor, weight_tensor, mesh_device):
            candidates.append(candidate)

    # Handle multi-device cases
    if mesh_device is not None:
        new_candidates = []
        for cand in candidates:
            # Add all_gather path
            ag_candidate = _MatmulConfigCandidate(**cand.__dict__)
            ag_candidate.ccl_path = _CCLPath.ALL_GATHER
            if _is_valid_config_for_inputs(ag_candidate, input_tensor, weight_tensor, mesh_device):
                new_candidates.append(ag_candidate)

            # Add reduce_scatter path
            rs_candidate = _MatmulConfigCandidate(**cand.__dict__)
            rs_candidate.ccl_path = _CCLPath.REDUCE_SCATTER
            if _is_valid_config_for_inputs(rs_candidate, input_tensor, weight_tensor, mesh_device):
                new_candidates.append(rs_candidate)

            # Add fused async paths if available
            if cand.backend == _MatmulBackend.MINIMAL:
                if hasattr(matmul_experimental, "matmul_reduce_scatter_async"):
                    fused_rs_candidate = _MatmulConfigCandidate(**cand.__dict__)
                    fused_rs_candidate.ccl_path = _CCLPath.FUSED_RS_ASYNC
                    if _is_valid_config_for_inputs(fused_rs_candidate, input_tensor, weight_tensor, mesh_device):
                        new_candidates.append(fused_rs_candidate)

                if hasattr(matmul_experimental, "strided_all_gather_minimal_matmul_async"):
                    fused_ag_candidate = _MatmulConfigCandidate(**cand.__dict__)
                    fused_ag_candidate.ccl_path = _CCLPath.FUSED_AG_MINIMAL_ASYNC
                    if _is_valid_config_for_inputs(fused_ag_candidate, input_tensor, weight_tensor, mesh_device):
                        new_candidates.append(fused_ag_candidate)

        candidates.extend(new_candidates)

    return candidates


def _benchmark_candidate(
    input_tensor: Tensor,
    weight_tensor: Tensor,
    bias_tensor: Optional[Tensor],
    candidate: _MatmulConfigCandidate,
    mesh_device: Optional[MeshDevice] = None,
    num_runs: int = 5,
) -> float:
    """Benchmark a candidate configuration and return average execution time in seconds."""
    device = mesh_device if mesh_device is not None else input_tensor.device()

    # Warmup
    for _ in range(2):
        _execute_matmul(input_tensor, weight_tensor, bias_tensor, candidate, mesh_device)

    # Benchmark
    start_event = ttnn.create_event(device)
    end_event = ttnn.create_event(device)
    times = []

    for _ in range(num_runs):
        ttnn.record_event(device, start_event)
        output = _execute_matmul(input_tensor, weight_tensor, bias_tensor, candidate, mesh_device)
        ttnn.record_event(device, end_event)
        ttnn.synchronize_devices(device)
        elapsed = ttnn.get_event_elapsed_time(start_event, end_event)
        times.append(elapsed)

    # Clean up
    if "output" in locals():
        del output

    return sum(times) / len(times)


def _execute_matmul(
    input_tensor: Tensor,
    weight_tensor: Tensor,
    bias_tensor: Optional[Tensor],
    config: _MatmulConfigCandidate,
    mesh_device: Optional[MeshDevice] = None,
) -> Tensor:
    """Execute matmul with the given configuration."""
    device = mesh_device if mesh_device is not None else input_tensor.device()

    # Apply CCL if needed
    if config.ccl_path == _CCLPath.ALL_GATHER:
        input_tensor = all_gather(input_tensor, dim=3, num_links=1, mesh_device=mesh_device)
    elif config.ccl_path == _CCLPath.REDUCE_SCATTER:
        # For reduce_scatter, we assume input is sharded and we're reducing the result
        # This is a simplification - actual sharding strategy depends on use case
        pass  # Will be handled in fused path or separately

    # Execute matmul
    if config.backend == _MatmulBackend.TTNN:
        output = ttnn.matmul(
            input_tensor,
            weight_tensor,
            bias=bias_tensor,
            program_config=config.program_config,
            dtype=config.dtype if config.dtype else weight_tensor.dtype(),
            layout=config.layout if config.layout else weight_tensor.layout(),
            memory_config=config.memory_config if config.memory_config else weight_tensor.memory_config(),
            fused_activation=config.fused_activation,
        )
    elif config.backend == _MatmulBackend.MINIMAL:
        if config.ccl_path == _CCLPath.FUSED_RS_ASYNC and hasattr(matmul_experimental, "matmul_reduce_scatter_async"):
            output = matmul_experimental.matmul_reduce_scatter_async(
                input_tensor, weight_tensor, num_links=1, mesh_device=mesh_device
            )
        elif (
            config.ccl_path == _CCLPath.FUSED_AG_MINIMAL_ASYNC
            and hasattr(matmul_experimental, "strided_all_gather_minimal_matmul_async")
        ):
            output = matmul_experimental.strided_all_gather_minimal_matmul_async(
                input_tensor, weight_tensor, num_links=1, mesh_device=mesh_device
            )
        else:
            output = matmul_experimental.minimal_matmul(input_tensor, weight_tensor)
    else:
        raise ValueError(f"Unknown backend: {config.backend}")

    # Apply reduce_scatter if needed
    if config.ccl_path == _CCLPath.REDUCE_SCATTER:
        output = reduce_scatter(output, reduce_op=ttnn.ReduceType.Sum, num_links=1, mesh_device=mesh_device)

    return output


def _suggest_optimal_tensor_properties(
    input_tensor: Tensor,
    weight_tensor: Tensor,
) -> Dict[str, Any]:
    """Suggest optimal tensor properties for better performance."""
    suggestions = {}

    # Suggest TILE layout if not already used
    if input_tensor.layout() != ttnn.TILE_LAYOUT:
        suggestions["input_layout"] = ttnn.TILE_LAYOUT
    if weight_tensor.layout() != ttnn.TILE_LAYOUT:
        suggestions["weight_layout"] = ttnn.TILE_LAYOUT

    # Suggest DRAM storage for large weights
    weight_size_mb = weight_tensor.volume() * (2 if weight_tensor.dtype() == ttnn.bfloat16 else 4) / (1024 * 1024)
    if weight_size_mb > 10 and weight_tensor.memory_config().buffer_type != BufferType.DRAM:
        suggestions["weight_memory_config"] = MemoryConfig(buffer_type=BufferType.DRAM)

    return suggestions


def automatmul(
    input_tensor: Tensor,
    weight_tensor: Tensor,
    bias_tensor: Optional[Tensor] = None,
    *,
    mesh_device: Optional[MeshDevice] = None,
    dtype: Optional[DataType] = None,
    layout: Optional[Layout] = None,
    memory_config: Optional[MemoryConfig] = None,
    fused_activation: Optional[Any] = None,
    force_recompute: bool = False,
) -> Tensor:
    """
    Automated matmul that selects the most optimal configuration based on input characteristics.
    This API mirrors torch.matmul in