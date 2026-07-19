# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import dataclasses
import hashlib
import inspect
import io
import importlib
import importlib.metadata as importlib_metadata
import json
import math
import os
import pathlib
import time
from datetime import datetime, timezone
from typing import Any, Callable


_DEFAULT_CACHE_DIR = pathlib.Path.home() / ".cache" / "ttnn" / "auto_matmul"
_L1_BUDGET_KB = 1400
_MN_BLOCK_MIN = 2
_MN_BLOCK_MAX = 16
_K_BLOCK_MIN = 2
_BENCHMARK_ITERS = 3
# Maximum number of minimal-matmul candidates to benchmark per signature.  The
# full cartesian product of block-size candidates can exceed 100 entries for
# large shapes (e.g. 784×192×576 → 174 entries), which would add tens of
# minutes to a CI run.  We sample evenly from the sorted candidate list to
# keep the sweep bounded while still covering diverse block-size regimes.
_MAX_MINIMAL_CANDIDATES = 4
# Maximum number of classic MatmulProgramConfig candidates (2D mcast, 1D mcast
# in0/in1) benchmarked per signature.  Each candidate is a distinct matmul
# execution mode; invalid ones for a given shape self-eliminate during the
# on-device benchmark loop, so we generate broadly and let measured latency pick.
_MAX_PROGRAM_CONFIG_CANDIDATES = 6
# Maximum relative L2 error a tuned candidate's output may have versus the
# trusted reference (default-config) output to be eligible for selection.
# Different but valid configs differ only by bf16 rounding (well under this);
# a config that is invalid for the shape (yet still runs) is off by O(1) and is
# rejected here.  This guards selection so we never pick a faster-but-wrong
# config.  Size-independent, unlike correlation which is degenerate for the tiny
# logical outputs many matmuls produce.
_CORRECTNESS_REL_ERR_THRESHOLD = 0.05
_CACHE_FILE_SUFFIX = ".json"
_DEFAULT_CCL_CHUNKS_PER_SYNC = 10
_DEFAULT_CCL_NUM_WORKERS_PER_LINK = 2
_DEFAULT_CCL_NUM_BUFFERS_PER_CHANNEL = 2
_DEFAULT_CACHE_VERSION_FALLBACK = "unknown"


def _ttnn():
    return importlib.import_module("ttnn")


def _get_cpp_base_operation(is_linear: bool) -> Any:
    matmul_module = importlib.import_module("ttnn.operations.matmul")
    return matmul_module._CPP_LINEAR if is_linear else matmul_module._CPP_MATMUL


def _load_tt_dit_matmul_helpers() -> Any | None:
    try:
        return importlib.import_module("models.tt_dit.utils.matmul")
    except Exception:
        return None


def _shape_to_tuple(shape: Any) -> tuple[int, ...]:
    if shape is None:
        return ()
    return tuple(int(dim) for dim in shape)


def _stringify(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _normalize_dim(dim: int, rank: int) -> int:
    return dim if dim >= 0 else rank + dim


def _serialize_memory_config(memory_config: Any) -> str | None:
    if memory_config is None:
        return None
    return str(memory_config)


def _serialize_topology(tensor: Any) -> dict[str, Any] | None:
    if not hasattr(tensor, "tensor_topology"):
        return None
    try:
        topology = tensor.tensor_topology()
        placements = list(topology.placements())
        rank = len(_shape_to_tuple(getattr(tensor, "shape", ())))
        return {
            "placements": [str(placement) for placement in placements],
            "placement_kinds": ["shard" if hasattr(placement, "dim") else "replicate" for placement in placements],
            "shard_dims": [int(placement.dim) if hasattr(placement, "dim") else None for placement in placements],
            "normalized_shard_dims": [
                _normalize_dim(int(placement.dim), rank) if hasattr(placement, "dim") else None
                for placement in placements
            ],
            "distribution_shape": [int(dim) for dim in topology.distribution_shape()],
        }
    except Exception:
        return None


def _serialize_tile(tensor: Any) -> dict[str, Any] | None:
    try:
        tile = tensor.get_tile()
    except Exception:
        tile = getattr(tensor, "tile", None)
    if tile is None:
        return None

    tile_shape = getattr(tile, "tile_shape", None)
    if tile_shape is None:
        return None

    return {
        "tile_shape": [int(tile_shape[0]), int(tile_shape[1])],
        "transpose_of_faces": _stringify(getattr(tile, "transpose_of_faces", None)),
    }


def _serialize_tensor(tensor: Any) -> dict[str, Any] | None:
    if tensor is None:
        return None
    memory_config = None
    if hasattr(tensor, "memory_config"):
        try:
            memory_config = tensor.memory_config()
        except Exception:
            memory_config = None
    return {
        "shape": list(_shape_to_tuple(getattr(tensor, "shape", ()))),
        "dtype": _stringify(getattr(tensor, "dtype", None)),
        "layout": _stringify(getattr(tensor, "layout", None)),
        "tile": _serialize_tile(tensor),
        "memory_config": _serialize_memory_config(memory_config),
        "topology": _serialize_topology(tensor),
    }


def _supports_matrix_auto_config(lhs_shape: tuple[int, ...], rhs_shape: tuple[int, ...]) -> bool:
    return len(lhs_shape) >= 2 and len(rhs_shape) >= 2


def _matmul_dims_compatible(
    lhs_shape: tuple[int, ...],
    rhs_shape: tuple[int, ...],
    transpose_a: bool,
    transpose_b: bool,
) -> bool:
    """Return True only if the contraction (K) dimensions line up.

    Auto-config cannot build a program config for a matmul the base op would
    reject.  When the shapes are incompatible we defer to the base op so it can
    raise its own native error, preserving matmul's error contract.
    """
    if not _supports_matrix_auto_config(lhs_shape, rhs_shape):
        return False
    lhs_inner = lhs_shape[-2] if transpose_a else lhs_shape[-1]
    rhs_inner = rhs_shape[-1] if transpose_b else rhs_shape[-2]
    return lhs_inner == rhs_inner


def _extract_mkn(
    lhs_shape: tuple[int, ...],
    rhs_shape: tuple[int, ...],
    transpose_a: bool,
    transpose_b: bool,
) -> tuple[int, int, int]:
    if not _supports_matrix_auto_config(lhs_shape, rhs_shape):
        raise ValueError("auto-config matmul requires tensors with at least 2 dimensions")

    lhs_rows = lhs_shape[-1] if transpose_a else lhs_shape[-2]
    lhs_inner = lhs_shape[-2] if transpose_a else lhs_shape[-1]
    rhs_inner = rhs_shape[-1] if transpose_b else rhs_shape[-2]
    rhs_cols = rhs_shape[-2] if transpose_b else rhs_shape[-1]

    if lhs_inner != rhs_inner:
        raise ValueError(f"Incompatible matmul dimensions: lhs K={lhs_inner}, rhs K={rhs_inner}")

    output_shape = _extract_output_shape(lhs_shape, rhs_shape, transpose_a, transpose_b)
    m = math.prod(output_shape[:-1]) if output_shape[:-1] else lhs_rows
    return int(m), int(lhs_inner), int(rhs_cols)


def _extract_output_shape(
    lhs_shape: tuple[int, ...],
    rhs_shape: tuple[int, ...],
    transpose_a: bool,
    transpose_b: bool,
) -> tuple[int, ...]:
    lhs_row_dim = lhs_shape[-1] if transpose_a else lhs_shape[-2]
    rhs_col_dim = rhs_shape[-2] if transpose_b else rhs_shape[-1]
    lhs_batch = lhs_shape[:-2]
    rhs_batch = rhs_shape[:-2]
    max_rank = max(len(lhs_batch), len(rhs_batch))
    lhs_batch = (1,) * (max_rank - len(lhs_batch)) + lhs_batch
    rhs_batch = (1,) * (max_rank - len(rhs_batch)) + rhs_batch
    batch_shape = []
    for lhs_dim, rhs_dim in zip(lhs_batch, rhs_batch):
        if lhs_dim == 1:
            batch_shape.append(rhs_dim)
        elif rhs_dim == 1 or lhs_dim == rhs_dim:
            batch_shape.append(lhs_dim)
        else:
            raise ValueError(f"Incompatible batch dimensions for matmul: {lhs_shape} vs {rhs_shape}")
    return tuple(batch_shape) + (int(lhs_row_dim), int(rhs_col_dim))


def _broadcast_shape(lhs_shape: tuple[int, ...], rhs_shape: tuple[int, ...]) -> tuple[int, ...]:
    max_rank = max(len(lhs_shape), len(rhs_shape))
    lhs_shape = (1,) * (max_rank - len(lhs_shape)) + lhs_shape
    rhs_shape = (1,) * (max_rank - len(rhs_shape)) + rhs_shape
    output_shape = []
    for lhs_dim, rhs_dim in zip(lhs_shape, rhs_shape):
        if lhs_dim == 1:
            output_shape.append(rhs_dim)
        elif rhs_dim == 1 or lhs_dim == rhs_dim:
            output_shape.append(lhs_dim)
        else:
            raise ValueError(f"Incompatible broadcast dimensions: {lhs_shape} vs {rhs_shape}")
    return tuple(int(dim) for dim in output_shape)


def _minimal_matmul_preserves_linear_bias_shape(signature: "AutoMatmulSignature") -> bool:
    if not signature.is_linear or signature.bias is None:
        return True

    lhs_shape = tuple(int(dim) for dim in signature.input_tensor_a.get("shape", ()))
    rhs_shape = tuple(int(dim) for dim in signature.input_tensor_b.get("shape", ()))
    bias_shape = tuple(int(dim) for dim in signature.bias.get("shape", ()))
    try:
        matmul_shape = _extract_output_shape(lhs_shape, rhs_shape, signature.transpose_a, signature.transpose_b)
        linear_shape = _broadcast_shape(matmul_shape, bias_shape)
    except (IndexError, TypeError, ValueError):
        return False
    return linear_shape == matmul_shape


def _uses_standard_tile_shape(tensor_descriptor: dict[str, Any] | None) -> bool:
    if not tensor_descriptor:
        return True
    tile_descriptor = tensor_descriptor.get("tile")
    if not tile_descriptor:
        return True
    return tuple(int(dim) for dim in tile_descriptor.get("tile_shape", (32, 32))) == (32, 32)


def _is_distributed(signature: "AutoMatmulSignature") -> bool:
    for descriptor in (
        signature.input_tensor_a.get("topology"),
        signature.input_tensor_b.get("topology"),
        signature.bias.get("topology") if signature.bias else None,
    ):
        if descriptor and any(int(dim) > 1 for dim in descriptor.get("distribution_shape", ())):
            return True
    return False


def _topology_distribution_factor(topology: dict[str, Any] | None, axis: int | None) -> int:
    if topology is None or axis is None:
        return 1
    distribution_shape = topology.get("distribution_shape", ())
    if axis < 0 or axis >= len(distribution_shape):
        return 1
    return max(1, int(distribution_shape[axis]))


def _topology_shard_axis_for_dim(topology: dict[str, Any] | None, normalized_dim: int) -> int | None:
    if topology is None:
        return None
    for axis, shard_dim in enumerate(topology.get("normalized_shard_dims", ())):
        if shard_dim == normalized_dim and _topology_distribution_factor(topology, axis) > 1:
            return axis
    return None


@dataclasses.dataclass(frozen=True)
class AutoMatmulSignature:
    arch: str
    device_count: int
    mesh_shape: tuple[int, ...]
    is_linear: bool
    transpose_a: bool
    transpose_b: bool
    activation: str | None
    output_memory_config: str | None
    output_dtype: str | None
    input_tensor_a: dict[str, Any]
    input_tensor_b: dict[str, Any]
    bias: dict[str, Any] | None
    m: int
    k: int
    n: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "arch": self.arch,
            "device_count": self.device_count,
            "mesh_shape": list(self.mesh_shape),
            "is_linear": self.is_linear,
            "transpose_a": self.transpose_a,
            "transpose_b": self.transpose_b,
            "activation": self.activation,
            "output_memory_config": self.output_memory_config,
            "output_dtype": self.output_dtype,
            "input_tensor_a": self.input_tensor_a,
            "input_tensor_b": self.input_tensor_b,
            "bias": self.bias,
            "m": self.m,
            "k": self.k,
            "n": self.n,
        }

    @property
    def cache_key(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclasses.dataclass
class PreparedMatmulInputs:
    input_tensor_a: Any
    input_tensor_b: Any
    bias: Any = None
    staged_rhs_from_host: bool = False
    staged_bias_from_host: bool = False


@dataclasses.dataclass
class Candidate:
    descriptor: dict[str, Any]
    run: Callable[[], Any]


@dataclasses.dataclass(frozen=True)
class DistributedCollectivePlan:
    kind: str
    collective_dim: int | None = None
    cluster_axis: int | None = None
    distribution_factor: int = 1
    lhs_shard_dim: int | None = None
    rhs_shard_dim: int | None = None


@dataclasses.dataclass
class WeightPlacement:
    """Result of ``place_weight``: a device-placed weight plus the distribution
    decision the module made and the measured evidence behind it.

    ``tensor`` is the placed ``ttnn.Tensor`` ready to hand to ``ttnn.matmul`` /
    ``ttnn.linear``.  ``strategy`` is the placement that was carried out
    (``"replicate"`` or ``"shard_k"``).  ``output_is_sharded`` is True when the
    chosen placement produces a reduce-scattered (row-parallel) output rather
    than a full output on every device, so the caller knows the result layout.
    ``verified`` records whether the chosen placement was checked against a torch
    ground-truth reference on-device."""

    tensor: Any
    strategy: str
    shard_dim: int | None = None
    cluster_axis: int | None = None
    mesh_shape: tuple[int, ...] = ()
    output_is_sharded: bool = False
    verified: bool = False
    candidate_timings: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    recommendations: list[str] = dataclasses.field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "shard_dim": self.shard_dim,
            "cluster_axis": self.cluster_axis,
            "mesh_shape": list(self.mesh_shape),
            "output_is_sharded": self.output_is_sharded,
            "verified": self.verified,
            "candidate_timings": list(self.candidate_timings),
            "recommendations": list(self.recommendations),
        }


def _get_default_version() -> str:
    override = os.environ.get("TTNN_AUTO_MATMUL_VERSION")
    if override:
        return override

    try:
        return importlib.import_module("ttnn.model_preprocessing").git_hash()
    except Exception:
        for env_var in ("GITHUB_SHA", "CI_COMMIT_SHA", "BUILD_SOURCEVERSION", "GIT_REF"):
            env_value = os.environ.get(env_var)
            if env_value:
                return env_value

        try:
            return f"ttnn-{importlib_metadata.version('ttnn')}"
        except Exception:
            return _DEFAULT_CACHE_VERSION_FALLBACK


class AutoMatmulCache:
    _runtime_records: dict[tuple[str, str], dict[str, Any]] = {}

    def __init__(self) -> None:
        cache_dir = pathlib.Path(os.environ.get("TTNN_AUTO_MATMUL_CACHE_DIR", _DEFAULT_CACHE_DIR))
        self.root_dir = cache_dir
        self.version = _get_default_version()

    @property
    def version_dir(self) -> pathlib.Path:
        return self.root_dir / self.version

    def path_for(self, signature: AutoMatmulSignature) -> pathlib.Path:
        return self.version_dir / f"{signature.cache_key}{_CACHE_FILE_SUFFIX}"

    def _runtime_key_for(self, signature: AutoMatmulSignature) -> tuple[str, str]:
        return (self.version, signature.cache_key)

    def load(self, signature: AutoMatmulSignature) -> dict[str, Any] | None:
        if os.environ.get("TTNN_AUTO_MATMUL_FORCE_RETUNE") == "1":
            return None
        path = self.path_for(signature)
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("version") != self.version:
            return None
        return payload

    def save(self, signature: AutoMatmulSignature, payload: dict[str, Any]) -> pathlib.Path:
        path = self.path_for(signature)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        return path

    def invalidate(self, signature: AutoMatmulSignature) -> None:
        path = self.path_for(signature)
        if path.exists():
            path.unlink()

    def load_runtime(self, signature: AutoMatmulSignature) -> dict[str, Any] | None:
        return self._runtime_records.get(self._runtime_key_for(signature))

    def save_runtime(self, signature: AutoMatmulSignature, payload: dict[str, Any]) -> None:
        self._runtime_records[self._runtime_key_for(signature)] = payload

    def invalidate_runtime(self, signature: AutoMatmulSignature) -> None:
        self._runtime_records.pop(self._runtime_key_for(signature), None)

    @classmethod
    def clear_runtime(cls) -> None:
        cls._runtime_records.clear()


def _get_local_num_devices(mesh_device: Any) -> int:
    if mesh_device is None:
        raise ValueError("mesh_device is required to determine CCL link counts")
    local_device_ids = mesh_device.get_device_ids()
    if not local_device_ids:
        raise ValueError("CCL link detection requires at least one host-local device")
    return len(local_device_ids)


def _determine_device_name(mesh_device: Any) -> str:
    ttnn = _ttnn()
    num_devices = _get_local_num_devices(mesh_device)
    arch_name = ttnn.get_arch_name()
    dram_grid_size = mesh_device.dram_grid_size()

    if "blackhole" in arch_name:
        dict_device_names = {
            1: "P100" if dram_grid_size and dram_grid_size.x == 7 else "P150",
            2: "P300",
            4: "P150x4",
            8: "P150x8",
            32: "BHGLX",
        }
    elif "wormhole_b0" in arch_name:
        dict_device_names = {
            1: "N150",
            2: "N300",
            4: "N150x4",
            8: "T3K",
            32: "TG",
        }
    else:
        raise ValueError(f"Unsupported architecture: {arch_name}")

    if num_devices not in dict_device_names:
        raise ValueError(f"Unsupported number of local devices: {num_devices} for {arch_name}")
    return dict_device_names[num_devices]


def _get_num_links(mesh_device: Any, cluster_axis: int | None = None) -> int:
    device_name = _determine_device_name(mesh_device)
    link_dict = {
        "P100": (0, 0),
        "P150": (0, 0),
        "N150": (0, 0),
        "N300": (1, 1),
        "T3K": (1, 1),
        "P150x4": (2, 2),
        "P150x8": (2, 2),
        "P300": (2, 2),
        "BHGLX": (2, 2),
        "TG": (4, 4),
        "N150x4": (1, 1),
    }
    device_links = link_dict[device_name]
    if cluster_axis is None:
        return min(device_links)
    if cluster_axis in (0, 1):
        return device_links[cluster_axis]
    raise ValueError(f"Unsupported cluster_axis: {cluster_axis}")


def _default_topology(mesh_device: Any) -> Any | None:
    ttnn = _ttnn()
    num_devices = mesh_device.get_num_devices()
    try:
        cluster_type = ttnn.cluster.get_cluster_type()
    except Exception:
        cluster_type = None
    if num_devices == 8 and cluster_type in [ttnn.cluster.ClusterType.T3K, ttnn.cluster.ClusterType.GALAXY]:
        return ttnn.Topology.Ring
    if num_devices > 1:
        return ttnn.Topology.Linear
    return None


class AutoMatmulCCLCache:
    """Core-owned semaphore cache for measured multi-device recipes."""

    _instances: dict[int, "AutoMatmulCCLCache"] = {}

    def __init__(self, mesh_device: Any) -> None:
        ttnn = _ttnn()
        self.mesh_device = mesh_device
        grid = mesh_device.compute_with_storage_grid_size()
        self.core_range = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))}
        )

        self.barrier_semaphore_idx = [0, 0, 0]
        self.barrier_semaphore_handles = [[], [], []]
        self.ag_semaphores_idx = [0, 0, 0]
        self.ag_semaphore_handles = [[], [], []]
        self.rs_semaphores_idx = [0, 0, 0]
        self.rs_semaphore_handles = [[], [], []]

        for axis_index in range(3):
            for _ in range(2):
                self.barrier_semaphore_handles[axis_index].append(
                    ttnn.create_global_semaphore(mesh_device, self.core_range, 0)
                )
                self.ag_semaphore_handles[axis_index].append(
                    [ttnn.create_global_semaphore(mesh_device, self.core_range, 0) for _ in range(2)]
                )
                self.rs_semaphore_handles[axis_index].append(
                    [ttnn.create_global_semaphore(mesh_device, self.core_range, 0) for _ in range(3)]
                )

    @classmethod
    def get(cls, mesh_device: Any) -> "AutoMatmulCCLCache":
        mesh_id = mesh_device.id()
        if mesh_id not in cls._instances:
            cls._instances[mesh_id] = cls(mesh_device)
        return cls._instances[mesh_id]

    @staticmethod
    def _axis_index(cluster_axis: int | None) -> int:
        return 2 if cluster_axis is None else cluster_axis

    def get_barrier(self, cluster_axis: int | None = None) -> Any:
        axis_index = self._axis_index(cluster_axis)
        current_idx = self.barrier_semaphore_idx[axis_index]
        self.barrier_semaphore_idx[axis_index] = (current_idx + 1) % 2
        return self.barrier_semaphore_handles[axis_index][current_idx]

    def get_all_gather(self, cluster_axis: int | None = None) -> list[Any]:
        axis_index = self._axis_index(cluster_axis)
        current_idx = self.ag_semaphores_idx[axis_index]
        self.ag_semaphores_idx[axis_index] = (current_idx + 1) % 2
        return self.ag_semaphore_handles[axis_index][current_idx]

    def get_reduce_scatter(self, cluster_axis: int | None = None) -> list[Any]:
        axis_index = self._axis_index(cluster_axis)
        current_idx = self.rs_semaphores_idx[axis_index]
        self.rs_semaphores_idx[axis_index] = (current_idx + 1) % 2
        return self.rs_semaphore_handles[axis_index][current_idx]


def _get_mesh_mapper_for_host_staging(device: Any) -> Any | None:
    ttnn = _ttnn()
    if getattr(device, "get_num_devices", lambda: 1)() > 1:
        return ttnn.ReplicateTensorToMesh(device)
    return None


def _hash_host_tensor_contents(tensor: Any) -> str:
    try:
        import torch
    except Exception:
        torch = None

    if torch is not None and torch.is_tensor(tensor):
        serialized = io.BytesIO()
        torch.save(tensor.detach().cpu().contiguous(), serialized)
        return hashlib.sha256(serialized.getbuffer()).hexdigest()
    return hashlib.sha256(repr(tensor).encode("utf-8")).hexdigest()


def _host_tensor_cache_name(role: str, tensor: Any, dtype: Any) -> pathlib.Path:
    shape = list(getattr(tensor, "shape", ()))
    dtype_name = _stringify(dtype) or str(getattr(tensor, "dtype", "unknown"))
    content_digest = _hash_host_tensor_contents(tensor)[:24]
    digest = hashlib.sha256(f"{role}:{shape}:{dtype_name}:{content_digest}".encode("utf-8")).hexdigest()[:16]
    cache_dir = pathlib.Path(os.environ.get("TTNN_AUTO_MATMUL_CACHE_DIR", _DEFAULT_CACHE_DIR)) / "host_tensors"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{role}_{digest}.bin"


def _stage_host_tensor(role: str, source_tensor: Any, reference_tensor: Any) -> Any:
    ttnn = _ttnn()
    layout = ttnn.TILE_LAYOUT
    target_dtype = getattr(reference_tensor, "dtype", None)
    device = reference_tensor.device()
    mesh_mapper = _get_mesh_mapper_for_host_staging(device)
    return ttnn.as_tensor(
        source_tensor,
        dtype=target_dtype,
        layout=layout,
        device=device,
        mesh_mapper=mesh_mapper,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        cache_file_name=_host_tensor_cache_name(role, source_tensor, target_dtype),
    )


def _prepare_inputs(input_tensor_a: Any, input_tensor_b: Any, bias: Any | None) -> PreparedMatmulInputs:
    ttnn = _ttnn()
    prepared = PreparedMatmulInputs(input_tensor_a=input_tensor_a, input_tensor_b=input_tensor_b, bias=bias)

    if input_tensor_a is None or not isinstance(input_tensor_a, ttnn.Tensor):
        return prepared

    try:
        import torch
    except Exception:
        torch = None

    if torch is not None and isinstance(input_tensor_b, torch.Tensor):
        prepared.input_tensor_b = _stage_host_tensor("rhs", input_tensor_b, input_tensor_a)
        prepared.staged_rhs_from_host = True

    if bias is not None and torch is not None and isinstance(bias, torch.Tensor):
        prepared.bias = _stage_host_tensor("bias", bias, input_tensor_a)
        prepared.staged_bias_from_host = True

    return prepared


def _build_signature(
    input_tensor_a: Any,
    input_tensor_b: Any,
    *,
    bias: Any | None,
    transpose_a: bool,
    transpose_b: bool,
    memory_config: Any,
    dtype: Any,
    activation: Any,
    is_linear: bool,
) -> AutoMatmulSignature:
    ttnn = _ttnn()
    lhs_shape = _shape_to_tuple(getattr(input_tensor_a, "shape", ()))
    rhs_shape = _shape_to_tuple(getattr(input_tensor_b, "shape", ()))
    m, k, n = _extract_mkn(lhs_shape, rhs_shape, transpose_a, transpose_b)
    device = input_tensor_a.device()
    device_count = getattr(device, "get_num_devices", lambda: 1)()
    mesh_shape_obj = getattr(device, "shape", (1,))
    if hasattr(mesh_shape_obj, "dims"):
        mesh_shape = tuple(int(mesh_shape_obj[i]) for i in range(mesh_shape_obj.dims()))
    else:
        mesh_shape = tuple(int(dim) for dim in mesh_shape_obj)
    return AutoMatmulSignature(
        arch=ttnn.get_arch_name(),
        device_count=int(device_count),
        mesh_shape=mesh_shape,
        is_linear=is_linear,
        transpose_a=transpose_a,
        transpose_b=transpose_b,
        activation=_stringify(activation),
        output_memory_config=_serialize_memory_config(memory_config),
        output_dtype=_stringify(dtype),
        input_tensor_a=_serialize_tensor(input_tensor_a) or {},
        input_tensor_b=_serialize_tensor(input_tensor_b) or {},
        bias=_serialize_tensor(bias),
        m=m,
        k=k,
        n=n,
    )


def _has_explicit_override(kwargs: dict[str, Any]) -> bool:
    for key in (
        "program_config",
        "core_grid",
        "global_cb",
        "optional_output_tensor",
        "sub_device_id",
        "compute_kernel_config",
        "output_tile",
    ):
        if kwargs.get(key) is not None:
            return True
    return False


def _deallocate_result(result: Any) -> None:
    if result is None:
        return
    if isinstance(result, (list, tuple)):
        for item in result:
            _deallocate_result(item)
        return
    if hasattr(result, "deallocate"):
        try:
            result.deallocate(True)
        except Exception:
            pass


def _sync_device(device: Any) -> None:
    if device is None:
        return
    ttnn = _ttnn()
    try:
        ttnn.synchronize_device(device)
    except Exception:
        pass


def _can_use_minimal_matmul_common(signature: AutoMatmulSignature, bias: Any | None) -> bool:
    ttnn = _ttnn()
    if signature.transpose_a or signature.transpose_b:
        return False
    if signature.activation is not None:
        # minimal_matmul does not support fused activations. Compound activations
        # such as SILU can hang the device in minimal_matmul on certain block
        # configurations. Fall back to default_matmul/default_linear, which apply
        # the activation correctly via the full C++ kernel path.
        return False
    if not _minimal_matmul_preserves_linear_bias_shape(signature):
        return False
    if signature.input_tensor_a.get("layout") != str(ttnn.TILE_LAYOUT):
        return False
    if signature.input_tensor_b.get("layout") != str(ttnn.TILE_LAYOUT):
        return False
    if not _uses_standard_tile_shape(signature.input_tensor_a):
        return False
    if not _uses_standard_tile_shape(signature.input_tensor_b):
        return False
    if not _uses_standard_tile_shape(signature.bias):
        return False
    if bias is not None and (_serialize_tensor(bias) or {}).get("layout") != str(ttnn.TILE_LAYOUT):
        return False
    if len(signature.input_tensor_b.get("shape", ())) < 2:
        return False
    if any(int(dim) != 1 for dim in signature.input_tensor_b["shape"][:-2]):
        return False
    return True


def _compute_tile_counts(m: int, k: int, n: int) -> tuple[int, int, int]:
    return max(1, math.ceil(m / 32)), max(1, math.ceil(k / 32)), max(1, math.ceil(n / 32))


def _get_mn_block_candidates(per_core_tiles: int) -> list[int]:
    if per_core_tiles <= 1:
        return [1]
    evens = set(range(_MN_BLOCK_MIN, _MN_BLOCK_MAX + 1, 2))
    divisors = {candidate for candidate in range(_MN_BLOCK_MIN, _MN_BLOCK_MAX + 1) if per_core_tiles % candidate == 0}
    return sorted(evens | divisors)


def _get_k_block_candidates(k_tiles: int) -> list[int]:
    if k_tiles <= 1:
        return [1]
    return sorted(candidate for candidate in range(_K_BLOCK_MIN, k_tiles + 1) if k_tiles % candidate == 0)


def _estimate_l1_kb(m_block: int, k_block: int, n_block: int, bias: Any | None) -> int:
    bf16_kb = 2
    f32_kb = 4
    total = (
        2 * m_block * k_block * bf16_kb
        + 2 * k_block * n_block * bf16_kb
        + 2 * m_block * n_block * bf16_kb
        + m_block * n_block * f32_kb
    )
    if bias is not None:
        total += n_block * bf16_kb
    return total


def _pick_subblock(m_block: int, n_block: int) -> tuple[int, int]:
    if m_block % 2 == 0 and n_block % 2 == 0:
        return (2, 2)
    best = (1, 1)
    best_area = 1
    for sub_h in range(1, min(4, m_block) + 1):
        if m_block % sub_h != 0:
            continue
        for sub_w in range(1, min(4, n_block) + 1):
            if n_block % sub_w != 0:
                continue
            area = sub_h * sub_w
            if area <= 4 and area > best_area:
                best = (sub_h, sub_w)
                best_area = area
    return best


def _make_minimal_descriptor(
    *,
    grid: Any,
    m_block: int,
    k_block: int,
    n_block: int,
    subblock_h: int,
    subblock_w: int,
) -> dict[str, Any]:
    return {
        "M_block_size": m_block,
        "K_block_size": k_block,
        "N_block_size": n_block,
        "subblock_h": subblock_h,
        "subblock_w": subblock_w,
        "grid": [int(grid.x), int(grid.y)],
    }


def _build_minimal_descriptors(
    signature: AutoMatmulSignature,
    *,
    bias: Any | None,
    grid: Any,
) -> list[dict[str, Any]]:
    m_tiles, k_tiles, n_tiles = _compute_tile_counts(signature.m, signature.k, signature.n)
    per_core_m = max(1, math.ceil(m_tiles / max(1, grid.x)))
    per_core_n = max(1, math.ceil(n_tiles / max(1, grid.y)))

    descriptors: list[dict[str, Any]] = []
    for m_block in _get_mn_block_candidates(per_core_m):
        for k_block in _get_k_block_candidates(k_tiles):
            for n_block in _get_mn_block_candidates(per_core_n):
                if _estimate_l1_kb(m_block, k_block, n_block, bias) > _L1_BUDGET_KB:
                    continue
                subblock_h, subblock_w = _pick_subblock(m_block, n_block)
                descriptors.append(
                    _make_minimal_descriptor(
                        grid=grid,
                        m_block=m_block,
                        k_block=k_block,
                        n_block=n_block,
                        subblock_h=subblock_h,
                        subblock_w=subblock_w,
                    )
                )
    return descriptors


def _make_minimal_config(config_descriptor: dict[str, Any]) -> Any:
    ttnn = _ttnn()
    return ttnn.MinimalMatmulConfig(
        M_block_size=config_descriptor["M_block_size"],
        K_block_size=config_descriptor["K_block_size"],
        N_block_size=config_descriptor["N_block_size"],
        subblock_h=config_descriptor["subblock_h"],
        subblock_w=config_descriptor["subblock_w"],
        compute_with_storage_grid_size=ttnn.CoreCoord(config_descriptor["grid"][0], config_descriptor["grid"][1]),
    )


def _run_minimal_matmul(
    signature: AutoMatmulSignature,
    prepared: PreparedMatmulInputs,
    kwargs: dict[str, Any],
    *,
    config_descriptor: dict[str, Any],
) -> Any:
    ttnn = _ttnn()
    call_kwargs = {
        "input_tensor": prepared.input_tensor_a,
        "weight_tensor": prepared.input_tensor_b,
        "config": _make_minimal_config(config_descriptor),
        "fused_activation": kwargs.get("activation"),
        "memory_config": kwargs.get("memory_config"),
        "dtype": kwargs.get("dtype"),
        "compute_kernel_config": kwargs.get("compute_kernel_config"),
    }
    if prepared.bias is not None:
        call_kwargs["bias_tensor"] = prepared.bias
    return ttnn.experimental.minimal_matmul(**call_kwargs)


def _get_ccl_runtime(mesh_device: Any, cluster_axis: int | None) -> dict[str, Any]:
    topology = _default_topology(mesh_device)
    try:
        num_links = _get_num_links(mesh_device, cluster_axis)
    except Exception:
        num_links = 1
    return {
        "topology": topology,
        "num_links": max(1, int(num_links)),
        "num_workers_per_link": _DEFAULT_CCL_NUM_WORKERS_PER_LINK,
        "num_buffers_per_channel": _DEFAULT_CCL_NUM_BUFFERS_PER_CHANNEL,
        "chunks_per_sync": _DEFAULT_CCL_CHUNKS_PER_SYNC,
    }


def _get_default_compute_kernel_config(device: Any, provided: Any) -> Any:
    if provided is not None:
        return provided
    ttnn = _ttnn()
    try:
        return ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
    except Exception:
        return None


def _high_accuracy_compute_kernel_config(device: Any) -> Any:
    """Maximum-accuracy matmul compute config: HiFi3 + fp32 accumulation.

    Used for single-tile matmuls where perf is irrelevant so the result matches
    the fp32/torch reference bit-for-bit (HiFi2 is not enough -- it can leave a
    ~1 ULP residual that trips sub-ULP numeric-check thresholds).  HiFi3 (not
    HiFi4) is used with fp32 accumulation per the documented Wormhole guidance
    that HiFi4+fp32 can be less accurate there due to a hardware bug."""
    ttnn = _ttnn()
    try:
        return ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi3,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
    except Exception:
        return None


def _infer_distributed_plan(signature: AutoMatmulSignature) -> DistributedCollectivePlan:
    if not _is_distributed(signature):
        return DistributedCollectivePlan(kind="none")

    lhs_shape = tuple(signature.input_tensor_a.get("shape", ()))
    rhs_shape = tuple(signature.input_tensor_b.get("shape", ()))
    lhs_topology = signature.input_tensor_a.get("topology")
    rhs_topology = signature.input_tensor_b.get("topology")

    lhs_k_dim = len(lhs_shape) - 2 if signature.transpose_a else len(lhs_shape) - 1
    rhs_k_dim = len(rhs_shape) - 1 if signature.transpose_b else len(rhs_shape) - 2
    lhs_shard_axis = _topology_shard_axis_for_dim(lhs_topology, lhs_k_dim)
    rhs_shard_axis = _topology_shard_axis_for_dim(rhs_topology, rhs_k_dim)

    # Both operands sharded along their contraction dim on the same mesh axis:
    # device i holds X[..., k_i] and W[k_i, ...], so the local matmul yields a
    # partial sum that must be reduce-scattered across devices (row-parallel /
    # tensor-parallel).  This is detected before the lhs-only all-gather case,
    # which would otherwise gather X to full K and multiply it against a
    # K-sharded W — a silent shape/semantic error.  This is the topology
    # ``place_weight`` produces when it shards a host weight on K to match a
    # K-sharded activation.
    if lhs_shard_axis is not None and rhs_shard_axis is not None and lhs_shard_axis == rhs_shard_axis:
        output_rank = len(_extract_output_shape(lhs_shape, rhs_shape, signature.transpose_a, signature.transpose_b))
        return DistributedCollectivePlan(
            kind="matmul_before_reduce_scatter",
            collective_dim=output_rank - 1,
            cluster_axis=rhs_shard_axis,
            distribution_factor=_topology_distribution_factor(rhs_topology, rhs_shard_axis),
            lhs_shard_dim=lhs_k_dim,
            rhs_shard_dim=rhs_k_dim,
        )

    if lhs_shard_axis is not None:
        return DistributedCollectivePlan(
            kind="gather_before_matmul",
            collective_dim=lhs_k_dim,
            cluster_axis=lhs_shard_axis,
            distribution_factor=_topology_distribution_factor(lhs_topology, lhs_shard_axis),
            lhs_shard_dim=lhs_k_dim,
            rhs_shard_dim=None,
        )

    if rhs_shard_axis is not None:
        output_rank = len(_extract_output_shape(lhs_shape, rhs_shape, signature.transpose_a, signature.transpose_b))
        return DistributedCollectivePlan(
            kind="matmul_before_reduce_scatter",
            collective_dim=output_rank - 1,
            cluster_axis=rhs_shard_axis,
            distribution_factor=_topology_distribution_factor(rhs_topology, rhs_shard_axis),
            lhs_shard_dim=None,
            rhs_shard_dim=rhs_k_dim,
        )

    return DistributedCollectivePlan(kind="unsupported")


def _build_local_minimal_candidates(
    signature: AutoMatmulSignature,
    prepared: PreparedMatmulInputs,
    kwargs: dict[str, Any],
) -> list[Candidate]:
    if not _can_use_minimal_matmul_common(signature, prepared.bias):
        return []
    if _is_distributed(signature):
        return []

    device = prepared.input_tensor_a.device()
    if device is None:
        return []
    grid = device.compute_with_storage_grid_size()
    descriptors = _build_minimal_descriptors(signature, bias=prepared.bias, grid=grid)
    if len(descriptors) > _MAX_MINIMAL_CANDIDATES:
        step = max(1, len(descriptors) // _MAX_MINIMAL_CANDIDATES)
        descriptors = descriptors[::step][:_MAX_MINIMAL_CANDIDATES]
    return [
        Candidate(
            descriptor={"kind": "minimal_matmul", **descriptor},
            run=lambda descriptor=descriptor: _run_minimal_matmul(
                signature, prepared, kwargs, config_descriptor=descriptor
            ),
        )
        for descriptor in descriptors
    ]


def _find_largest_divisor(value: int, max_divisor: int = 8) -> int:
    for divisor in range(max_divisor, 0, -1):
        if value % divisor == 0:
            return divisor
    return 1


def _pick_out_subblock_w(per_core_n: int, out_subblock_h: int) -> int:
    # out_subblock_h * out_subblock_w must fit the compute engine (<= 4 here, the
    # conservative bound used across the model configs) and evenly divide per_core_N.
    out_subblock_w = 4
    while out_subblock_w > 1:
        if out_subblock_w * out_subblock_h <= 4 and per_core_n % out_subblock_w == 0:
            break
        out_subblock_w -= 1
    return out_subblock_w


def _pick_out_subblock_h(per_core_m: int, out_subblock_w: int) -> int:
    return max([i for i in range(1, 4 + 1) if per_core_m % i == 0 and i * out_subblock_w <= 4] or [1])


def _supports_program_config_sweep(signature: AutoMatmulSignature) -> bool:
    """Eligibility for the classic 2D/1D mcast program-config sweep.

    Restricted to the plain (non-transposed, non-batched) M×K · K×N case in TILE
    layout — the regime the 2D/1D multicast factories are parameterized for here.
    Transposed/batched shapes are left to the default heuristic candidate.
    """
    ttnn = _ttnn()
    if signature.transpose_a or signature.transpose_b:
        return False
    if signature.input_tensor_a.get("layout") != str(ttnn.TILE_LAYOUT):
        return False
    if signature.input_tensor_b.get("layout") != str(ttnn.TILE_LAYOUT):
        return False
    if not _uses_standard_tile_shape(signature.input_tensor_a):
        return False
    if not _uses_standard_tile_shape(signature.input_tensor_b):
        return False
    lhs_shape = signature.input_tensor_a.get("shape", ())
    rhs_shape = signature.input_tensor_b.get("shape", ())
    if len(lhs_shape) < 2 or len(rhs_shape) < 2:
        return False
    # No batch dims on either operand (leading dims must collapse to 1).
    if any(int(dim) != 1 for dim in lhs_shape[:-2]):
        return False
    if any(int(dim) != 1 for dim in rhs_shape[:-2]):
        return False
    # Single-output-tile matmuls (both M and N fit in one 32x32 tile) cannot be
    # distributed across the core grid, so the 2D/1D multicast configs are
    # degenerate: they add rounding error without any parallelism benefit.  Leave
    # these to the default heuristic / minimal-matmul families.
    m_tiles, _, n_tiles = _compute_tile_counts(signature.m, signature.k, signature.n)
    if m_tiles <= 1 and n_tiles <= 1:
        return False
    return True


def _make_program_config(descriptor: dict[str, Any]) -> Any:
    ttnn = _ttnn()
    grid = (int(descriptor["grid"][0]), int(descriptor["grid"][1]))
    if descriptor["mode"] == "2d_mcast":
        return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=grid,
            in0_block_w=descriptor["in0_block_w"],
            out_subblock_h=descriptor["out_subblock_h"],
            out_subblock_w=descriptor["out_subblock_w"],
            per_core_M=descriptor["per_core_M"],
            per_core_N=descriptor["per_core_N"],
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=True,
        )
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=grid,
        in0_block_w=descriptor["in0_block_w"],
        out_subblock_h=descriptor["out_subblock_h"],
        out_subblock_w=descriptor["out_subblock_w"],
        per_core_M=descriptor["per_core_M"],
        per_core_N=descriptor["per_core_N"],
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=(descriptor["mode"] == "1d_mcast_in0"),
    )


def _run_program_config_matmul(
    base_operation: Any,
    signature: AutoMatmulSignature,
    prepared: PreparedMatmulInputs,
    kwargs: dict[str, Any],
    *,
    descriptor: dict[str, Any],
) -> Any:
    # Activation (if any) flows through kwargs so the C++ op fuses it; we leave
    # fused_activation=None on the program config to avoid double application.
    call_kwargs = dict(kwargs)
    call_kwargs["program_config"] = _make_program_config(descriptor)
    if signature.is_linear:
        call_kwargs["bias"] = prepared.bias
    return base_operation(prepared.input_tensor_a, prepared.input_tensor_b, **call_kwargs)


def _build_program_config_candidates(
    signature: AutoMatmulSignature,
    prepared: PreparedMatmulInputs,
    kwargs: dict[str, Any],
    *,
    base_operation: Any,
) -> list[Candidate]:
    """Generate candidates spanning the distinct ttnn.matmul execution modes.

    ttnn.matmul dispatches to several MatmulProgramConfig variants (2D systolic
    multicast, 1D multicast gathering in0 or in1, ...).  Rather than trusting a
    single heuristic, this builds a representative config for each mode from the
    M/K/N tile counts and the device grid, then lets the on-device benchmark loop
    measure them and keep the fastest.  Configs that are invalid for the shape
    throw during benchmarking and are discarded there.
    """
    if _is_distributed(signature):
        return []
    if not _supports_program_config_sweep(signature):
        return []
    device = prepared.input_tensor_a.device()
    if device is None:
        return []

    grid = device.compute_with_storage_grid_size()
    grid_x, grid_y = int(grid.x), int(grid.y)
    num_cores = max(1, grid_x * grid_y)
    m_tiles, k_tiles, n_tiles = _compute_tile_counts(signature.m, signature.k, signature.n)

    descriptors: list[dict[str, Any]] = []

    # 2D systolic multicast: M distributed over grid rows, N over grid columns.
    per_core_m_2d = max(1, math.ceil(m_tiles / grid_y))
    per_core_n_2d = max(1, math.ceil(n_tiles / grid_x))
    subblock_w_2d = _pick_out_subblock_w(per_core_n_2d, 1)
    descriptors.append(
        {
            "kind": "program_config",
            "mode": "2d_mcast",
            "grid": [grid_x, grid_y],
            "in0_block_w": _find_largest_divisor(max(1, k_tiles // grid_y)),
            "per_core_M": per_core_m_2d,
            "per_core_N": per_core_n_2d,
            "out_subblock_h": 1,
            "out_subblock_w": subblock_w_2d,
        }
    )

    in0_block_w_1d = _find_largest_divisor(max(1, k_tiles // num_cores))

    # 1D multicast gathering in0: all cores split N (best when M is small).
    per_core_m_in0 = m_tiles
    per_core_n_in0 = max(1, math.ceil(n_tiles / num_cores))
    subblock_w_in0 = _pick_out_subblock_w(per_core_n_in0, 1)
    descriptors.append(
        {
            "kind": "program_config",
            "mode": "1d_mcast_in0",
            "grid": [grid_x, grid_y],
            "in0_block_w": in0_block_w_1d,
            "per_core_M": per_core_m_in0,
            "per_core_N": per_core_n_in0,
            "out_subblock_h": _pick_out_subblock_h(per_core_m_in0, subblock_w_in0),
            "out_subblock_w": subblock_w_in0,
        }
    )

    # 1D multicast gathering in1: all cores split M (best when N is small).
    per_core_m_in1 = max(1, math.ceil(m_tiles / num_cores))
    per_core_n_in1 = n_tiles
    subblock_w_in1 = _pick_out_subblock_w(per_core_n_in1, 1)
    descriptors.append(
        {
            "kind": "program_config",
            "mode": "1d_mcast_in1",
            "grid": [grid_x, grid_y],
            "in0_block_w": in0_block_w_1d,
            "per_core_M": per_core_m_in1,
            "per_core_N": per_core_n_in1,
            "out_subblock_h": _pick_out_subblock_h(per_core_m_in1, subblock_w_in1),
            "out_subblock_w": subblock_w_in1,
        }
    )

    # Drop configs whose per-core working set clearly exceeds L1, de-duplicate,
    # and cap the sweep so CI stays bounded.
    seen: set[str] = set()
    candidates: list[Candidate] = []
    for descriptor in descriptors:
        if (
            _estimate_l1_kb(
                descriptor["per_core_M"], descriptor["in0_block_w"], descriptor["per_core_N"], prepared.bias
            )
            > _L1_BUDGET_KB
        ):
            continue
        key = json.dumps(descriptor, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(
            Candidate(
                descriptor=descriptor,
                run=lambda descriptor=descriptor: _run_program_config_matmul(
                    base_operation, signature, prepared, kwargs, descriptor=descriptor
                ),
            )
        )
        if len(candidates) >= _MAX_PROGRAM_CONFIG_CANDIDATES:
            break
    return candidates


def _is_single_output_tile(signature: AutoMatmulSignature) -> bool:
    m_tiles, _, n_tiles = _compute_tile_counts(signature.m, signature.k, signature.n)
    return m_tiles <= 1 and n_tiles <= 1


def _default_candidate_kwargs(
    signature: AutoMatmulSignature,
    prepared: PreparedMatmulInputs,
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Effective kwargs for the default candidate.

    For single-output-tile matmuls, run the default op with the maximum-accuracy
    compute config (HiFi3 + fp32 accumulation) when the caller hasn't pinned one:
    tuning is skipped at this size, and the lower-precision stock default can
    drift ~1 ULP from the reference, tripping sub-ULP numeric-check thresholds.
    Applied identically here and in cache reconstruction so a cached winner
    rebuilds the same op.  Larger shapes are returned unchanged."""
    if not _is_single_output_tile(signature):
        return kwargs
    if kwargs.get("compute_kernel_config") is not None:
        return kwargs
    device = prepared.input_tensor_a.device()
    accurate_cc = _high_accuracy_compute_kernel_config(device)
    if accurate_cc is None:
        return kwargs
    adjusted = dict(kwargs)
    adjusted["compute_kernel_config"] = accurate_cc
    return adjusted


def _build_default_candidate(
    *,
    base_operation: Any,
    prepared: PreparedMatmulInputs,
    kwargs: dict[str, Any],
    signature: AutoMatmulSignature,
) -> Candidate:
    descriptor = {"kind": "default_linear" if signature.is_linear else "default_matmul"}

    def _runner() -> Any:
        call_kwargs = dict(kwargs)
        if signature.is_linear:
            call_kwargs["bias"] = prepared.bias
        return base_operation(prepared.input_tensor_a, prepared.input_tensor_b, **call_kwargs)

    return Candidate(descriptor=descriptor, run=_runner)


def _make_ag_offset_candidates(grid: Any) -> list[list[int]]:
    offsets: list[list[int]] = []
    for y_coord in {max(0, int(grid.y) - 2), max(0, int(grid.y) // 2), 4}:
        offsets.append([0, y_coord])
    deduped: list[list[int]] = []
    seen = set()
    for offset in offsets:
        key = tuple(offset)
        if key not in seen:
            seen.add(key)
            deduped.append(offset)
    return deduped


def _build_all_gather_then_matmul_candidate(
    signature: AutoMatmulSignature,
    prepared: PreparedMatmulInputs,
    kwargs: dict[str, Any],
    plan: DistributedCollectivePlan,
    *,
    base_operation: Any,
) -> Candidate:
    descriptor = {
        "kind": "all_gather_then_linear" if signature.is_linear else "all_gather_then_matmul",
        "collective_dim": plan.collective_dim,
        "cluster_axis": plan.cluster_axis,
    }

    def _runner() -> Any:
        ttnn = _ttnn()
        mesh_device = prepared.input_tensor_a.device()
        ccl_cache = AutoMatmulCCLCache.get(mesh_device)
        ccl_runtime = _get_ccl_runtime(mesh_device, plan.cluster_axis)
        gathered = ttnn.experimental.all_gather_async(
            prepared.input_tensor_a,
            persistent_output_buffer=None,
            dim=plan.collective_dim,
            multi_device_global_semaphore=ccl_cache.get_all_gather(plan.cluster_axis),
            num_links=ccl_runtime["num_links"],
            memory_config=None,
            topology=ccl_runtime["topology"],
            subdevice_id=kwargs.get("sub_device_id"),
            cluster_axis=plan.cluster_axis,
            use_optimal_ccl_for_llama=False,
            barrier_semaphore=ccl_cache.get_barrier(plan.cluster_axis),
            use_broadcast=False,
            chunks_per_sync=ccl_runtime["chunks_per_sync"],
            num_workers_per_link=ccl_runtime["num_workers_per_link"],
            num_buffers_per_channel=ccl_runtime["num_buffers_per_channel"],
            sub_core_grids=None,
        )
        try:
            call_kwargs = dict(kwargs)
            if signature.is_linear:
                call_kwargs["bias"] = prepared.bias
            return base_operation(gathered, prepared.input_tensor_b, **call_kwargs)
        finally:
            _deallocate_result(gathered)

    return Candidate(descriptor=descriptor, run=_runner)


def _build_all_gather_matmul_candidates(
    signature: AutoMatmulSignature,
    prepared: PreparedMatmulInputs,
    kwargs: dict[str, Any],
    plan: DistributedCollectivePlan,
) -> list[Candidate]:
    if signature.transpose_a or signature.transpose_b:
        return []

    device = prepared.input_tensor_a.device()
    if device is None:
        return []

    candidates: list[Candidate] = []
    for offset in _make_ag_offset_candidates(device.compute_with_storage_grid_size()):
        descriptor = {
            "kind": "all_gather_matmul_async",
            "collective_dim": plan.collective_dim,
            "cluster_axis": plan.cluster_axis,
            "all_gather_core_grid_offset": offset,
        }

        def _runner(offset=offset) -> Any:
            ttnn = _ttnn()
            mesh_device = prepared.input_tensor_a.device()
            ccl_cache = AutoMatmulCCLCache.get(mesh_device)
            ccl_runtime = _get_ccl_runtime(mesh_device, plan.cluster_axis)
            _, result = ttnn.experimental.all_gather_matmul_async(
                prepared.input_tensor_a,
                prepared.input_tensor_b,
                None,
                plan.collective_dim,
                ccl_cache.get_all_gather(plan.cluster_axis),
                ttnn.CoreCoord(offset[0], offset[1]),
                bias=prepared.bias,
                num_links=ccl_runtime["num_links"],
                memory_config_ag=None,
                topology=ccl_runtime["topology"],
                barrier_semaphore=ccl_cache.get_barrier(plan.cluster_axis),
                subdevice_id=kwargs.get("sub_device_id"),
                memory_config_mm=kwargs.get("memory_config"),
                transpose_a=False,
                transpose_b=False,
                dtype=kwargs.get("dtype"),
                program_config=None,
                activation=kwargs.get("activation"),
                compute_kernel_config=kwargs.get("compute_kernel_config"),
                core_grid=kwargs.get("core_grid"),
                chunks_per_sync=ccl_runtime["chunks_per_sync"],
                num_workers_per_link=ccl_runtime["num_workers_per_link"],
                num_buffers_per_channel=ccl_runtime["num_buffers_per_channel"],
            )
            return result

        candidates.append(Candidate(descriptor=descriptor, run=_runner))
    return candidates


def _build_all_gather_minimal_candidates(
    signature: AutoMatmulSignature,
    prepared: PreparedMatmulInputs,
    kwargs: dict[str, Any],
    plan: DistributedCollectivePlan,
) -> list[Candidate]:
    if not _can_use_minimal_matmul_common(signature, prepared.bias):
        return []

    device = prepared.input_tensor_a.device()
    if device is None:
        return []
    full_grid = device.compute_with_storage_grid_size()
    trimmed_grid = _ttnn().CoreCoord(full_grid.x, max(1, full_grid.y - 1))
    descriptors = _build_minimal_descriptors(signature, bias=prepared.bias, grid=trimmed_grid)
    candidates: list[Candidate] = []
    for descriptor in descriptors:
        candidate_descriptor = {
            "kind": "all_gather_minimal_matmul_async",
            "collective_dim": plan.collective_dim,
            "cluster_axis": plan.cluster_axis,
            **descriptor,
        }

        def _runner(descriptor=descriptor) -> Any:
            ttnn = _ttnn()
            mesh_device = prepared.input_tensor_a.device()
            ccl_cache = AutoMatmulCCLCache.get(mesh_device)
            ccl_runtime = _get_ccl_runtime(mesh_device, plan.cluster_axis)
            outputs = ttnn.experimental.all_gather_minimal_matmul_async(
                prepared.input_tensor_a,
                prepared.input_tensor_b,
                bias_tensor=prepared.bias,
                fused_activation=kwargs.get("activation"),
                config=_make_minimal_config(descriptor),
                multi_device_global_semaphore=ccl_cache.get_all_gather(plan.cluster_axis),
                topology=ccl_runtime["topology"],
                memory_config=kwargs.get("memory_config"),
                dtype=kwargs.get("dtype"),
                compute_kernel_config=kwargs.get("compute_kernel_config"),
                persistent_output_buffer=None,
                num_links=ccl_runtime["num_links"],
                cluster_axis=plan.cluster_axis,
                barrier_semaphore=ccl_cache.get_barrier(plan.cluster_axis),
                force_transpose=signature.m >= signature.n,
                num_workers_per_link=ccl_runtime["num_workers_per_link"],
                num_buffers_per_channel=ccl_runtime["num_buffers_per_channel"],
            )
            if isinstance(outputs, (list, tuple)):
                return outputs[0]
            return outputs

        candidates.append(Candidate(descriptor=candidate_descriptor, run=_runner))
    return candidates


def _build_all_gather_then_minimal_candidates(
    signature: AutoMatmulSignature,
    prepared: PreparedMatmulInputs,
    kwargs: dict[str, Any],
    plan: DistributedCollectivePlan,
) -> list[Candidate]:
    if not _can_use_minimal_matmul_common(signature, prepared.bias):
        return []

    device = prepared.input_tensor_a.device()
    if device is None:
        return []
    full_grid = device.compute_with_storage_grid_size()
    trimmed_grid = _ttnn().CoreCoord(full_grid.x, max(1, full_grid.y - 1))
    descriptors = _build_minimal_descriptors(signature, bias=prepared.bias, grid=trimmed_grid)
    candidates: list[Candidate] = []
    for descriptor in descriptors:
        candidate_descriptor = {
            "kind": "all_gather_then_minimal_matmul",
            "collective_dim": plan.collective_dim,
            "cluster_axis": plan.cluster_axis,
            **descriptor,
        }

        def _runner(descriptor=descriptor) -> Any:
            ttnn = _ttnn()
            mesh_device = prepared.input_tensor_a.device()
            ccl_cache = AutoMatmulCCLCache.get(mesh_device)
            ccl_runtime = _get_ccl_runtime(mesh_device, plan.cluster_axis)
            gathered = ttnn.experimental.all_gather_async(
                prepared.input_tensor_a,
                persistent_output_buffer=None,
                dim=plan.collective_dim,
                multi_device_global_semaphore=ccl_cache.get_all_gather(plan.cluster_axis),
                num_links=ccl_runtime["num_links"],
                memory_config=None,
                topology=ccl_runtime["topology"],
                subdevice_id=kwargs.get("sub_device_id"),
                cluster_axis=plan.cluster_axis,
                use_optimal_ccl_for_llama=False,
                use_broadcast=False,
                barrier_semaphore=ccl_cache.get_barrier(plan.cluster_axis),
                chunks_per_sync=ccl_runtime["chunks_per_sync"],
                num_workers_per_link=ccl_runtime["num_workers_per_link"],
                num_buffers_per_channel=ccl_runtime["num_buffers_per_channel"],
                sub_core_grids=None,
            )
            original_input = prepared.input_tensor_a
            try:
                prepared.input_tensor_a = gathered
                return _run_minimal_matmul(signature, prepared, kwargs, config_descriptor=descriptor)
            finally:
                prepared.input_tensor_a = original_input
                _deallocate_result(gathered)

        candidates.append(Candidate(descriptor=candidate_descriptor, run=_runner))
    return candidates


def _build_minimal_then_reduce_scatter_candidates(
    signature: AutoMatmulSignature,
    prepared: PreparedMatmulInputs,
    kwargs: dict[str, Any],
    plan: DistributedCollectivePlan,
) -> list[Candidate]:
    if not _can_use_minimal_matmul_common(signature, prepared.bias):
        return []

    device = prepared.input_tensor_a.device()
    if device is None:
        return []
    grid = device.compute_with_storage_grid_size()
    descriptors = _build_minimal_descriptors(signature, bias=prepared.bias, grid=grid)
    candidates: list[Candidate] = []
    for descriptor in descriptors:
        candidate_descriptor = {
            "kind": "minimal_matmul_then_reduce_scatter",
            "collective_dim": plan.collective_dim,
            "cluster_axis": plan.cluster_axis,
            **descriptor,
        }

        def _runner(descriptor=descriptor) -> Any:
            ttnn = _ttnn()
            mesh_device = prepared.input_tensor_a.device()
            ccl_cache = AutoMatmulCCLCache.get(mesh_device)
            ccl_runtime = _get_ccl_runtime(mesh_device, plan.cluster_axis)
            mm_out = _run_minimal_matmul(signature, prepared, kwargs, config_descriptor=descriptor)
            try:
                return ttnn.experimental.reduce_scatter_minimal_async(
                    mm_out,
                    None,
                    plan.collective_dim,
                    ccl_cache.get_reduce_scatter(plan.cluster_axis),
                    barrier_semaphore=ccl_cache.get_barrier(plan.cluster_axis),
                    num_links=ccl_runtime["num_links"],
                    memory_config=kwargs.get("memory_config"),
                    intermediate_memory_config=None,
                    topology=ccl_runtime["topology"],
                    subdevice_id=kwargs.get("sub_device_id"),
                    cluster_axis=plan.cluster_axis,
                    chunks_per_sync=ccl_runtime["chunks_per_sync"],
                    num_workers_per_link=ccl_runtime["num_workers_per_link"],
                    num_buffers_per_channel=ccl_runtime["num_buffers_per_channel"],
                    compute_kernel_config=kwargs.get("compute_kernel_config"),
                )
            finally:
                _deallocate_result(mm_out)

        candidates.append(Candidate(descriptor=candidate_descriptor, run=_runner))
    return candidates


def _build_matmul_then_reduce_scatter_candidate(
    signature: AutoMatmulSignature,
    prepared: PreparedMatmulInputs,
    kwargs: dict[str, Any],
    plan: DistributedCollectivePlan,
    *,
    base_operation: Any,
) -> Candidate:
    descriptor = {
        "kind": "linear_then_reduce_scatter" if signature.is_linear else "matmul_then_reduce_scatter",
        "collective_dim": plan.collective_dim,
        "cluster_axis": plan.cluster_axis,
    }

    def _runner() -> Any:
        ttnn = _ttnn()
        mesh_device = prepared.input_tensor_a.device()
        ccl_cache = AutoMatmulCCLCache.get(mesh_device)
        ccl_runtime = _get_ccl_runtime(mesh_device, plan.cluster_axis)
        call_kwargs = dict(kwargs)
        if signature.is_linear:
            call_kwargs["bias"] = prepared.bias
        mm_out = base_operation(prepared.input_tensor_a, prepared.input_tensor_b, **call_kwargs)
        try:
            return ttnn.experimental.reduce_scatter_minimal_async(
                mm_out,
                None,
                plan.collective_dim,
                ccl_cache.get_reduce_scatter(plan.cluster_axis),
                barrier_semaphore=ccl_cache.get_barrier(plan.cluster_axis),
                num_links=ccl_runtime["num_links"],
                memory_config=kwargs.get("memory_config"),
                intermediate_memory_config=None,
                topology=ccl_runtime["topology"],
                subdevice_id=kwargs.get("sub_device_id"),
                cluster_axis=plan.cluster_axis,
                chunks_per_sync=ccl_runtime["chunks_per_sync"],
                num_workers_per_link=ccl_runtime["num_workers_per_link"],
                num_buffers_per_channel=ccl_runtime["num_buffers_per_channel"],
                compute_kernel_config=kwargs.get("compute_kernel_config"),
            )
        finally:
            _deallocate_result(mm_out)

    return Candidate(descriptor=descriptor, run=_runner)


def _build_minimal_matmul_reduce_scatter_candidates(
    signature: AutoMatmulSignature,
    prepared: PreparedMatmulInputs,
    kwargs: dict[str, Any],
    plan: DistributedCollectivePlan,
) -> list[Candidate]:
    if "blackhole" in signature.arch:
        return []
    if not _can_use_minimal_matmul_common(signature, prepared.bias):
        return []

    ttnn = _ttnn()
    device = prepared.input_tensor_a.device()
    if device is None:
        return []

    ccl_runtime = _get_ccl_runtime(device, plan.cluster_axis)
    compute_kernel_config = _get_default_compute_kernel_config(device, kwargs.get("compute_kernel_config"))
    if compute_kernel_config is None:
        return []
    helper_module = _load_tt_dit_matmul_helpers()
    fused_descriptor: dict[str, Any]
    if helper_module is not None and hasattr(helper_module, "get_fused_mmrs_config"):
        fused_params = helper_module.get_fused_mmrs_config(
            signature.m,
            signature.k,
            signature.n,
            device.compute_with_storage_grid_size(),
            ccl_runtime["num_links"],
        )
        fused_config = fused_params["config"]
        fused_descriptor = {
            "kind": "minimal_matmul_strided_reduce_scatter_async",
            "collective_dim": plan.collective_dim,
            "cluster_axis": plan.cluster_axis,
            "reduce_scatter_core_grid_offset": [
                int(fused_params["reduce_scatter_core_grid_offset"].x),
                int(fused_params["reduce_scatter_core_grid_offset"].y),
            ],
            "num_links": int(fused_params["num_links"]),
            "num_workers_per_link": fused_params.get("num_workers_per_link"),
            "num_buffers_per_channel": fused_params.get("num_buffers_per_channel"),
            "chunk_width_in_mm_blocks": fused_params.get("chunk_width_in_mm_blocks"),
            **_make_minimal_descriptor(
                grid=fused_config.compute_with_storage_grid_size,
                m_block=int(fused_config.M_block_size),
                k_block=int(fused_config.K_block_size),
                n_block=int(fused_config.N_block_size),
                subblock_h=int(fused_config.subblock_h),
                subblock_w=int(fused_config.subblock_w),
            ),
        }
    else:
        compute_grid = device.compute_with_storage_grid_size()
        mm_grid = ttnn.CoreCoord(compute_grid.x, max(1, compute_grid.y - 1))
        descriptors = _build_minimal_descriptors(signature, bias=prepared.bias, grid=mm_grid)
        if not descriptors:
            return []
        descriptor = descriptors[0]
        rs_zone_capacity = max(1, (compute_grid.y - mm_grid.y) * compute_grid.x)
        fused_descriptor = {
            "kind": "minimal_matmul_strided_reduce_scatter_async",
            "collective_dim": plan.collective_dim,
            "cluster_axis": plan.cluster_axis,
            "reduce_scatter_core_grid_offset": [0, int(mm_grid.y)],
            "num_links": ccl_runtime["num_links"],
            "num_workers_per_link": max(1, rs_zone_capacity // max(1, 2 * ccl_runtime["num_links"]) - 1),
            "num_buffers_per_channel": ccl_runtime["num_buffers_per_channel"],
            "chunk_width_in_mm_blocks": 1,
            **descriptor,
        }

    def _runner() -> Any:
        ccl_cache = AutoMatmulCCLCache.get(prepared.input_tensor_a.device())
        mm_out, rs_out = ttnn.experimental.minimal_matmul_strided_reduce_scatter_async(
            prepared.input_tensor_a,
            prepared.input_tensor_b,
            plan.collective_dim,
            ccl_cache.get_reduce_scatter(plan.cluster_axis),
            ttnn.CoreCoord(
                fused_descriptor["reduce_scatter_core_grid_offset"][0],
                fused_descriptor["reduce_scatter_core_grid_offset"][1],
            ),
            compute_kernel_config=compute_kernel_config,
            num_links=fused_descriptor["num_links"],
            memory_config_mm=kwargs.get("memory_config"),
            rs_output_mem_config=kwargs.get("memory_config"),
            rs_intermediate_mem_config=None,
            topology=_get_ccl_runtime(prepared.input_tensor_a.device(), plan.cluster_axis)["topology"],
            cluster_axis=plan.cluster_axis,
            bias=prepared.bias,
            fused_activation=kwargs.get("activation"),
            config=_make_minimal_config(fused_descriptor),
            barrier_semaphore=ccl_cache.get_barrier(plan.cluster_axis),
            using_persistent_buffers=False,
            sub_device_id=kwargs.get("sub_device_id"),
            num_workers_per_link=fused_descriptor["num_workers_per_link"],
            num_buffers_per_channel=fused_descriptor["num_buffers_per_channel"],
            chunk_width_in_mm_blocks=fused_descriptor["chunk_width_in_mm_blocks"],
            optional_rs_output_tensor=None,
            fused_ternary_scalar=None,
            addcmul_input_tensor1=None,
            addcmul_input_tensor2=None,
            dtype=kwargs.get("dtype"),
        )
        _deallocate_result(mm_out)
        return rs_out

    return [Candidate(descriptor=fused_descriptor, run=_runner)]


def _build_candidate_from_descriptor(
    descriptor: dict[str, Any],
    *,
    base_operation: Any,
    prepared: PreparedMatmulInputs,
    kwargs: dict[str, Any],
    signature: AutoMatmulSignature,
) -> Candidate | None:
    kind = descriptor.get("kind")
    plan = _infer_distributed_plan(signature)

    if kind in {"default_matmul", "default_linear"}:
        return _build_default_candidate(
            base_operation=base_operation,
            prepared=prepared,
            kwargs=_default_candidate_kwargs(signature, prepared, kwargs),
            signature=signature,
        )
    if kind == "minimal_matmul":
        for candidate in _build_local_minimal_candidates(signature, prepared, kwargs):
            if candidate.descriptor == descriptor:
                return candidate
    if kind == "program_config":
        for candidate in _build_program_config_candidates(signature, prepared, kwargs, base_operation=base_operation):
            if candidate.descriptor == descriptor:
                return candidate
    if kind in {"all_gather_then_matmul", "all_gather_then_linear"}:
        candidate = _build_all_gather_then_matmul_candidate(
            signature, prepared, kwargs, plan, base_operation=base_operation
        )
        return candidate if candidate.descriptor == descriptor else None
    if kind == "all_gather_matmul_async":
        for candidate in _build_all_gather_matmul_candidates(signature, prepared, kwargs, plan):
            if candidate.descriptor == descriptor:
                return candidate
    if kind == "all_gather_minimal_matmul_async":
        for candidate in _build_all_gather_minimal_candidates(signature, prepared, kwargs, plan):
            if candidate.descriptor == descriptor:
                return candidate
    if kind == "all_gather_then_minimal_matmul":
        for candidate in _build_all_gather_then_minimal_candidates(signature, prepared, kwargs, plan):
            if candidate.descriptor == descriptor:
                return candidate
    if kind in {"matmul_then_reduce_scatter", "linear_then_reduce_scatter"}:
        candidate = _build_matmul_then_reduce_scatter_candidate(
            signature, prepared, kwargs, plan, base_operation=base_operation
        )
        return candidate if candidate.descriptor == descriptor else None
    if kind == "minimal_matmul_then_reduce_scatter":
        for candidate in _build_minimal_then_reduce_scatter_candidates(signature, prepared, kwargs, plan):
            if candidate.descriptor == descriptor:
                return candidate
    if kind == "minimal_matmul_strided_reduce_scatter_async":
        for candidate in _build_minimal_matmul_reduce_scatter_candidates(signature, prepared, kwargs, plan):
            if candidate.descriptor == descriptor:
                return candidate
    return None


def _build_candidates(
    signature: AutoMatmulSignature,
    prepared: PreparedMatmulInputs,
    kwargs: dict[str, Any],
    *,
    base_operation: Any,
) -> list[Candidate]:
    distributed_plan = _infer_distributed_plan(signature)
    candidates: list[Candidate] = []
    if distributed_plan.kind == "gather_before_matmul":
        candidates.append(
            _build_all_gather_then_matmul_candidate(
                signature, prepared, kwargs, distributed_plan, base_operation=base_operation
            )
        )
        candidates.extend(_build_all_gather_then_minimal_candidates(signature, prepared, kwargs, distributed_plan))
        candidates.extend(_build_all_gather_matmul_candidates(signature, prepared, kwargs, distributed_plan))
        candidates.extend(_build_all_gather_minimal_candidates(signature, prepared, kwargs, distributed_plan))
    elif distributed_plan.kind == "matmul_before_reduce_scatter":
        candidates.append(
            _build_matmul_then_reduce_scatter_candidate(
                signature, prepared, kwargs, distributed_plan, base_operation=base_operation
            )
        )
        candidates.extend(_build_minimal_then_reduce_scatter_candidates(signature, prepared, kwargs, distributed_plan))
        candidates.extend(
            _build_minimal_matmul_reduce_scatter_candidates(signature, prepared, kwargs, distributed_plan)
        )
    elif distributed_plan.kind == "unsupported":
        return []
    else:
        # A single-output-tile matmul has nothing to distribute across the core
        # grid and completes in noise-level time, so tuning is pointless and only
        # risks selecting a less-accurate config by timing noise.  Skip the sweep
        # and run the default op alone (with fp32 accumulation via
        # _default_candidate_kwargs for reference precision).
        if _is_single_output_tile(signature):
            return [
                _build_default_candidate(
                    base_operation=base_operation,
                    prepared=prepared,
                    kwargs=_default_candidate_kwargs(signature, prepared, kwargs),
                    signature=signature,
                )
            ]

        candidates.append(
            _build_default_candidate(
                base_operation=base_operation,
                prepared=prepared,
                kwargs=kwargs,
                signature=signature,
            )
        )
        candidates.extend(_build_program_config_candidates(signature, prepared, kwargs, base_operation=base_operation))
        candidates.extend(_build_local_minimal_candidates(signature, prepared, kwargs))

    return candidates


def _result_to_torch(result: Any) -> Any:
    """Convert a candidate output (ttnn.Tensor, or first element of a tuple) to a
    flattened float32 host torch tensor for correctness comparison, or None."""
    if isinstance(result, (list, tuple)):
        result = result[0] if result else None
    if result is None:
        return None
    try:
        import torch
    except Exception:
        return None
    ttnn = _ttnn()
    try:
        host = ttnn.to_torch(result)
        return host.detach().to(torch.float32).flatten()
    except Exception:
        return None


def _outputs_match(reference: Any, candidate: Any) -> bool:
    """Size-independent correctness check via relative L2 error.

    Correlation (PCC) is unusable here because many matmuls have tiny logical
    outputs (e.g. a (1,2)·(2,2) matmul yields 2 elements) and correlation over
    2 points is always ±1.  Relative L2 error works for any size: a valid config
    differs from the reference only by bf16 rounding (well under the threshold),
    while a config that is invalid-but-runs is off by O(1)."""
    if reference is None or candidate is None:
        return False
    try:
        import torch
    except Exception:
        return False
    if reference.numel() == 0 or reference.numel() != candidate.numel():
        return False
    ref = reference.to(torch.float32)
    cand = candidate.to(torch.float32)
    if torch.isnan(cand).any() or torch.isinf(cand).any():
        return False
    ref_norm = torch.linalg.vector_norm(ref).item()
    diff_norm = torch.linalg.vector_norm(cand - ref).item()
    if ref_norm <= 1e-8:
        # Reference is ~zero; the candidate must be ~zero too.
        return diff_norm <= 1e-6
    return (diff_norm / ref_norm) <= _CORRECTNESS_REL_ERR_THRESHOLD


def _candidate_matches_reference(candidate: Candidate, device: Any, reference_torch: Any) -> bool:
    """Run a candidate once and check its output matches the reference."""
    try:
        _sync_device(device)
        output = candidate.run()
        _sync_device(device)
    except Exception:
        return False
    try:
        candidate_torch = _result_to_torch(output)
    finally:
        _deallocate_result(output)
    return _outputs_match(reference_torch, candidate_torch)


def _compute_reference_output(candidates: list[Candidate], device: Any) -> Any:
    """Compute the trusted reference output from the default-config candidate.

    The default candidate runs the stock ttnn.matmul/linear with no program
    config, which is the ground truth every tuned candidate must match.  Returns
    a flattened host torch tensor, or None if unavailable (correctness gating is
    then skipped and selection falls back to latency-only)."""
    default_candidate = next((c for c in candidates if str(c.descriptor.get("kind", "")).startswith("default")), None)
    if default_candidate is None:
        return None
    try:
        _sync_device(device)
        output = default_candidate.run()
        _sync_device(device)
    except Exception:
        return None
    try:
        return _result_to_torch(output)
    finally:
        _deallocate_result(output)


def _benchmark_candidate_eager(candidate: Candidate, device: Any) -> tuple[float, list[float], str]:
    _sync_device(device)
    warmup_output = candidate.run()
    _sync_device(device)
    _deallocate_result(warmup_output)

    samples_us: list[float] = []
    for _ in range(_BENCHMARK_ITERS):
        start = time.perf_counter()
        output = candidate.run()
        _sync_device(device)
        elapsed_us = (time.perf_counter() - start) * 1_000_000.0
        samples_us.append(elapsed_us)
        _deallocate_result(output)
    average_us = sum(samples_us) / len(samples_us)
    return average_us, samples_us, "eager"


def _benchmark_candidate_trace(
    candidate: Candidate, device: Any, *, queue_id: int = 0
) -> tuple[float, list[float], str]:
    ttnn = _ttnn()
    _sync_device(device)
    warmup_output = candidate.run()
    _sync_device(device)
    _deallocate_result(warmup_output)

    trace_id = None
    trace_output = None
    try:
        trace_id = ttnn.begin_trace_capture(device, cq_id=queue_id)
        trace_output = candidate.run()
        ttnn.end_trace_capture(device, trace_id, cq_id=queue_id)
        _sync_device(device)

        samples_us: list[float] = []
        for _ in range(_BENCHMARK_ITERS):
            start = time.perf_counter()
            ttnn.execute_trace(device, trace_id, cq_id=queue_id, blocking=False)
            _sync_device(device)
            samples_us.append((time.perf_counter() - start) * 1_000_000.0)
        average_us = sum(samples_us) / len(samples_us)
        return average_us, samples_us, "trace"
    finally:
        if trace_id is not None:
            try:
                ttnn.release_trace(device, trace_id)
            except Exception:
                pass
        _deallocate_result(trace_output)


def _benchmark_candidate(candidate: Candidate, device: Any, *, queue_id: int = 0) -> tuple[float, list[float], str]:
    ttnn = _ttnn()
    if all(
        hasattr(ttnn, attr) for attr in ("begin_trace_capture", "end_trace_capture", "execute_trace", "release_trace")
    ):
        try:
            return _benchmark_candidate_trace(candidate, device, queue_id=queue_id)
        except Exception:
            pass
    return _benchmark_candidate_eager(candidate, device)


def _callable_accepts_keyword(function: Callable[..., Any], keyword: str) -> bool:
    try:
        parameters = inspect.signature(function).parameters.values()
    except (TypeError, ValueError):
        return True
    return any(parameter.kind == inspect.Parameter.VAR_KEYWORD or parameter.name == keyword for parameter in parameters)


def _make_recommendations(signature: AutoMatmulSignature, prepared: PreparedMatmulInputs) -> list[str]:
    recommendations: list[str] = []
    if prepared.staged_rhs_from_host:
        recommendations.append("RHS weight was staged from host and replicated to the target device/mesh.")
    if prepared.staged_bias_from_host:
        recommendations.append("Bias was staged from host and replicated to the target device/mesh.")

    if signature.input_tensor_a.get("layout") is not None and "TILE" not in signature.input_tensor_a.get("layout", ""):
        recommendations.append("TILE activations would unlock additional minimal-matmul candidates.")
    if signature.input_tensor_b.get("layout") is not None and "TILE" not in signature.input_tensor_b.get("layout", ""):
        recommendations.append("TILE RHS weights would unlock additional minimal-matmul candidates.")

    plan = _infer_distributed_plan(signature)
    if plan.kind == "gather_before_matmul" and prepared.staged_rhs_from_host:
        recommendations.append(
            "Pre-sharding the RHS along the output dimension can unlock fused all-gather matmul recipes."
        )
    if plan.kind == "matmul_before_reduce_scatter" and prepared.staged_rhs_from_host:
        recommendations.append(
            "Pre-sharding the RHS along its K dimension can unlock fused reduce-scatter matmul recipes."
        )
    if plan.kind == "unsupported":
        recommendations.append(
            "Distributed topology did not match the supported v1 gather-before-matmul or matmul-before-reduce-scatter patterns."
        )
    if "blackhole" in signature.arch and plan.kind == "matmul_before_reduce_scatter":
        recommendations.append(
            "Blackhole currently forces the selector away from fused minimal matmul + reduce-scatter due to known instability."
        )

    return recommendations


def _append_recommendation(recommendations: list[str], message: str) -> list[str]:
    if message in recommendations:
        return recommendations
    return [*recommendations, message]


def _make_passthrough_selection(
    signature: AutoMatmulSignature,
    prepared: PreparedMatmulInputs,
    *,
    winner_kind: str,
    message: str,
) -> dict[str, Any]:
    cache = AutoMatmulCache()
    return {
        "cache_hit": False,
        "cache_path": str(cache.path_for(signature)),
        "winner": {"kind": winner_kind},
        "candidate_timings_us": [],
        "recommendations": _append_recommendation(_make_recommendations(signature, prepared), message),
        "candidate": None,
    }


def _selection_summary(selection: dict[str, Any]) -> dict[str, Any]:
    return {
        "winner": selection.get("winner"),
        "candidate_timings_us": list(selection.get("candidate_timings_us", [])),
        "recommendations": list(selection.get("recommendations", [])),
    }


def _select_candidate(
    signature: AutoMatmulSignature,
    prepared: PreparedMatmulInputs,
    kwargs: dict[str, Any],
    *,
    base_operation: Any,
    allow_tuning: bool,
) -> dict[str, Any]:
    cache = AutoMatmulCache()
    cached_record = cache.load(signature)
    if cached_record is not None:
        candidate = _build_candidate_from_descriptor(
            cached_record["winner"],
            base_operation=base_operation,
            prepared=prepared,
            kwargs=kwargs,
            signature=signature,
        )
        if candidate is not None:
            return {
                "cache_hit": True,
                "cache_path": str(cache.path_for(signature)),
                "winner": cached_record["winner"],
                "candidate_timings_us": cached_record.get("candidate_timings_us", []),
                "recommendations": cached_record.get("recommendations", []),
                "candidate": candidate,
            }
        cache.invalidate(signature)

    if not allow_tuning:
        runtime_record = cache.load_runtime(signature)
        if runtime_record is not None:
            return {
                "cache_hit": False,
                "cache_path": str(cache.path_for(signature)),
                "winner": runtime_record.get("winner"),
                "candidate_timings_us": runtime_record.get("candidate_timings_us", []),
                "recommendations": runtime_record.get("recommendations", []),
                "candidate": None,
            }
        return {
            "cache_hit": False,
            "cache_path": str(cache.path_for(signature)),
            "winner": None,
            "candidate_timings_us": [],
            "recommendations": ["No cache entry available and tuning was disabled."],
            "candidate": None,
        }

    candidates = _build_candidates(signature, prepared, kwargs, base_operation=base_operation)
    if not candidates:
        return _make_passthrough_selection(
            signature,
            prepared,
            winner_kind="no_supported_candidates",
            message="No supported auto-config candidates were available for this signature.",
        )

    device = prepared.input_tensor_a.device()
    candidate_timings: list[dict[str, Any]] = []
    winner: Candidate | None = None
    winner_avg_us: float | None = None
    queue_id = int(kwargs["queue_id"] if "queue_id" in kwargs else (kwargs.get("cq_id") or 0))
    benchmark_accepts_queue_id = _callable_accepts_keyword(_benchmark_candidate, "queue_id")

    # Trusted reference output (default config).  Any tuned candidate whose result
    # doesn't correlate with it is rejected before selection, so we never pick a
    # faster-but-numerically-wrong config for a shape it happens to run on.
    reference_torch = _compute_reference_output(candidates, device)

    for candidate in candidates:
        is_default = str(candidate.descriptor.get("kind", "")).startswith("default")
        if reference_torch is not None and not is_default:
            if not _candidate_matches_reference(candidate, device, reference_torch):
                candidate_timings.append(
                    {
                        "descriptor": candidate.descriptor,
                        "status": "incorrect",
                    }
                )
                continue

        try:
            if benchmark_accepts_queue_id:
                avg_us, samples_us, benchmark_mode = _benchmark_candidate(candidate, device, queue_id=queue_id)
            else:
                avg_us, samples_us, benchmark_mode = _benchmark_candidate(candidate, device)
        except Exception as exc:
            candidate_timings.append(
                {
                    "descriptor": candidate.descriptor,
                    "status": "error",
                    "error": str(exc),
                }
            )
            continue

        candidate_timings.append(
            {
                "descriptor": candidate.descriptor,
                "status": "ok",
                "average_us": avg_us,
                "samples_us": samples_us,
                "benchmark_mode": benchmark_mode,
            }
        )
        if winner is None or avg_us < winner_avg_us:
            winner = candidate
            winner_avg_us = avg_us

    if winner is None:
        winner = candidates[0]
        candidate_timings.append(
            {
                "descriptor": winner.descriptor,
                "status": "fallback",
                "average_us": None,
                "samples_us": [],
                "benchmark_mode": "none",
            }
        )

    recommendations = _make_recommendations(signature, prepared)
    record = {
        "version": cache.version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "signature": signature.to_dict(),
        "winner": winner.descriptor,
        "candidate_timings_us": candidate_timings,
        "recommendations": recommendations,
    }
    cache.save(signature, record)
    return {
        "cache_hit": False,
        "cache_path": str(cache.path_for(signature)),
        "winner": winner.descriptor,
        "candidate_timings_us": candidate_timings,
        "recommendations": recommendations,
        "candidate": winner,
    }


def _execute_selected_candidate(selection: dict[str, Any]) -> Any:
    candidate = selection.get("candidate")
    if candidate is None:
        raise RuntimeError("No candidate selected for auto-config matmul execution.")
    return candidate.run()


def _run_base_operation(
    *,
    base_operation: Any,
    input_tensor_a: Any,
    input_tensor_b: Any,
    bias: Any | None,
    is_linear: bool,
    kwargs: dict[str, Any],
) -> Any:
    # Strip None-valued kwargs so the C++ binding uses its own defaults, matching
    # the pre-wrapper behaviour where only explicitly-provided arguments were passed.
    # Passing None explicitly can select different code paths in the C++ pybind11
    # binding (e.g. output_tile=None forcing a default tile that conflicts with an
    # explicit program_config).
    forwarded = {k: v for k, v in kwargs.items() if v is not None}
    if is_linear:
        return base_operation(input_tensor_a, input_tensor_b, bias=bias, **forwarded)
    return base_operation(input_tensor_a, input_tensor_b, **forwarded)


def _device_mesh_shape(device: Any) -> tuple[int, ...]:
    shape_obj = getattr(device, "shape", (1,))
    if hasattr(shape_obj, "dims"):
        return tuple(int(shape_obj[i]) for i in range(shape_obj.dims()))
    try:
        return tuple(int(dim) for dim in shape_obj)
    except TypeError:
        return (1,)


def _device_num_devices(device: Any) -> int:
    return int(getattr(device, "get_num_devices", lambda: 1)())


def _resolve_placement_device(activation: Any, mesh_device: Any) -> Any:
    if mesh_device is not None:
        return mesh_device
    ttnn = _ttnn()
    if isinstance(activation, ttnn.Tensor):
        return activation.device()
    raise ValueError("place_weight requires either `activation` (a ttnn.Tensor on device) or `mesh_device`.")


def _activation_k_shard(activation: Any, transpose_a: bool) -> tuple[int | None, int, int | None]:
    """Describe how the activation is sharded along its contraction (K) dim.

    Returns ``(cluster_axis, distribution_factor, normalized_k_dim)``.  The
    cluster axis is None when the activation is not sharded on K."""
    ttnn = _ttnn()
    if not isinstance(activation, ttnn.Tensor):
        return None, 1, None
    rank = len(_shape_to_tuple(getattr(activation, "shape", ())))
    if rank < 2:
        return None, 1, None
    k_dim = rank - 2 if transpose_a else rank - 1
    normalized_k = _normalize_dim(k_dim, rank)
    topology = _serialize_topology(activation)
    axis = _topology_shard_axis_for_dim(topology, normalized_k)
    if axis is None:
        return None, 1, normalized_k
    return axis, _topology_distribution_factor(topology, axis), normalized_k


def _weight_contraction_dim(weight_shape: tuple[int, ...], transpose_b: bool) -> int:
    rank = len(weight_shape)
    return rank - 1 if transpose_b else rank - 2


def _resolve_weight_dtype(dtype: Any, activation: Any) -> Any:
    if dtype is not None:
        return dtype
    act_dtype = getattr(activation, "dtype", None)
    if act_dtype is not None:
        return act_dtype
    return _ttnn().bfloat16


def _make_shard_to_mesh_mapper(
    device: Any, *, weight_k_dim: int, cluster_axis: int | None, mesh_shape: tuple[int, ...]
) -> Any:
    ttnn = _ttnn()
    non_unit_axes = [index for index, dim in enumerate(mesh_shape) if int(dim) > 1]
    if len(mesh_shape) <= 1 or len(non_unit_axes) <= 1 or cluster_axis is None:
        return ttnn.ShardTensorToMesh(device, dim=weight_k_dim)
    dims: list[int | None] = [None, None]
    dims[cluster_axis] = weight_k_dim
    return ttnn.ShardTensor2dMesh(device, mesh_shape=tuple(int(dim) for dim in mesh_shape), dims=tuple(dims))


def _stage_weight(
    weight: Any, device: Any, *, role: str, dtype: Any, layout: Any, memory_config: Any, mesh_mapper: Any
) -> Any:
    ttnn = _ttnn()
    return ttnn.as_tensor(
        weight,
        dtype=dtype,
        layout=layout,
        device=device,
        mesh_mapper=mesh_mapper,
        memory_config=memory_config,
        cache_file_name=_host_tensor_cache_name(role, weight, dtype),
    )


def _placement_reference_output(
    activation: Any,
    weight_host: Any,
    *,
    transpose_a: bool,
    transpose_b: bool,
    k_shard_axis: int | None,
    k_dim: int | None,
) -> Any:
    """Trusted torch ground-truth ``activation @ weight`` (flattened float32).

    The host weight makes an independent reference possible even though the
    distributed recipes are not internally correctness-gated.  Returns None if a
    reference cannot be built, in which case sharded placements are not trusted."""
    try:
        import torch
    except Exception:
        return None
    ttnn = _ttnn()
    try:
        if k_shard_axis is not None and k_dim is not None:
            act_host = ttnn.to_torch(activation, mesh_composer=ttnn.ConcatMeshToTensor(activation.device(), dim=k_dim))
        else:
            shards = ttnn.get_device_tensors(activation)
            act_host = ttnn.to_torch(shards[0] if shards else activation)
        act_host = act_host.detach().to(torch.float32)
        weight = (
            weight_host.detach().to(torch.float32)
            if torch.is_tensor(weight_host)
            else torch.as_tensor(weight_host, dtype=torch.float32)
        )
        lhs = act_host.transpose(-1, -2) if transpose_a else act_host
        rhs = weight.transpose(-1, -2) if transpose_b else weight
        return (lhs @ rhs).flatten()
    except Exception:
        return None


def _placed_output_to_host_flat(result: Any, device: Any, output_shard_dim: int | None) -> Any:
    try:
        import torch
    except Exception:
        return None
    ttnn = _ttnn()
    if isinstance(result, (list, tuple)):
        result = result[0] if result else None
    if result is None:
        return None
    try:
        if output_shard_dim is not None:
            host = ttnn.to_torch(result, mesh_composer=ttnn.ConcatMeshToTensor(device, dim=output_shard_dim))
        else:
            shards = ttnn.get_device_tensors(result)
            host = ttnn.to_torch(shards[0] if shards else result)
        return host.detach().to(torch.float32).flatten()
    except Exception:
        return None


def _winner_avg_us(selection: dict[str, Any]) -> float | None:
    winner = selection.get("winner")
    best: float | None = None
    for entry in selection.get("candidate_timings_us", []):
        if entry.get("status") != "ok":
            continue
        average_us = entry.get("average_us")
        if average_us is None:
            continue
        if entry.get("descriptor") == winner and best is None:
            best = average_us
        if best is None or average_us < best:
            best = average_us
    return best


def _measure_weight_placement(
    activation: Any,
    staged_weight: Any,
    *,
    transpose_a: bool,
    transpose_b: bool,
    is_linear: bool,
    base_operation: Any,
    reference_flat: Any,
) -> dict[str, Any]:
    """Run the measured selector for one staged weight placement and verify the
    winner against the torch ground-truth reference.

    Any failure while building the signature, selecting, or running the recipe is
    treated as an unverified placement rather than propagated: place_weight must
    never crash the caller just because a distributed recipe was not runnable."""
    kwargs = {
        "transpose_a": transpose_a,
        "transpose_b": transpose_b,
        "memory_config": None,
        "dtype": None,
        "activation": None,
    }
    try:
        signature = _build_signature(
            activation,
            staged_weight,
            bias=None,
            transpose_a=transpose_a,
            transpose_b=transpose_b,
            memory_config=None,
            dtype=None,
            activation=None,
            is_linear=is_linear,
        )
        plan = _infer_distributed_plan(signature)
        prepared = PreparedMatmulInputs(input_tensor_a=activation, input_tensor_b=staged_weight, bias=None)
        selection = _select_candidate(signature, prepared, kwargs, base_operation=base_operation, allow_tuning=True)
        output_is_sharded = plan.kind == "matmul_before_reduce_scatter"
        verified = _verify_placement_output(
            selection.get("candidate"),
            activation.device(),
            reference_flat,
            output_shard_dim=plan.collective_dim if output_is_sharded else None,
        )
        return {
            "avg_us": _winner_avg_us(selection),
            "winner": selection.get("winner"),
            "verified": verified,
            "output_is_sharded": output_is_sharded,
        }
    except Exception as exc:
        return {"avg_us": None, "winner": None, "verified": False, "output_is_sharded": True, "error": str(exc)}


def _verify_placement_output(candidate: Any, device: Any, reference_flat: Any, *, output_shard_dim: int | None) -> bool:
    if candidate is None or reference_flat is None:
        return False
    try:
        _sync_device(device)
        output = candidate.run()
        _sync_device(device)
    except Exception:
        return False
    try:
        host = _placed_output_to_host_flat(output, device, output_shard_dim)
    finally:
        _deallocate_result(output)
    return _outputs_match(reference_flat, host)


def place_weight(
    weight: Any,
    *,
    activation: Any = None,
    mesh_device: Any = None,
    transpose_a: bool = False,
    transpose_b: bool = False,
    is_linear: bool = False,
    dtype: Any = None,
    memory_config: Any = None,
    layout: Any = None,
    strategy: str = "auto",
    measure: bool | None = None,
) -> WeightPlacement:
    """Place a host weight on the target device, choosing (and carrying out) the
    distribution the module considers best for it.

    The reviewer ask this answers: a user should not need to know how to shard a
    weight across a mesh.  Hand ``place_weight`` a host ``torch.Tensor`` (plus the
    activation it will multiply, or a ``mesh_device``) and it returns a placed
    ``ttnn.Tensor`` ready for ``ttnn.matmul`` / ``ttnn.linear``.

    Placement follows the activation's layout, because only one weight placement
    forms a shape-consistent matmul: a replicated activation pairs with a
    replicated weight; a K-sharded (row-parallel) activation pairs with a
    K-sharded weight (reduce-scatter recipe).  On a single device the weight is
    replicated.

    Deciding and staging the placement runs nothing on-device.  Passing
    ``measure=True`` additionally executes the K-sharded reduce-scatter recipe,
    lets the selector pick the fastest variant, and verifies it against a torch
    ground-truth reference built from the host weight (``WeightPlacement.verified``).
    Measurement is opt-in because the "minimal" reduce-scatter collectives are
    built for ring topologies and can hang on non-ring layouts; keep it off unless
    the target mesh is CCL-capable.

    Args:
        weight: host ``torch.Tensor`` (or an already-placed ``ttnn.Tensor``).

    Keyword Args:
        activation: the ``ttnn.Tensor`` this weight will multiply.  Used to infer
            the target device, K-sharding, and to build the correctness reference.
        mesh_device: target device/mesh; required if ``activation`` is not given.
        transpose_a / transpose_b: matches the eventual matmul/linear call.
        is_linear: build linear (vs matmul) candidate recipes when measuring.
        dtype / memory_config / layout: staging controls (sensible defaults).
        strategy: ``"auto"`` (default), ``"replicate"``, or ``"shard_k"``.
        measure: opt in (``True``) to execute + verify a K-sharded reduce-scatter
            placement on-device.  Defaults to ``False`` (decide + stage only) so
            place_weight never launches a collective that could hang on non-ring
            hardware.

    Returns:
        WeightPlacement: the placed tensor plus the recorded decision.
    """
    ttnn = _ttnn()
    device = _resolve_placement_device(activation, mesh_device)
    num_devices = _device_num_devices(device)
    mesh_shape = _device_mesh_shape(device)
    layout = layout if layout is not None else ttnn.TILE_LAYOUT
    memory_config = memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG
    recommendations: list[str] = []

    # An already-placed ttnn weight is respected as-is; place_weight only decides
    # placement for host tensors it stages itself.
    if isinstance(weight, ttnn.Tensor):
        return WeightPlacement(
            tensor=weight,
            strategy="preplaced",
            mesh_shape=mesh_shape,
            recommendations=["Weight was already a device tensor; placement left unchanged."],
        )

    target_dtype = _resolve_weight_dtype(dtype, activation)

    # Single device (or degenerate mesh): replication is the only placement.
    if num_devices <= 1:
        staged = _stage_weight(
            weight,
            device,
            role="weight_replicate",
            dtype=target_dtype,
            layout=layout,
            memory_config=memory_config,
            mesh_mapper=None,
        )
        return WeightPlacement(tensor=staged, strategy="replicate", mesh_shape=mesh_shape)

    k_shard_axis, _factor, k_dim = _activation_k_shard(activation, transpose_a)
    weight_shape = _shape_to_tuple(getattr(weight, "shape", ()))
    weight_k_dim = _weight_contraction_dim(weight_shape, transpose_b)

    # Decide which placements are viable for this context.
    want_replicate = strategy in ("auto", "replicate")
    want_shard_k = strategy in ("auto", "shard_k")

    # For a given activation topology only ONE weight placement forms a
    # shape-consistent, correct matmul (a distributed ttnn tensor reports its
    # per-shard local shape, not the global one):
    #   - a K-sharded activation must pair with a K-sharded weight
    #     (row-parallel; both operands carry the same local K -> reduce-scatter),
    #   - a replicated activation pairs with a replicated weight.
    # A replicated weight against a K-sharded activation is the all-gather
    # recipe, whose signature mixes the local activation K with the full weight
    # K; that layout is outside the v1 candidate families, so place_weight does
    # not offer it.
    shard_k_viable = (
        want_shard_k
        and k_shard_axis is not None
        and len(weight_shape) >= 2
        and weight_shape[weight_k_dim] % max(1, int(mesh_shape[k_shard_axis])) == 0
    )

    if want_shard_k and k_shard_axis is not None and not shard_k_viable:
        recommendations.append(
            "Weight K dimension is not divisible by the mesh shard count; placing a replicated weight instead."
        )

    if shard_k_viable:
        strategy_choice = "shard_k"
        shard_dim: int | None = weight_k_dim
        mesh_mapper = _make_shard_to_mesh_mapper(
            device, weight_k_dim=weight_k_dim, cluster_axis=k_shard_axis, mesh_shape=mesh_shape
        )
    else:
        if k_shard_axis is None:
            recommendations.append(
                "Activation is not sharded along its contraction (K) dimension, so only a replicated weight "
                "produces a full result with the v1 recipes. Shard the activation on K (row-parallel) to let "
                "place_weight place and measure a reduce-scatter tensor-parallel weight."
            )
        elif want_replicate:
            recommendations.append(
                "Activation is sharded on K but a replicated weight was requested; note the all-gather recipe "
                "for this layout is outside the v1 candidate families."
            )
        strategy_choice = "replicate"
        shard_dim = None
        mesh_mapper = ttnn.ReplicateTensorToMesh(device)

    staged = _stage_weight(
        weight,
        device,
        role=f"weight_{strategy_choice}",
        dtype=target_dtype,
        layout=layout,
        memory_config=memory_config,
        mesh_mapper=mesh_mapper,
    )

    output_is_sharded = strategy_choice == "shard_k"
    verified = False
    candidate_timings: list[dict[str, Any]] = []

    # Measurement is OPT-IN (measure=True), never the default.  Measuring a
    # K-sharded placement executes the reduce-scatter collective on-device, and
    # those "minimal" CCL recipes are built for ring topologies (e.g. T3K): on a
    # non-ring layout the collective can *hang* rather than raise, and a device
    # hang cannot be caught the way an exception can.  So by default place_weight
    # only decides + stages the placement (no collective is run); callers on
    # CCL-capable hardware pass measure=True to pick the best reduce-scatter
    # recipe and verify it against the host-weight ground truth.
    should_measure = bool(measure) and output_is_sharded and isinstance(activation, ttnn.Tensor)
    if not should_measure and output_is_sharded:
        recommendations.append(
            "K-sharded (reduce-scatter) placement was staged but not executed/verified (measure=False). "
            "Pass measure=True on CCL-capable hardware (e.g. a ring topology) to measure and verify it."
        )
    if should_measure and output_is_sharded:
        base_operation = _get_cpp_base_operation(is_linear)
        reference_flat = _placement_reference_output(
            activation,
            weight,
            transpose_a=transpose_a,
            transpose_b=transpose_b,
            k_shard_axis=k_shard_axis,
            k_dim=k_dim,
        )
        measurement = _measure_weight_placement(
            activation,
            staged,
            transpose_a=transpose_a,
            transpose_b=transpose_b,
            is_linear=is_linear,
            base_operation=base_operation,
            reference_flat=reference_flat,
        )
        verified = bool(measurement["verified"])
        candidate_timings.append(
            {
                "strategy": strategy_choice,
                "shard_dim": shard_dim,
                "avg_us": measurement["avg_us"],
                "verified": verified,
                "output_is_sharded": output_is_sharded,
                "winner": measurement["winner"],
            }
        )
        if not verified:
            recommendations.append(
                "K-sharded (reduce-scatter) placement did not match the ground-truth reference on this device; "
                "verify the recipe before relying on it."
            )

    return WeightPlacement(
        tensor=staged,
        strategy=strategy_choice,
        shard_dim=shard_dim,
        cluster_axis=k_shard_axis if output_is_sharded else None,
        mesh_shape=mesh_shape,
        output_is_sharded=output_is_sharded,
        verified=verified,
        candidate_timings=candidate_timings,
        recommendations=recommendations,
    )


def dispatch_matmul(
    *,
    base_operation: Any,
    input_tensor_a: Any,
    input_tensor_b: Any,
    bias: Any | None,
    is_linear: bool,
    auto_config: bool,
    **kwargs: Any,
) -> Any:
    ttnn = _ttnn()
    if not isinstance(input_tensor_a, ttnn.Tensor):
        return _run_base_operation(
            base_operation=base_operation,
            input_tensor_a=input_tensor_a,
            input_tensor_b=input_tensor_b,
            bias=bias,
            is_linear=is_linear,
            kwargs=kwargs,
        )

    # 1-D inputs (e.g. dot-product: ttnn.matmul(vec, vec)) cannot be handled by
    # the auto-config candidate families which all require M×K × K×N shapes.
    # Check BEFORE _prepare_inputs: staging a 1-D host torch.Tensor via
    # ttnn.as_tensor(layout=TILE_LAYOUT) would silently reshape it to 2-D,
    # making the shape check fire too late.
    if not _supports_matrix_auto_config(
        _shape_to_tuple(getattr(input_tensor_a, "shape", ())),
        _shape_to_tuple(getattr(input_tensor_b, "shape", ())),
    ):
        return _run_base_operation(
            base_operation=base_operation,
            input_tensor_a=input_tensor_a,
            input_tensor_b=input_tensor_b,
            bias=bias,
            is_linear=is_linear,
            kwargs=kwargs,
        )

    prepared = _prepare_inputs(input_tensor_a, input_tensor_b, bias)

    if not auto_config:
        return _run_base_operation(
            base_operation=base_operation,
            input_tensor_a=prepared.input_tensor_a,
            input_tensor_b=prepared.input_tensor_b,
            bias=prepared.bias,
            is_linear=is_linear,
            kwargs=kwargs,
        )

    # Graph-report capture should reflect the user-visible op, not internal
    # tuning warmups and candidate sweeps.
    if getattr(getattr(ttnn, "CONFIG", None), "enable_graph_report", False):
        return _run_base_operation(
            base_operation=base_operation,
            input_tensor_a=prepared.input_tensor_a,
            input_tensor_b=prepared.input_tensor_b,
            bias=prepared.bias,
            is_linear=is_linear,
            kwargs=kwargs,
        )

    # During profiling sessions (slow-dispatch Tracy or device-profiler sync mode)
    # the op identity must remain unambiguous. Benchmarking warmups would inject
    # extra ops into the trace, corrupt op counts, or cause post-processing to
    # time out waiting on oversized device logs.
    # Known limitation: TT_METAL_SLOW_DISPATCH_MODE=1 is also used by non-profiler
    # slow-dispatch workloads (e.g. DeepSeek B1 persistent-loop). Auto-config is
    # disabled there too because trace-based benchmarking does not work correctly in
    # slow dispatch mode. Those callers fall back to the default matmul config,
    # identical to pre-auto-config behaviour.
    if os.environ.get("TT_METAL_SLOW_DISPATCH_MODE") == "1" or os.environ.get("TT_METAL_PROFILER_SYNC") == "1":
        return _run_base_operation(
            base_operation=base_operation,
            input_tensor_a=prepared.input_tensor_a,
            input_tensor_b=prepared.input_tensor_b,
            bias=prepared.bias,
            is_linear=is_linear,
            kwargs=kwargs,
        )

    if not isinstance(prepared.input_tensor_b, ttnn.Tensor):
        return _run_base_operation(
            base_operation=base_operation,
            input_tensor_a=prepared.input_tensor_a,
            input_tensor_b=prepared.input_tensor_b,
            bias=prepared.bias,
            is_linear=is_linear,
            kwargs=kwargs,
        )

    if _has_explicit_override(kwargs):
        return _run_base_operation(
            base_operation=base_operation,
            input_tensor_a=prepared.input_tensor_a,
            input_tensor_b=prepared.input_tensor_b,
            bias=prepared.bias,
            is_linear=is_linear,
            kwargs=kwargs,
        )

    # Incompatible contraction dims: auto-config cannot build a config the base
    # op would accept.  Defer so the base op raises its own native error rather
    # than auto-config surfacing a different exception type/message.
    if not _matmul_dims_compatible(
        _shape_to_tuple(getattr(prepared.input_tensor_a, "shape", ())),
        _shape_to_tuple(getattr(prepared.input_tensor_b, "shape", ())),
        kwargs.get("transpose_a", False),
        kwargs.get("transpose_b", False),
    ):
        return _run_base_operation(
            base_operation=base_operation,
            input_tensor_a=prepared.input_tensor_a,
            input_tensor_b=prepared.input_tensor_b,
            bias=prepared.bias,
            is_linear=is_linear,
            kwargs=kwargs,
        )

    signature = _build_signature(
        prepared.input_tensor_a,
        prepared.input_tensor_b,
        bias=prepared.bias,
        transpose_a=kwargs.get("transpose_a", False),
        transpose_b=kwargs.get("transpose_b", False),
        memory_config=kwargs.get("memory_config"),
        dtype=kwargs.get("dtype"),
        activation=kwargs.get("activation"),
        is_linear=is_linear,
    )
    if _infer_distributed_plan(signature).kind == "unsupported":
        return _run_base_operation(
            base_operation=base_operation,
            input_tensor_a=prepared.input_tensor_a,
            input_tensor_b=prepared.input_tensor_b,
            bias=prepared.bias,
            is_linear=is_linear,
            kwargs=kwargs,
        )

    cache = AutoMatmulCache()
    selection = _select_candidate(
        signature,
        prepared,
        kwargs,
        base_operation=base_operation,
        allow_tuning=True,
    )
    if selection.get("candidate") is None:
        cache.save_runtime(signature, _selection_summary(selection))
        return _run_base_operation(
            base_operation=base_operation,
            input_tensor_a=prepared.input_tensor_a,
            input_tensor_b=prepared.input_tensor_b,
            bias=prepared.bias,
            is_linear=is_linear,
            kwargs=kwargs,
        )

    try:
        result = _execute_selected_candidate(selection)
        cache.save_runtime(signature, _selection_summary(selection))
        return result
    except Exception:
        if selection.get("cache_hit"):
            cache.invalidate(signature)
            cache.invalidate_runtime(signature)
            selection = _select_candidate(
                signature,
                prepared,
                kwargs,
                base_operation=base_operation,
                allow_tuning=True,
            )
            result = _execute_selected_candidate(selection)
            cache.save_runtime(signature, _selection_summary(selection))
            return result
        raise


def explain_matmul(
    input_tensor_a: Any,
    input_tensor_b: Any,
    *,
    bias: Any | None = None,
    transpose_a: bool = False,
    transpose_b: bool = False,
    memory_config: Any = None,
    dtype: Any = None,
    program_config: Any = None,
    activation: Any = None,
    compute_kernel_config: Any = None,
    core_grid: Any = None,
    output_tile: Any = None,
    optional_output_tensor: Any = None,
    global_cb: Any = None,
    sub_device_id: Any = None,
    queue_id: int | None = None,
    is_linear: bool = False,
    allow_tuning: bool = True,
) -> dict[str, Any]:
    ttnn = _ttnn()
    if not isinstance(input_tensor_a, ttnn.Tensor):
        raise TypeError("explain_matmul requires input_tensor_a to be a ttnn.Tensor on device or mesh")

    # When only reading the cache (allow_tuning=False), skip host-tensor staging:
    # staging allocates device memory and writes cache files, which is unexpected
    # for a read-only inspection call.
    if allow_tuning:
        prepared = _prepare_inputs(input_tensor_a, input_tensor_b, bias)
    else:
        prepared = PreparedMatmulInputs(input_tensor_a=input_tensor_a, input_tensor_b=input_tensor_b, bias=bias)
    if not isinstance(prepared.input_tensor_b, ttnn.Tensor):
        raise TypeError("explain_matmul requires input_tensor_b to be a ttnn.Tensor or a host torch.Tensor")

    kwargs = {
        "transpose_a": transpose_a,
        "transpose_b": transpose_b,
        "memory_config": memory_config,
        "dtype": dtype,
        "program_config": program_config,
        "activation": activation,
        "compute_kernel_config": compute_kernel_config,
        "core_grid": core_grid,
        "output_tile": output_tile,
        "optional_output_tensor": optional_output_tensor,
        "global_cb": global_cb,
        "sub_device_id": sub_device_id,
    }
    if queue_id is not None:
        kwargs["queue_id"] = queue_id
    signature = _build_signature(
        prepared.input_tensor_a,
        prepared.input_tensor_b,
        bias=prepared.bias,
        transpose_a=transpose_a,
        transpose_b=transpose_b,
        memory_config=memory_config,
        dtype=dtype,
        activation=activation,
        is_linear=is_linear,
    )
    distributed_plan = _infer_distributed_plan(signature)
    if _has_explicit_override(kwargs):
        cache = AutoMatmulCache()
        return {
            "signature": signature.to_dict(),
            "cache_hit": False,
            "cache_path": str(cache.path_for(signature)),
            "winner": {"kind": "explicit_override"},
            "candidate_timings_us": [],
            "recommendations": [
                "Auto-config is bypassed because explicit low-level configuration arguments were supplied."
            ],
            "distributed_plan": dataclasses.asdict(distributed_plan),
        }
    if distributed_plan.kind == "unsupported":
        selection = _make_passthrough_selection(
            signature,
            prepared,
            winner_kind="unsupported_topology_passthrough",
            message="Auto-config tuning was bypassed because this distributed topology is outside the supported v1 candidate families.",
        )
        return {
            "signature": signature.to_dict(),
            "cache_hit": selection["cache_hit"],
            "cache_path": selection["cache_path"],
            "winner": selection["winner"],
            "candidate_timings_us": selection["candidate_timings_us"],
            "recommendations": selection["recommendations"],
            "distributed_plan": dataclasses.asdict(distributed_plan),
        }
    selection = _select_candidate(
        signature,
        prepared,
        kwargs,
        base_operation=_get_cpp_base_operation(is_linear),
        allow_tuning=allow_tuning,
    )
    return {
        "signature": signature.to_dict(),
        "cache_hit": selection["cache_hit"],
        "cache_path": selection["cache_path"],
        "winner": selection["winner"],
        "candidate_timings_us": selection["candidate_timings_us"],
        "recommendations": selection["recommendations"],
        "distributed_plan": dataclasses.asdict(distributed_plan),
    }
