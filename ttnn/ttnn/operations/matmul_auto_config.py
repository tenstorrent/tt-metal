import json
import math
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import ttnn


@dataclass
class MatmulShapeProfile:
    """Shape profile capturing key dimensions that affect matmul performance."""
    M: int
    K: int
    N: int
    batch_size: int = 1
    is_sharded: bool = False
    shard_layout: Optional[str] = None
    dtype: str = "bfloat16"
    num_devices: int = 1

    def to_tuple(self) -> Tuple[int, ...]:
        return (self.M, self.K, self.N, self.batch_size, self.num_devices)

    def is_compatible(self, other: "MatmulShapeProfile") -> bool:
        return (
            abs(self.M - other.M) / max(self.M, 1) < 0.1
            and abs(self.K - other.K) / max(self.K, 1) < 0.1
            and abs(self.N - other.N) / max(self.N, 1) < 0.1
            and self.batch_size == other.batch_size
            and self.num_devices == other.num_devices
            and self.dtype == other.dtype
        )


@dataclass
class MatmulConfigPerformance:
    """Performance measurement for a specific matmul configuration."""
    config_name: str
    config_params: Dict[str, Any]
    runtime_us: float = 0.0
    peak_sustained_bw: float = 0.0
    est_perf_tflops: float = 0.0
    l1_usage_bytes: int = 0
    sample_count: int = 0

    @property
    def score(self) -> float:
        if self.runtime_us <= 0:
            return float("-inf")
        return -self.runtime_us


class MatmulConfigDatabase:
    """Stores and queries performance profiles for matmul configurations."""

    def __init__(self, db_path: Optional[str] = None):
        self.profiles: Dict[Tuple, Dict[str, MatmulConfigPerformance]] = {}
        self.db_path = Path(db_path) if db_path else None
        if self.db_path and self.db_path.exists():
            self.load()

    def register(
        self,
        shape: MatmulShapeProfile,
        config_name: str,
        config_params: Dict[str, Any],
        runtime_us: float,
        l1_usage_bytes: int = 0,
    ):
        key = shape.to_tuple()
        if key not in self.profiles:
            self.profiles[key] = {}
        if config_name not in self.profiles[key]:
            self.profiles[key][config_name] = MatmulConfigPerformance(
                config_name=config_name,
                config_params=config_params,
                sample_count=0,
            )
        record = self.profiles[key][config_name]
        n = record.sample_count
        record.runtime_us = (record.runtime_us * n + runtime_us) / (n + 1)
        record.l1_usage_bytes = max(record.l1_usage_bytes, l1_usage_bytes)
        record.sample_count += 1
        self.save()

    def query(self, shape: MatmulShapeProfile) -> Optional[MatmulConfigPerformance]:
        key = shape.to_tuple()
        if key in self.profiles and self.profiles[key]:
            return min(self.profiles[key].values(), key=lambda x: x.runtime_us)
        for stored_key, configs in self.profiles.items():
            stored_shape = MatmulShapeProfile(*stored_key)
            if stored_shape.is_compatible(shape) and configs:
                return min(configs.values(), key=lambda x: x.runtime_us)
        return None

    def get_config_for_shape(self, shape: MatmulShapeProfile) -> Optional[Dict[str, Any]]:
        best = self.query(shape)
        if best:
            return best.config_params
        return None

    def get_all_configs(self, shape: MatmulShapeProfile) -> List[MatmulConfigPerformance]:
        key = shape.to_tuple()
        if key in self.profiles:
            return list(self.profiles[key].values())
        result = []
        for stored_key, configs in self.profiles.items():
            stored_shape = MatmulShapeProfile(*stored_key)
            if stored_shape.is_compatible(shape):
                result.extend(configs.values())
        return result

    def save(self):
        if self.db_path:
            with open(self.db_path, "wb") as f:
                pickle.dump(
                    {str(k): {ck: vars(cv) for ck, cv in v.items()} for k, v in self.profiles.items()}, f
                )

    def load(self):
        if self.db_path and self.db_path.exists():
            with open(self.db_path, "rb") as f:
                raw = pickle.load(f)
            for key_str, configs in raw.items():
                key = tuple(int(x) for x in key_str.strip("()").split(", "))
                self.profiles[key] = {}
                for ck, cv in configs.items():
                    cp = MatmulConfigPerformance(**cv)
                    self.profiles[key][ck] = cp

    def to_json(self, filepath: str):
        serializable = {
            "profiles": {
                str(k): {
                    ck: {
                        "config_name": cv.config_name,
                        "runtime_us": cv.runtime_us,
                        "sample_count": cv.sample_count,
                        "l1_usage_bytes": cv.l1_usage_bytes,
                    }
                    for ck, cv in v.items()
                }
                for k, v in self.profiles.items()
            }
        }
        with open(filepath, "w") as f:
            json.dump(serializable, f, indent=2)


class MatmulConfigHeuristic:
    """Heuristic-based matmul config selection using analytical cost model."""

    def __init__(self):
        self.mm_multicore_reuse = ttnn.MatmulMultiCoreReuseProgramConfig
        self.mm_multicast = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig
        self.mm_1d = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig
        self.mm_dram = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig

    def select(
        self,
        M: int,
        K: int,
        N: int,
        batch_size: int = 1,
        is_sharded: bool = False,
        shard_layout: Optional[str] = None,
        num_devices: int = 1,
        device_grid: Optional[Tuple[int, int]] = None,
        dtype: str = "bfloat16",
    ) -> Tuple[str, Dict[str, Any]]:
        grid_x, grid_y = device_grid or (8, 8)
        num_cores = grid_x * grid_y
        total_M = M * batch_size

        if is_sharded and shard_layout == "width_sharded":
            return self._select_1d_width_sharded(M, K, N, batch_size, grid_x, grid_y, num_cores, dtype)

        if is_sharded and shard_layout == "height_sharded":
            return self._select_1d_height_sharded(M, K, N, batch_size, grid_x, grid_y, num_cores, dtype)

        if is_sharded and shard_layout == "block_sharded":
            return self._select_block_sharded(M, K, N, batch_size, grid_x, grid_y, dtype)

        if num_devices > 1:
            return self._select_multi_device(M, K, N, batch_size, num_devices, grid_x, grid_y, dtype)

        narrow_threshold = 8
        height_width_ratio = (
            total_M / N if total_M > N else N / total_M
        ) if N > 0 else 0

        is_narrow = height_width_ratio > narrow_threshold or M <= 32 or N <= 32

        if is_narrow:
            mcand_in0 = N > total_M
            return self._select_1d_systolic(M, K, N, batch_size, grid_x, grid_y, num_cores, mcand_in0, dtype)

        per_core_M = math.ceil(total_M / 32 / grid_y)
        per_core_N = math.ceil(N / 32 / grid_x)
        tile_M = total_M // 32
        tile_N = N // 32
        tile_K = K // 32

        dram_config = self._try_dram_sharded(tile_M, tile_N, tile_K, num_cores, K)
        if dram_config:
            return dram_config

        in0_block_w = min(4, tile_K)
        while in0_block_w > 1 and tile_K % in0_block_w != 0:
            in0_block_w -= 1

        batch_fused = batch_size > 1 and per_core_M >= 1
        per_core_M_val = max(1, per_core_M)
        per_core_N_val = max(1, per_core_N)

        return (
            "MatmulMultiCoreReuseMultiCastProgramConfig",
            {
                "compute_with_storage_grid_size": (grid_x, grid_y),
                "in0_block_w": in0_block_w,
                "out_subblock_h": min(4, per_core_M_val),
                "out_subblock_w": min(2, per_core_N_val),
                "out_block_h": per_core_M_val,
                "out_block_w": per_core_N_val,
                "per_core_M": per_core_M_val,
                "per_core_N": per_core_N_val,
                "transpose_mcast": False,
                "fuse_batch": batch_fused,
            },
        )

    def _select_1d_systolic(
        self, M, K, N, batch_size, grid_x, grid_y, num_cores, mcast_in0, dtype
    ):
        total_M = M * batch_size
        tile_M = total_M // 32
        tile_N = N // 32
        tile_K = K // 32

        if mcast_in0:
            per_core_M = tile_M
            per_core_N = max(1, math.ceil(tile_N / num_cores))
        else:
            per_core_M = max(1, math.ceil(tile_M / num_cores))
            per_core_N = tile_N

        in0_block_w = 2 if tile_K % 2 == 0 else 1

        return (
            "MatmulMultiCoreReuseMultiCast1DProgramConfig",
            {
                "compute_with_storage_grid_size": (grid_x, grid_y),
                "in0_block_w": in0_block_w,
                "out_subblock_h": min(4, per_core_M),
                "out_subblock_w": 1,
                "out_block_h": per_core_M,
                "out_block_w": per_core_N,
                "per_core_M": per_core_M,
                "per_core_N": per_core_N,
                "fuse_batch": batch_size > 1,
                "mcast_in0": mcast_in0,
            },
        )

    def _select_1d_width_sharded(self, M, K, N, batch_size, grid_x, grid_y, num_cores, dtype):
        tile_N = N // 32
        per_core_N = max(1, math.ceil(tile_N / num_cores))
        tile_K = K // 32
        in0_block_w = 2 if tile_K % 2 == 0 else 1

        return (
            "MatmulMultiCoreReuseMultiCast1DProgramConfig",
            {
                "compute_with_storage_grid_size": (grid_x, grid_y),
                "in0_block_w": in0_block_w,
                "out_subblock_h": 1,
                "out_subblock_w": min(4, per_core_N),
                "out_block_h": M // 32,
                "out_block_w": per_core_N,
                "per_core_M": M // 32,
                "per_core_N": per_core_N,
                "fuse_batch": batch_size > 1,
                "mcast_in0": True,
            },
        )

    def _select_1d_height_sharded(self, M, K, N, batch_size, grid_x, grid_y, num_cores, dtype):
        total_M = M * batch_size
        per_core_M = max(1, math.ceil(total_M / 32 / num_cores))

        return (
            "MatmulMultiCoreReuseMultiCast1DProgramConfig",
            {
                "compute_with_storage_grid_size": (grid_x, grid_y),
                "in0_block_w": 2,
                "out_subblock_h": min(4, per_core_M),
                "out_subblock_w": 1,
                "out_block_h": per_core_M,
                "out_block_w": N // 32,
                "per_core_M": per_core_M,
                "per_core_N": N // 32,
                "fuse_batch": batch_size > 1,
                "mcast_in0": False,
            },
        )

    def _select_block_sharded(self, M, K, N, batch_size, grid_x, grid_y, dtype):
        total_M = M * batch_size
        per_core_M = max(1, math.ceil(total_M / 32 / grid_y))
        per_core_N = max(1, math.ceil(N / 32 / grid_x))
        tile_K = K // 32
        in0_block_w = min(4, tile_K)

        return (
            "MatmulMultiCoreReuseMultiCastProgramConfig",
            {
                "compute_with_storage_grid_size": (grid_x, grid_y),
                "in0_block_w": in0_block_w,
                "out_subblock_h": min(4, per_core_M),
                "out_subblock_w": min(2, per_core_N),
                "out_block_h": per_core_M,
                "out_block_w": per_core_N,
                "per_core_M": per_core_M,
                "per_core_N": per_core_N,
                "transpose_mcast": False,
                "fuse_batch": batch_size > 1,
            },
        )

    def _select_multi_device(self, M, K, N, batch_size, num_devices, grid_x, grid_y, dtype):
        per_device_M = M
        per_device_N = N // num_devices
        return self.select(
            per_device_M, K, per_device_N, batch_size,
            device_grid=(grid_x, grid_y), dtype=dtype,
        )

    def _try_dram_sharded(self, tile_M, tile_N, tile_K, num_cores, K):
        is_large = (tile_M * tile_N) > 256
        is_narrow_small = tile_M < 4 or tile_N < 4
        if is_large and not is_narrow_small and K > 4096:
            return (
                "MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig",
                {
                    "in0_block_w": min(4, tile_K),
                    "per_core_M": max(1, min(8, tile_M)),
                    "per_core_N": max(1, min(8, tile_N)),
                },
            )
        return None


class MatmulAutoConfig:
    """Main API for automatic matmul configuration selection.

    Uses a combination of heuristic analysis and a performance database
    to select the optimal matmul configuration for any given input shape.
    """

    _instance = None
    _db: MatmulConfigDatabase = None
    _heuristic: MatmulConfigHeuristic = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._db = MatmulConfigDatabase()
            cls._heuristic = MatmulConfigHeuristic()
        return cls._instance

    @classmethod
    def get_config(
        cls,
        input_tensor_a: ttnn.Tensor,
        input_tensor_b: ttnn.Tensor,
        transpose_a: bool = False,
        transpose_b: bool = False,
        batch_size: int = 1,
        device_grid: Optional[Tuple[int, int]] = None,
    ) -> ttnn.MatmulProgramConfig:
        shape = cls._extract_shape(input_tensor_a, input_tensor_b, transpose_a, transpose_b)
        best_config = cls._db.get_config_for_shape(shape)
        if best_config:
            config_name = best_config.get("config_name", "")
            params = {k: v for k, v in best_config.items() if k != "config_name"}
            return cls._build_config(config_name, params)

        config_name, params = cls._heuristic.select(
            M=shape.M,
            K=shape.K,
            N=shape.N,
            batch_size=shape.batch_size,
            is_sharded=shape.is_sharded,
            shard_layout=shape.shard_layout,
            num_devices=shape.num_devices,
            device_grid=device_grid,
            dtype=shape.dtype,
        )
        return cls._build_config(config_name, params)

    @classmethod
    def benchmark_and_register(
        cls,
        input_tensor_a: ttnn.Tensor,
        input_tensor_b: ttnn.Tensor,
        program_config: ttnn.MatmulProgramConfig,
        transpose_a: bool = False,
        transpose_b: bool = False,
    ):
        shape = cls._extract_shape(input_tensor_a, input_tensor_b, transpose_a, transpose_b)
        config_name = type(program_config).__name__
        config_params = {}
        if hasattr(program_config, "__dict__"):
            config_params = {
                k: v for k, v in program_config.__dict__.items()
                if not k.startswith("_") and not callable(v)
            }

        import time
        device = input_tensor_a.device()
        start = time.perf_counter()
        num_warmup = 3
        num_iters = 10
        for _ in range(num_warmup):
            ttnn.matmul(input_tensor_a, input_tensor_b, program_config=program_config, transpose_a=transpose_a,
                       transpose_b=transpose_b)
            ttnn.synchronize_device(device)
        start = time.perf_counter()
        for _ in range(num_iters):
            ttnn.matmul(input_tensor_a, input_tensor_b, program_config=program_config, transpose_a=transpose_a,
                       transpose_b=transpose_b)
            ttnn.synchronize_device(device)
        elapsed = time.perf_counter() - start
        avg_runtime_us = (elapsed / num_iters) * 1e6

        cls._db.register(shape, config_name, config_params, avg_runtime_us)
        return avg_runtime_us

    @classmethod
    def _extract_shape(
        cls,
        input_tensor_a: ttnn.Tensor,
        input_tensor_b: ttnn.Tensor,
        transpose_a: bool = False,
        transpose_b: bool = False,
    ) -> MatmulShapeProfile:
        a_shape = input_tensor_a.shape
        b_shape = input_tensor_b.shape

        M = a_shape[-2]
        K_a = a_shape[-1]
        K_b = b_shape[-2]
        K = K_a if not transpose_a else a_shape[-2]
        N = b_shape[-1] if not transpose_b else b_shape[-2]

        batch_dims = list(a_shape[:-2])
        batch_size = max(1, math.prod(batch_dims)) if batch_dims else 1

        dtype = str(input_tensor_a.dtype).split(".")[-1]
        is_sharded = hasattr(input_tensor_a, "is_sharded") and input_tensor_a.is_sharded()

        num_devices = 1
        if hasattr(input_tensor_a, "device"):
            device = input_tensor_a.device()
            if hasattr(device, "num_devices"):
                num_devices = device.num_devices()

        shard_layout = None
        if is_sharded:
            mem_config = input_tensor_a.memory_config()
            shard_layout = str(mem_config.memory_layout) if hasattr(mem_config, "memory_layout") else None

        return MatmulShapeProfile(
            M=M, K=K, N=N, batch_size=batch_size,
            is_sharded=is_sharded, shard_layout=shard_layout,
            dtype=dtype, num_devices=num_devices,
        )

    @classmethod
    def _build_config(cls, config_name: str, params: Dict[str, Any]) -> ttnn.MatmulProgramConfig:
        config_map = {
            "MatmulMultiCoreReuseProgramConfig": ttnn.MatmulMultiCoreReuseProgramConfig,
            "MatmulMultiCoreReuseMultiCastProgramConfig": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig,
            "MatmulMultiCoreReuseMultiCast1DProgramConfig": ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig,
            "MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig": ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig,
            "MatmulMultiCoreReuseMultiCastBatchedDRAMShardedProgramConfig": ttnn.MatmulMultiCoreReuseMultiCastBatchedDRAMShardedProgramConfig,
        }
        config_cls = config_map.get(config_name, ttnn.MatmulMultiCoreReuseMultiCastProgramConfig)

        filtered_params = {}
        for k, v in params.items():
            if k == "compute_with_storage_grid_size" and isinstance(v, (list, tuple)):
                filtered_params[k] = ttnn.CoreCoord(v[0], v[1])
            elif k == "fused_activation" and v is not None:
                if isinstance(v, dict):
                    filtered_params[k] = ttnn.UnaryWithParam(**v)
            elif k == "allowed_worker_cores" and v is not None:
                if isinstance(v, (list, tuple)):
                    from ttnn import CoreRange, CoreRangeSet, CoreCoord
                    if len(v) == 2:
                        filtered_params[k] = CoreRangeSet(
                            CoreRange(CoreCoord(0, 0), CoreCoord(v[0] - 1, v[1] - 1))
                        )
            else:
                filtered_params[k] = v

        return config_cls(**filtered_params)

    @classmethod
    def save_database(cls, path: str):
        cls._db.save()
        cls._db.to_json(path.replace(".pkl", ".json") if path.endswith(".pkl") else path + ".json")

    @classmethod
    def load_database(cls, path: str):
        cls._db = MatmulConfigDatabase(db_path=path)
        cls._db.load()


def matmul_auto(
    input_tensor_a: ttnn.Tensor,
    input_tensor_b: ttnn.Tensor,
    transpose_a: bool = False,
    transpose_b: bool = False,
    memory_config: Optional[ttnn.MemoryConfig] = None,
    dtype: Optional[ttnn.DataType] = None,
    activation: Optional[Union[str, ttnn.UnaryWithParam]] = None,
    compute_kernel_config: Optional[ttnn.DeviceComputeKernelConfig] = None,
    core_grid: Optional[ttnn.CoreGrid] = None,
) -> ttnn.Tensor:
    """Torch.matmul-like API with automatic optimal config selection.

    Accepts both single-device and multi-device input tensors.
    Automatically selects the most performant matmul configuration.
    """
    auto_config = MatmulAutoConfig()
    program_config = auto_config.get_config(
        input_tensor_a, input_tensor_b,
        transpose_a=transpose_a, transpose_b=transpose_b,
    )
    return ttnn.matmul(
        input_tensor_a,
        input_tensor_b,
        transpose_a=transpose_a,
        transpose_b=transpose_b,
        memory_config=memory_config,
        dtype=dtype,
        program_config=program_config,
        activation=activation,
        compute_kernel_config=compute_kernel_config,
        core_grid=core_grid,
    )


ttnn.matmul_auto = matmul_auto
ttnn.MatmulAutoConfig = MatmulAutoConfig
