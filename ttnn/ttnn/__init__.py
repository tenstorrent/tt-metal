# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import contextlib
import dataclasses
import json
import os
import pathlib
import pprint
import subprocess
from typing import Optional

from loguru import logger

import tt_lib as _tt_lib
import ttnn._ttnn

CPP_CONFIG: ttnn._ttnn.core.Config = ttnn._ttnn.CONFIG

UnaryWithParam = _tt_lib.tensor.FusibleActivationWithParam
UnaryOpType = _tt_lib.tensor.FusibleActivation


@dataclasses.dataclass
class Config:
    cache_path: pathlib.Path = pathlib.Path().home() / ".cache" / "ttnn"
    model_cache_path: pathlib.Path = cache_path / "models"
    tmp_dir: pathlib.Path = pathlib.Path("/") / "tmp" / "ttnn"
    enable_model_cache: bool = False
    enable_fast_runtime_mode: bool = True
    throw_exception_on_fallback: bool = False
    enable_logging: bool = False
    enable_graph_report: bool = False
    enable_detailed_buffer_report: bool = False
    enable_detailed_tensor_report: bool = False
    enable_comparison_mode: bool = False
    comparison_mode_pcc: float = 0.9999
    root_report_path: pathlib.Path = pathlib.Path("generated") / "ttnn" / "reports"
    report_name: Optional[str] = None

    @property
    def report_path(self):
        import zlib
        import pickle

        if self.report_name is None:
            return None
        return self.root_report_path / f"{zlib.adler32(pickle.dumps(self.report_name))}"

    def __setattr__(self, name: str, value) -> None:
        object.__setattr__(self, name, value)
        if isinstance(value, pathlib.Path):
            value = str(value)
        setattr(CPP_CONFIG, name, value)
        self.validate(name)

    def validate(self, name):
        if name in {"enable_fast_runtime_mode", "enable_logging"}:
            if self.enable_fast_runtime_mode:
                if self.enable_logging:
                    logger.warning(
                        "Logging cannot be enabled in fast runtime mode. Please disable fast runtime mode if you want to enable logging."
                    )

        if name in {
            "enable_logging",
            "enable_graph_report",
            "enable_detailed_buffer_report",
            "enable_detailed_tensor_report",
        }:
            if not self.enable_logging:
                if self.enable_graph_report:
                    logger.warning("Running without logging. Please enable logging to save graph report")
                if self.enable_detailed_buffer_report:
                    logger.warning("Running without logging. Please enable logging to save detaile buffer report")
                if self.enable_detailed_tensor_report:
                    logger.warning("Running without logging. Please enable logging to save detailed tensor report")


CONFIG = Config()
CONFIG_PATH = None
if "TTNN_CONFIG_PATH" in os.environ:
    CONFIG_PATH = pathlib.Path(os.environ["TTNN_CONFIG_PATH"])

CONFIG_OVERRIDES = os.environ.get("TTNN_CONFIG_OVERRIDES", None)


def load_config_from_dictionary(config, from_file=False):
    global CONFIG
    for key, value in config.items():
        if hasattr(CONFIG, key):
            if getattr(CONFIG, key) is not None:
                value = type(getattr(CONFIG, key))(value)
            setattr(CONFIG, key, value)
        elif from_file:
            logger.warning(
                f"Unknown configuration key: {key}. Please update your configuration file: {CONFIG_PATH}. Or delete it to get the new default config"
            )
        else:
            raise ValueError(f"Unknown configuration key: {key}")


def load_config_from_json_file(json_path):
    global CONFIG
    try:
        with open(json_path, "r") as f:
            config = json.load(f)
        load_config_from_dictionary(config, from_file=True)
    except Exception as e:
        logger.warning(f"Failed to load ttnn configuration from {json_path}: {e}")


def save_config_to_json_file(json_path):
    with open(json_path, "w") as f:
        normalized_config = dataclasses.asdict(CONFIG)
        for key, value in normalized_config.items():
            if isinstance(value, pathlib.Path):
                value = str(value)
            normalized_config[key] = value
        json.dump(normalized_config, f, indent=4)


if CONFIG_PATH is not None:
    if CONFIG_PATH.exists():
        logger.debug(f"Loading ttnn configuration from {CONFIG_PATH}")
        load_config_from_json_file(CONFIG_PATH)
    else:
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        save_config_to_json_file(CONFIG_PATH)

if CONFIG_OVERRIDES is not None:
    logger.debug(f"Loading ttnn configuration overrides from environment variable TTNN_CONFIG_OVERRIDES")
    load_config_from_dictionary(json.loads(CONFIG_OVERRIDES))


import tt_lib as _tt_lib

_tt_lib._check_so_rpath("_ttnn", pathlib.Path(__file__).parent.parent / "tt_lib" / "build" / "lib")
import ttnn._ttnn

logger.debug(f"Initial ttnn.CONFIG:\n{pprint.pformat(dataclasses.asdict(CONFIG))}")


@contextlib.contextmanager
def manage_config(name, value):
    global CONFIG
    original_value = getattr(CONFIG, name)
    setattr(CONFIG, name, value)
    logger.debug(f"Set ttnn.CONFIG.{name} to {value}")
    yield
    setattr(CONFIG, name, original_value)
    logger.debug(f"Restored ttnn.CONFIG.{name} to {original_value}")


from ttnn._ttnn.multi_device import get_device_tensor, get_device_tensors, aggregate_as_tensor

from ttnn.types import (
    TILE_SIZE,
    DataType,
    uint8,
    uint16,
    int32,
    uint32,
    bfloat8_b,
    bfloat4_b,
    bfloat16,
    float32,
    MathFidelity,
    MemoryConfig,
    BufferType,
    TensorMemoryLayout,
    DRAM_MEMORY_CONFIG,
    L1_MEMORY_CONFIG,
    L1_BLOCK_SHARDED_MEMORY_CONFIG,
    L1_HEIGHT_SHARDED_MEMORY_CONFIG,
    L1_WIDTH_SHARDED_MEMORY_CONFIG,
    ShardStrategy,
    ShardOrientation,
    ShardSpec,
    CoreRangeSet,
    CoreRange,
    CoreCoord,
    Layout,
    ROW_MAJOR_LAYOUT,
    TILE_LAYOUT,
    StorageType,
    DEVICE_STORAGE_TYPE,
    CoreGrid,
    CoreRange,
    Shape,
    Tensor,
    DeviceComputeKernelConfig,
    WormholeComputeKernelConfig,
    GrayskullComputeKernelConfig,
    DeviceGrid,
)

from ttnn.device import (
    Device,
    open_device,
    close_device,
    enable_program_cache,
    disable_and_clear_program_cache,
    manage_device,
    synchronize_device,
    dump_device_memory_state,
)

from ttnn.multi_device import (
    DeviceMesh,
    open_device_mesh,
    close_device_mesh,
    get_num_pcie_devices,
    get_num_devices,
    get_pcie_device_ids,
    get_device_ids,
    create_device_mesh,
    TensorToMesh,
    ShardTensorToMesh,
    ReplicateTensorToMesh,
    MeshToTensor,
    ConcatMeshToTensor,
    ListMeshToTensor,
)

from ttnn.core import (
    set_printoptions,
    has_storage_type_of,
    is_tensor_storage_on_device,
    has_tile_padding,
    is_sharded,
    get_memory_config,
    create_sharded_memory_config,
    create_sharded_memory_config_,
    dump_memory_config,
    load_memory_config,
    dump_stack_trace_on_segfault,
)

import ttnn.reflection
from ttnn.validation import validate_input_tensor
import ttnn.tracer
import ttnn.database

from ttnn.decorators import (
    register_operation,
    query_registered_operations,
    register_pre_operation_hook,
    register_post_operation_hook,
    get_golden_function,
    get_fallback_function,
)

import ttnn.experimental
import ttnn.experimental.golden_functions

from ttnn.operations.core import (
    from_torch,
    to_torch,
    to_device,
    from_device,
    to_layout,
    to_dtype,
    reshape,
    to_memory_config,
    deallocate,
    reallocate,
    load_tensor,
    dump_tensor,
    unsqueeze_to_4D,
    squeeze,
    clone,
    as_tensor,
    allocate_tensor_on_device,
    copy_host_to_device_tensor,
)

from ttnn.operations.matmul import (
    matmul,
    linear,
    MatmulMultiCoreReuseProgramConfig,
    MatmulMultiCoreReuseMultiCastProgramConfig,
    MatmulMultiCoreReuseMultiCast1DProgramConfig,
    MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig,
)

from ttnn.operations.embedding import (
    embedding,
)

from ttnn.operations.comparison import (
    pearson_correlation_coefficient,
)

from ttnn.operations.creation import (
    arange,
    empty,
    empty_like,
    full,
    full_like,
    ones,
    ones_like,
    zeros,
    zeros_like,
)

from ttnn.operations.reduction import (
    std,
    var,
    max,
    min,
    sum,
    mean,
    argmax,
    topk,
)

from ttnn.operations.losses import (
    l1_loss,
    mse_loss,
)

from ttnn.operations.data_movement import (
    concat,
    pad,
    permute,
    repeat_interleave,
    repeat,
    upsample,
)

from ttnn.operations.downsample import (
    downsample,
)

from ttnn.operations.unary import (
    abs,
    acos,
    acosh,
    asin,
    asinh,
    atan,
    atanh,
    cbrt,
    celu,
    clip,
    cos,
    cosh,
    deg2rad,
    digamma,
    elu,
    eqz,
    erf,
    erfc,
    erfinv,
    exp,
    exp2,
    expm1,
    glu,
    gelu,
    geglu,
    gez,
    gtz,
    hardshrink,
    hardsigmoid,
    hardswish,
    hardtanh,
    heaviside,
    i0,
    isfinite,
    isinf,
    isnan,
    isneginf,
    isposinf,
    leaky_relu,
    lez,
    logical_not,
    ltz,
    lgamma,
    log,
    log10,
    log1p,
    log2,
    log_sigmoid,
    logit,
    log_sigmoid,
    mish,
    multigammaln,
    neg,
    nez,
    polygamma,
    prelu,
    rad2deg,
    reciprocal,
    relu,
    reglu,
    relu6,
    rsqrt,
    sigmoid,
    sigmoid_accurate,
    sign,
    signbit,
    silu,
    sin,
    sinh,
    softplus,
    softshrink,
    softsign,
    sqrt,
    square,
    swiglu,
    swish,
    tan,
    tanh,
    tanhshrink,
    threshold,
    tril,
    triu,
)

from ttnn.operations.binary import (
    pow,
    add,
    add_,
    sub,
    sub_,
    subtract,
    subtract_,
    mul,
    mul_,
    multiply,
    multiply_,
    ldexp,
    logical_and,
    logical_or,
    logical_xor,
    logaddexp,
    logaddexp2,
    xlogy,
    nextafter,
    polyval,
    maximum,
    minimum,
    atan2,
    hypot,
    squared_difference,
    gt,
    ge,
    lt,
    le,
    eq,
    ne,
    isclose,
    bias_gelu,
    divide,
)


from ttnn.operations.binary_backward import (
    atan2_bw,
    embedding_bw,
    addalpha_bw,
    subalpha_bw,
    sub_bw,
    xlogy_bw,
    hypot_bw,
    ldexp_bw,
    logaddexp_bw,
    logaddexp2_bw,
    squared_difference_bw,
    add_bw,
    binary_eq_bw,
    binary_assign_bw,
    concat_bw,
    binary_le_bw,
    rsub_bw,
    bias_gelu_bw,
)

from ttnn.operations.ternary import (
    addcdiv,
    addcmul,
    mac,
    where,
    lerp,
)

from ttnn.operations.normalization import (
    softmax,
    layer_norm,
    rms_norm,
    group_norm,
    create_group_norm_weight_bias_rm,
    create_group_norm_input_mask,
    determine_expected_group_norm_sharded_config_and_grid_size,
    get_group_norm_cores_accross_channel,
)

from ttnn.operations.trace import (
    begin_trace_capture,
    end_trace_capture,
    execute_trace,
    release_trace,
)

from ttnn.operations.ccl import all_gather

from ttnn.operations import transformer
from ttnn.operations import kv_cache
from ttnn.operations.conv2d import Conv2d, conv2d, Conv2dConfig, get_conv_output_dim
from ttnn.operations.pool import (
    MaxPool2d,
    global_avg_pool2d,
)
from ttnn.operations.copy import typecast
