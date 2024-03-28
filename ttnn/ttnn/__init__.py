# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import pathlib

from loguru import logger

TTNN_CACHE_PATH = pathlib.Path().home() / ".cache" / "ttnn"
MODEL_CACHE_PATH = TTNN_CACHE_PATH / "models"
TMP_DIR = pathlib.Path("/") / "tmp" / "ttnn"


def get_bool_env_var(name, default):
    variable = os.environ.get(name, f"{default}")
    if variable == "True":
        return True
    elif variable == "False":
        return False
    else:
        raise RuntimeError(f'The value has to be either "True" or "False"')


TTNN_ENABLE_MODEL_CACHE = get_bool_env_var("TTNN_ENABLE_MODEL_CACHE", "False")
if TTNN_ENABLE_MODEL_CACHE:
    logger.info(f"ttnn: model cache was enabled")

TTNN_ENABLE_FAST_RUNTIME_MODE = get_bool_env_var("TTNN_ENABLE_FAST_RUNTIME_MODE", "False")
if TTNN_ENABLE_FAST_RUNTIME_MODE:
    logger.info(f"ttnn: fast runtime mode was enabled")

TTNN_ENABLE_LOGGING = get_bool_env_var("TTNN_ENABLE_LOGGING", "False")
if TTNN_ENABLE_LOGGING:
    logger.info(f"ttnn: enabled logging (and disabled fast dispatch mode)")

import tt_lib as _tt_lib
import ttnn._ttnn

from ttnn._ttnn.multi_device import get_device_tensors, aggregate_as_tensor

from ttnn.types import (
    TILE_SIZE,
    DataType,
    uint16,
    uint32,
    bfloat8_b,
    bfloat4_b,
    bfloat16,
    float32,
    MemoryConfig,
    MathFidelity,
    DRAM_MEMORY_CONFIG,
    L1_MEMORY_CONFIG,
    L1_BLOCK_SHARDED_MEMORY_CONFIG,
    L1_HEIGHT_SHARDED_MEMORY_CONFIG,
    L1_WIDTH_SHARDED_MEMORY_CONFIG,
    ShardStrategy,
    ShardOrientation,
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
)

from ttnn.validation import validate_input_tensor
import ttnn.tracer

from ttnn.decorators import (
    register_operation,
    query_operations,
    enable_debug_decorator,
    override_pcc_of_debug_decorator,
    disable_validate_decorator,
    register_pre_operation_hook,
    register_post_operation_hook,
)

import ttnn.experimental

from ttnn.operations.core import (
    from_torch,
    to_torch,
    to_device,
    from_device,
    to_layout,
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
)

from ttnn.operations.matmul import (
    matmul,
    linear,
    create_matmul_program_config,
    create_matmul_1d_systolic_array_program_config,
)

from ttnn.operations.others import (
    embedding,
    # fused operations
    softmax,
    # reduction operations
    mean,
    upsample,
)

from ttnn.operations.creation import (
    arange,
    empty,
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
)

from ttnn.operations.losses import (
    l1_loss,
    mse_loss,
)

from ttnn.operations.data_movement import (
    concat,
    pad,
    permute,
    split,
    repeat_interleave,
    repeat,
)

from ttnn.operations.unary import (
    exp,
    tanh,
    gelu,
    rsqrt,
    relu,
    silu,
    log,
    sin,
    cos,
    tan,
    asin,
    acos,
    atan,
    sinh,
    cosh,
    asinh,
    acosh,
    atanh,
    logical_not,
    logit,
    signbit,
)

from ttnn.operations.binary import (
    pow,
    add,
    sub,
    subtract,
    mul,
    multiply,
    ldexp,
    logical_and,
    logical_or,
    logical_xor,
    logaddexp,
    logaddexp2,
    xlogy,
    add_and_apply_activation,
    add_and_apply_activation_,
    nextafter,
    polyval,
    maximum,
    minimum,
)

from ttnn.operations.ternary import (
    addcdiv,
    addcmul,
    mac,
    where,
)

from ttnn.operations.relational import (
    gtz,
    ltz,
    gez,
    lez,
    nez,
    eqz,
    gt,
    gte,
    lt,
    lte,
    eq,
    ne,
    isclose,
)

from ttnn.operations.activation import (
    clip,
    elu,
    hardshrink,
    hardsigmoid,
    hardswish,
    hardtanh,
    heaviside,
    leaky_relu,
    log_sigmoid,
    mish,
    prelu,
    relu6,
    sigmoid,
    sign,
    softshrink,
    softsign,
    swish,
    softplus,
    tanhshrink,
    threshold,
    glu,
    geglu,
    reglu,
    swiglu,
)

from ttnn.operations.math import (
    i0,
    isfinite,
    isinf,
    isnan,
    isneginf,
    isposinf,
    lgamma,
    log10,
    log1p,
    log2,
    multigammaln,
    neg,
    abs,
    cbrt,
    deg2rad,
    digamma,
    erf,
    erfc,
    erfinv,
    exp2,
    expm1,
    atan2,
    hypot,
    squared_difference,
    lerp,
    polygamma,
    rad2deg,
    reciprocal,
    sqrt,
    square,
    tril,
    triu,
)

from ttnn.operations.normalization import (
    layer_norm,
    rms_norm,
    group_norm,
    create_group_norm_weight_bias_rm,
    create_group_norm_input_mask,
    determine_expected_group_norm_sharded_config_and_grid_size,
)

from ttnn.operations.ccl import all_gather

from ttnn.operations import transformer
from ttnn.operations import kv_cache
from ttnn.operations.conv2d import Conv2d
from ttnn.operations.maxpool2d import (
    MaxPool2d,
    global_avg_pool2d,
)

from ttnn._ttnn.reports import print_l1_buffers
