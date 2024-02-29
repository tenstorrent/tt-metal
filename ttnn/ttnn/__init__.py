# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import pathlib

TTNN_CACHE_PATH = pathlib.Path().home() / ".cache" / "ttnn"
MODEL_CACHE_PATH = TTNN_CACHE_PATH / "models"
TMP_DIR = pathlib.Path("/") / "tmp" / "ttnn"


def get_bool_env_var(name, default):
    variable = os.environ.get("TTNN_ENABLE_MODEL_CACHE", f"{default}")
    if variable == "True":
        return True
    elif variable == "False":
        return False
    else:
        raise RuntimeError(f'The value has to be either "True" or "False"')


TTNN_ENABLE_MODEL_CACHE = get_bool_env_var("TTNN_ENABLE_MODEL_CACHE", "False")

import tt_lib as _tt_lib
import ttnn._ttnn

from ttnn.types import (
    TILE_SIZE,
    DataType,
    uint16,
    uint32,
    bfloat8_b,
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
)

from ttnn.device import Device, open_device, close_device, manage_device, dump_device_memory_state

from ttnn.core import (
    has_storage_type_of,
    has_padding,
    is_sharded,
    get_memory_config,
    create_sharded_memory_config,
)

from ttnn.validation import validate_input_tensor
import ttnn.tracer

from ttnn.decorators import (
    register_operation,
    query_all_registered_operations,
    enable_debug_decorator,
    override_pcc_of_debug_decorator,
    disable_validate_decorator,
)

import ttnn.experimental

from ttnn.program_cache import (
    enable_program_cache,
)

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
)

from ttnn.operations.matmul import (
    matmul,
    linear,
    create_matmul_1d_systolic_array_config,
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
)

from ttnn.operations import transformer
from ttnn.operations.conv2d import Conv2d
from ttnn.operations.maxpool2d import (
    MaxPool2d,
    global_avg_pool2d,
)
