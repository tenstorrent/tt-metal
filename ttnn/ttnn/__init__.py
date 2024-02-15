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

import tt_lib as ttl
import ttnn._ttnn

from ttnn._ttnn import TTNN_ENABLE_LOGGING

from ttnn.types import (
    TILE_SIZE,
    Device,
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
    ShardStrategy,
    ShardOrientation,
    DEFAULT_SHARD_ORIENTATION,
    Layout,
    ROW_MAJOR_LAYOUT,
    TILE_LAYOUT,
    StorageType,
    DEVICE_STORAGE_TYPE,
    Shape,
    Tensor,
)

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

import ttnn.ttl as ttl

from ttnn.device import open, close

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
)

from ttnn.operations.matmul import (
    matmul,
    linear,
)

from ttnn.operations.others import (
    embedding,
    pad_to_tile,
    unpad_from_tile,
    # fused operations
    softmax,
    # reduction operations
    mean,
    upsample,
)

from ttnn.operations.data_movement import (
    concat,
    pad,
    permute,
    split,
    repeat_interleave,
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
    clone,
)

from ttnn.operations.binary import (
    pow,
    add,
    sub,
    subtract,
    mul,
    multiply,
    add_and_apply_activation,
    add_and_apply_activation_,
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
)

from ttnn.operations.activation import (
    clip,
    elu,
    hardshrink,
    hardswish,
    hardtanh,
    heaviside,
    leaky_relu,
    log_sigmoid,
    mish,
    prelu,
    relu_max,
    relu_min,
    relu6,
    sigmoid,
    sign,
    softshrink,
    softsign,
    swish,
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
