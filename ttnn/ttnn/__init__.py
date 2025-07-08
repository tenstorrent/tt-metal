# SPDX-FileCopyrightText: © 2023 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import contextlib
import json
import importlib
import os
import pathlib
import re
from types import ModuleType

from loguru import logger

# Sets env and updates shared libs rpath
# This is a tweak required for a proper wheel functioning
import ttnn.library_tweaks

library_tweaks.setup_ttnn_so()

import ttnn._ttnn


Config = ttnn._ttnn.core.Config
CONFIG = ttnn._ttnn.CONFIG
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
        normalized_config = {}
        for key in dir(CONFIG):
            if re.match("^_.+_$", key):
                continue
            value = getattr(CONFIG, key)
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

logger.debug(f"Initial ttnn.CONFIG:\n{CONFIG}")


@contextlib.contextmanager
def manage_config(name, value):
    global CONFIG
    original_value = getattr(CONFIG, name)
    setattr(CONFIG, name, value)
    logger.debug(f"Set ttnn.CONFIG.{name} to {value}")
    yield
    setattr(CONFIG, name, original_value)
    logger.debug(f"Restored ttnn.CONFIG.{name} to {original_value}")


from ttnn._ttnn.multi_device import (
    CppMeshToTensor,
    CppTensorToMesh,
    PlacementReplicate,
    PlacementShard,
    MeshMapperConfig,
    MeshComposerConfig,
    get_device_tensors,
    from_host_shards,
    combine_device_tensors,
    replicate_tensor_to_mesh_mapper,
    shard_tensor_to_mesh_mapper,
    create_mesh_mapper,
    concat_mesh_to_tensor_composer,
    create_mesh_composer,
    aggregate_tensor,
    distribute_tensor,
    get_t3k_physical_device_ids_ring,
)

from ttnn._ttnn.events import (
    MeshEvent,
    record_event,
    wait_for_event,
    event_synchronize,
)

from ttnn._ttnn.operations.trace import (
    MeshTraceId,
    begin_trace_capture,
    end_trace_capture,
    execute_trace,
    release_trace,
)

from ttnn._ttnn.global_circular_buffer import (
    create_global_circular_buffer,
)

from ttnn._ttnn.fabric import FabricConfig, FabricReliabilityMode, set_fabric_config

from ttnn._ttnn.global_semaphore import (
    create_global_semaphore,
    get_global_semaphore_address,
    reset_global_semaphore_value,
)

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
    ShardMode,
    ShardSpec,
    NdShardSpec,
    CoreRangeSet,
    CoreRange,
    CoreCoord,
    Tile,
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
    MeshShape,
    MeshCoordinate,
    MeshCoordinateRange,
    MeshCoordinateRangeSet,
    QueueId,
    UnaryWithParam,
    UnaryOpType,
    BinaryOpType,
    BcastOpMath,
    BcastOpDim,
    CBFormatDescriptor,
    CBDescriptor,
    ReaderConfigDescriptor,
    WriterConfigDescriptor,
    ComputeConfigDescriptor,
    KernelDescriptor,
    SemaphoreDescriptor,
    ProgramDescriptor,
)

from ttnn.device import (
    Device,
    DispatchCoreType,
    DispatchCoreAxis,
    DispatchCoreConfig,
    open_device,
    close_device,
    manage_device,
    synchronize_device,
    dump_device_memory_state,
    get_memory_view,
    get_max_worker_l1_unreserved_size,
    GetPCIeDeviceID,
    GetNumPCIeDevices,
    GetNumAvailableDevices,
    CreateDevice,
    CreateDevices,
    CloseDevice,
    CloseDevices,
    DumpDeviceProfiler,
    SetDefaultDevice,
    GetDefaultDevice,
    format_input_tensor,
    format_output_tensor,
    pad_to_tile_shape,
    SubDevice,
    SubDeviceId,
    SubDeviceManagerId,
    DefaultQueueId,
    init_device_compute_kernel_config,
)

from ttnn.profiler import start_tracy_zone, stop_tracy_zone, tracy_message, tracy_frame

# TODO: remove this after the distributed module is fully integrated
from ttnn.distributed import *

from ttnn.core import (
    set_printoptions,
    has_storage_type_of,
    is_tensor_storage_on_device,
    has_tile_padding,
    is_sharded,
    get_memory_config,
    light_metal_begin_capture,
    light_metal_end_capture,
    LightMetalReplay,
    create_sharded_memory_config,
    create_sharded_memory_config_,
    dump_memory_config,
    load_memory_config,
    dump_stack_trace_on_segfault,
    num_cores_to_corerangeset,
    num_cores_to_corerangeset_in_subcoregrids,
)

import ttnn.reflection
import ttnn.database

from ttnn.decorators import (
    attach_golden_function,
    create_module_if_not_exists,
    dump_operations,
    get_golden_function,
    get_fallback_function,
    query_registered_operations,
    register_cpp_operation,
    register_post_operation_hook,
    register_pre_operation_hook,
    register_python_operation,
)


def auto_register_ttnn_cpp_operations(module):
    for attribute_name in dir(module):
        attribute = getattr(module, attribute_name)
        if hasattr(attribute, "__ttnn_operation__") and attribute.__ttnn_operation__ is None:
            full_name = attribute.python_fully_qualified_name
            module_path, _, func_name = full_name.rpartition(".")
            target_module = create_module_if_not_exists(module_path)
            register_cpp_operation(target_module, func_name, attribute)
        elif isinstance(attribute, ModuleType):
            auto_register_ttnn_cpp_operations(attribute)


auto_register_ttnn_cpp_operations(ttnn._ttnn)

import ttnn.experimental_loader
import ttnn.experimental_loader.golden_functions

import ttnn.operations

sub = ttnn.subtract
sub_ = ttnn.subtract_
mul = ttnn.multiply
mul_ = ttnn.multiply_
div_ = ttnn.divide_


# TODO: pybind the overloaded operators below
ttnn.Tensor.__add__ = lambda self, *args, **kwargs: ttnn.add(self, *args, **kwargs)
ttnn.Tensor.__radd__ = lambda self, *args, **kwargs: ttnn.add(self, *args, **kwargs)
ttnn.Tensor.__sub__ = lambda self, *args, **kwargs: ttnn.subtract(self, *args, **kwargs)
ttnn.Tensor.__mul__ = lambda self, *args, **kwargs: ttnn.multiply(self, *args, **kwargs)
ttnn.Tensor.__rmul__ = lambda self, *args, **kwargs: ttnn.multiply(self, *args, **kwargs)
ttnn.Tensor.__eq__ = lambda self, *args, **kwargs: ttnn.eq(self, *args, **kwargs)
ttnn.Tensor.__ne__ = lambda self, *args, **kwargs: ttnn.ne(self, *args, **kwargs)
ttnn.Tensor.__gt__ = lambda self, *args, **kwargs: ttnn.gt(self, *args, **kwargs)
ttnn.Tensor.__ge__ = lambda self, *args, **kwargs: ttnn.ge(self, *args, **kwargs)
ttnn.Tensor.__lt__ = lambda self, *args, **kwargs: ttnn.lt(self, *args, **kwargs)
ttnn.Tensor.__le__ = lambda self, *args, **kwargs: ttnn.le(self, *args, **kwargs)
ttnn.Tensor.__getitem__ = lambda self, *args, **kwargs: ttnn.operations.core.__getitem__(self, *args, **kwargs)

from ttnn.operations.matmul import (
    MatmulMultiCoreReuseProgramConfig,
    MatmulMultiCoreReuseMultiCastProgramConfig,
    MatmulMultiCoreReuseMultiCast1DProgramConfig,
    MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig,
)

from ttnn.operations.normalization import (
    SoftmaxProgramConfig,
    SoftmaxDefaultProgramConfig,
    SoftmaxShardedMultiCoreProgramConfig,
    LayerNormDefaultProgramConfig,
    LayerNormShardedMultiCoreProgramConfig,
    create_group_norm_weight_bias_rm,
    create_group_norm_input_mask,
    determine_expected_group_norm_sharded_config_and_grid_size,
)

from ttnn.operations.embedding import (
    EmbeddingsType,
)

from ttnn.operations.losses import (
    LossReductionMode,
)

from ttnn.operations.reduction import (
    ReduceType,
)

from ttnn.operations.ccl import (
    Topology,
    teardown_edm_fabric,
    initialize_edm_fabric,
)

from ttnn.operations.conv2d import (
    Conv2dConfig,
    get_conv_output_dim,
    Conv2dSliceConfig,
    Conv2dSliceHeight,
    Conv2dSliceWidth,
    prepare_conv_weights,
    prepare_conv_bias,
    prepare_conv_transpose2d_weights,
    prepare_conv_transpose2d_bias,
    SlidingWindowParallelConfig,
)
from ttnn._ttnn.operations.conv import (
    convert_conv_weight_tensor_to_tiled_layout,
    convert_conv_weight_tensor_to_special_padding_tiled_layout,
    convert_conv_weight_tensor_to_grouped_layout,
)

from ttnn._ttnn.operations.experimental import Conv3dConfig

Conv1dConfig = ttnn._ttnn.operations.conv.Conv2dConfig

from ttnn.operations.transformer import SDPAProgramConfig

import ttnn.graph

if importlib.util.find_spec("torch") is not None:
    import ttnn.tracer

from ttnn._ttnn.device import get_arch_name as _get_arch_name


def get_arch_name():
    return _get_arch_name()
