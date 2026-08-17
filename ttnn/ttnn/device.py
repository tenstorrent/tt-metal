# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import contextlib
from typing import Optional, List

import ttnn
from loguru import logger


# TODO: Device = ttnn._ttnn.Device
Device = ttnn._ttnn.multi_device.MeshDevice
DispatchCoreType = ttnn._ttnn.device.DispatchCoreType
DispatchCoreAxis = ttnn._ttnn.device.DispatchCoreAxis
_DispatchCoreConfig = ttnn._ttnn.device.DispatchCoreConfig
Arch = ttnn._ttnn.device.Arch
DEFAULT_L1_SMALL_SIZE = ttnn._ttnn.device.DEFAULT_L1_SMALL_SIZE
DEFAULT_TRACE_REGION_SIZE = ttnn._ttnn.device.DEFAULT_TRACE_REGION_SIZE
get_max_worker_l1_unreserved_size = ttnn._ttnn.device.get_max_worker_l1_unreserved_size
get_dram_alignment = ttnn._ttnn.device.get_dram_alignment
get_l1_alignment = ttnn._ttnn.device.get_l1_alignment
get_optimal_dram_bank_to_logical_worker_assignment = (
    ttnn._ttnn.device.get_optimal_dram_bank_to_logical_worker_assignment
)
enable_asynchronous_slow_dispatch = ttnn._ttnn.device.enable_asynchronous_slow_dispatch
disable_asynchronous_slow_dispatch = ttnn._ttnn.device.disable_asynchronous_slow_dispatch
is_asynchronous_slow_dispatch_enabled = ttnn._ttnn.device.is_asynchronous_slow_dispatch_enabled


class DispatchCoreConfig(_DispatchCoreConfig):
    def __init__(
        self,
        type: Optional[DispatchCoreType] = None,
        axis: Optional[DispatchCoreAxis] = None,
        fabric_tensix_config=None,
    ):
        resolved_config = ttnn._ttnn.device.create_dispatch_core_config(
            type=type, axis=axis, fabric_tensix_config=fabric_tensix_config
        )
        super().__init__(resolved_config.type, resolved_config.axis)


open_device = ttnn._ttnn.device.open_device
init_device_compute_kernel_config = ttnn._ttnn.operations.core.init_device_compute_kernel_config


def close_device(device: "ttnn.device.Device"):
    """
    Close the device and remove it from the device cache.

    Args:
        device (ttnn.device.Device): The device to close.

    Returns:
        `None`: the device is closed.

    Example:
        >>> device_id = 0
        >>> device = ttnn.open_device(device_id = device_id)
        >>> success = ttnn.close_device(device)
        Closing device 0

    """
    # Try to synchronize first, but don't let failures prevent device close.
    # If synchronize fails (e.g., due to device timeout/hang), we still need
    # to close the device to release handles and allow subsequent operations.
    try:
        synchronize_device(device)
    except Exception:
        logger.exception("close_device: synchronize_device failed. Continuing with device close.")

    ttnn._ttnn.device.close_device(device)


synchronize_device = ttnn._ttnn.device.synchronize_device
SetRootDir = ttnn._ttnn.device.SetRootDir
GetDefaultDevice = ttnn._ttnn.device.GetDefaultDevice
SetDefaultDevice = ttnn._ttnn.device.SetDefaultDevice
GetPCIeDeviceID = ttnn._ttnn.device.GetPCIeDeviceID
GetNumPCIeDevices = ttnn._ttnn.device.GetNumPCIeDevices


def is_wormhole_b0(device=None):
    if device is not None:
        return device.arch() == ttnn._ttnn.device.Arch.WORMHOLE_B0
    ARCH_NAME = ttnn._ttnn.device.get_arch_name()
    return "wormhole_b0" in ARCH_NAME


def is_blackhole(device=None):
    if device is not None:
        return device.arch() == ttnn._ttnn.device.Arch.BLACKHOLE
    ARCH_NAME = ttnn._ttnn.device.get_arch_name()
    return "blackhole" in ARCH_NAME


def CreateDevice(
    device_id: int,
    num_command_queues: int = 1,
    l1_small_size: int = ttnn._ttnn.device.DEFAULT_L1_SMALL_SIZE,
    trace_region_size: int = ttnn._ttnn.device.DEFAULT_TRACE_REGION_SIZE,
    dispatch_core_config: DispatchCoreConfig = None,
    *,
    worker_l1_size: int = ttnn._ttnn.device.DEFAULT_WORKER_L1_SIZE,
):
    return ttnn._ttnn.device.CreateDevice(
        device_id,
        num_command_queues,
        l1_small_size,
        trace_region_size,
        dispatch_core_config,
        worker_l1_size=worker_l1_size,
    )


def CreateDevices(
    device_ids: List[int],
    num_command_queues: int = 1,
    l1_small_size: int = ttnn._ttnn.device.DEFAULT_L1_SMALL_SIZE,
    trace_region_size: int = ttnn._ttnn.device.DEFAULT_TRACE_REGION_SIZE,
    dispatch_core_config: DispatchCoreConfig = None,
    *,
    worker_l1_size: int = ttnn._ttnn.device.DEFAULT_WORKER_L1_SIZE,
):
    return ttnn._ttnn.device.CreateDevices(
        device_ids,
        num_command_queues,
        l1_small_size,
        trace_region_size,
        dispatch_core_config,
        worker_l1_size=worker_l1_size,
    )


CloseDevice = ttnn._ttnn.device.CloseDevice
CloseDevices = ttnn._ttnn.device.CloseDevices


def ReadDeviceProfiler(device):
    ttnn._ttnn.device.ReadDeviceProfiler(device)


GetNumAvailableDevices = ttnn._ttnn.device.GetNumAvailableDevices
ClearKernelCache = ttnn._ttnn.device.ClearKernelCache
EnableMemoryReports = ttnn._ttnn.device.EnableMemoryReports
DisableMemoryReports = ttnn._ttnn.device.DisableMemoryReports
DeallocateBuffers = ttnn._ttnn.device.deallocate_buffers


@contextlib.contextmanager
def manage_device(device_id: int) -> "ttnn.device.Device":
    """
    Context manager for opening and closing a device.

    Args:
        device_id (int): The device ID to open.

    Returns:
        ttnn.device.Device: the opened device. The device will be closed automatically when the block is exited, even if an error occurs.

    Example:
        >>> with manage_device(device_id=0) as device:
            >>> # Perform operations with the device
            >>> tensor = ttnn.zeros((2, 3), device=device)
            >>> print(tensor)
        ttnn.Tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    """
    device = open_device(device_id=device_id)
    try:
        yield device
    finally:
        close_device(device)


initialize_fast_dispatch = ttnn._ttnn.device.initialize_fast_dispatch
terminate_fast_dispatch = ttnn._ttnn.device.terminate_fast_dispatch


@contextlib.contextmanager
def setup_fast_dispatch(device):
    """
    Context manager that enables Fast Dispatch for the duration of the block.
    The device must have been opened in Slow Dispatch mode (e.g. TT_METAL_SLOW_DISPATCH_MODE=1).
    On exit, Fast Dispatch is terminated and the device returns to Slow Dispatch.

    Args:
        device: The device to enable Fast Dispatch on.

    Yields:
        None: Use the device inside the block; it is in Fast Dispatch mode.

    Example:
        >>> mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
        >>> with ttnn.device.setup_fast_dispatch(mesh_device):
        ...     # issue writes or other FD operations
        ...     pass
        >>> # FD terminated; device is back in Slow Dispatch
    """
    initialize_fast_dispatch(device)
    try:
        yield
    finally:
        terminate_fast_dispatch(device)


def dump_device_memory_state(device, prefix=""):
    ttnn._ttnn.device.DumpDeviceMemoryState(device, prefix)


def get_memory_view(device, buffer_type):
    return ttnn._ttnn.device.GetMemoryView(device, buffer_type)


def get_allocator_base_address(device, buffer_type):
    """Return the lowest address (bytes) of the given allocator region.

    For ``ttnn.BufferType.L1`` this is the worker-L1 unreserved base; combined
    with the per-bank L1 size from :func:`get_memory_view`, callers can derive
    the absolute L1 top address used by the device-side allocator.
    """
    return ttnn._ttnn.device.GetAllocatorBaseAddress(device, buffer_type)


SubDevice = ttnn._ttnn.device.SubDevice
SubDeviceId = ttnn._ttnn.device.SubDeviceId
SubDeviceManagerId = ttnn._ttnn.device.SubDeviceManagerId

# Real-time profiler callbacks (experimental)
ProgramRealtimeRecord = ttnn._ttnn.device.ProgramRealtimeRecord
ProgramRealtimeRecordBatch = ttnn._ttnn.device.ProgramRealtimeRecordBatch
RegisterProgramRealtimeProfilerCallback = ttnn._ttnn.device.RegisterProgramRealtimeProfilerCallback
UnregisterProgramRealtimeProfilerCallback = ttnn._ttnn.device.UnregisterProgramRealtimeProfilerCallback
IsProgramRealtimeProfilerActive = ttnn._ttnn.device.IsProgramRealtimeProfilerActive

__all__ = [
    "ProgramRealtimeRecord",
    "ProgramRealtimeRecordBatch",
    "RegisterProgramRealtimeProfilerCallback",
    "UnregisterProgramRealtimeProfilerCallback",
    "IsProgramRealtimeProfilerActive",
]
