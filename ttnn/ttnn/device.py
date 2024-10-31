# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import contextlib
from typing import Optional, List

import ttnn
import os


def get_device_core_grid(device):
    compute_with_storage_grid_size = device.compute_with_storage_grid_size()
    return ttnn.CoreGrid(y=compute_with_storage_grid_size.y, x=compute_with_storage_grid_size.x)


# TODO: Device = ttnn._ttnn.Device
Device = ttnn._ttnn.device.Device
Device.core_grid = property(get_device_core_grid)
DispatchCoreType = ttnn._ttnn.device.DispatchCoreType
DispatchCoreAxis = ttnn._ttnn.device.DispatchCoreAxis
DispatchCoreConfig = ttnn._ttnn.device.DispatchCoreConfig
Arch = ttnn._ttnn.device.Arch
EPS_GS = ttnn._ttnn.device.EPS_GS
EPS_WHB0 = ttnn._ttnn.device.EPS_WHB0
EPS_BH = ttnn._ttnn.device.EPS_BH
DEFAULT_L1_SMALL_SIZE = ttnn._ttnn.device.DEFAULT_L1_SMALL_SIZE
DEFAULT_TRACE_REGION_SIZE = ttnn._ttnn.device.DEFAULT_TRACE_REGION_SIZE

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
    synchronize_device(device)
    ttnn._ttnn.device.close_device(device)


enable_program_cache = ttnn._ttnn.device.enable_program_cache
disable_and_clear_program_cache = ttnn._ttnn.device.disable_and_clear_program_cache

synchronize_device = ttnn._ttnn.device.synchronize_device
GetDefaultDevice = ttnn._ttnn.device.GetDefaultDevice
SetDefaultDevice = ttnn._ttnn.device.SetDefaultDevice
GetPCIeDeviceID = ttnn._ttnn.device.GetPCIeDeviceID
GetNumPCIeDevices = ttnn._ttnn.device.GetNumPCIeDevices


def CreateDevice(
    device_id: int,
    num_command_queues: int = 1,
    l1_small_size: int = ttnn._ttnn.device.DEFAULT_L1_SMALL_SIZE,
    trace_region_size: int = ttnn._ttnn.device.DEFAULT_TRACE_REGION_SIZE,
    dispatch_core_config: DispatchCoreConfig = ttnn._ttnn.device.DispatchCoreConfig(),
):
    return ttnn._ttnn.device.CreateDevice(
        device_id, num_command_queues, l1_small_size, trace_region_size, dispatch_core_config
    )


def CreateDevices(
    device_ids: List[int],
    num_command_queues: int = 1,
    l1_small_size: int = ttnn._ttnn.device.DEFAULT_L1_SMALL_SIZE,
    trace_region_size: int = ttnn._ttnn.device.DEFAULT_TRACE_REGION_SIZE,
    dispatch_core_config: DispatchCoreConfig = ttnn._ttnn.device.DispatchCoreConfig(),
):
    return ttnn._ttnn.device.CreateDevices(
        device_ids, num_command_queues, l1_small_size, trace_region_size, dispatch_core_config
    )


CloseDevice = ttnn._ttnn.device.CloseDevice
CloseDevices = ttnn._ttnn.device.CloseDevices


def DumpDeviceProfiler(device):
    ttnn._ttnn.device.DumpDeviceProfiler(device)


GetNumAvailableDevices = ttnn._ttnn.device.GetNumAvailableDevices
EnablePersistentKernelCache = ttnn._ttnn.device.EnablePersistentKernelCache
DisablePersistentKernelCache = ttnn._ttnn.device.DisablePersistentKernelCache
EnableCompilationReports = ttnn._ttnn.device.EnableCompilationReports
DisableCompilationReports = ttnn._ttnn.device.DisableCompilationReports
EnableMemoryReports = ttnn._ttnn.device.EnableMemoryReports
DisableMemoryReports = ttnn._ttnn.device.DisableMemoryReports
SetLazyCommandQueueMode = ttnn._ttnn.device.SetLazyCommandQueueMode
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


def dump_device_memory_state(device, prefix=""):
    ttnn._ttnn.device.DumpDeviceMemoryState(device, prefix)


def is_wormhole_b0(device=None):
    if device is not None:
        return device.arch() == ttnn._ttnn.device.Arch.WORMHOLE_B0
    ARCH_NAME = os.environ.get("ARCH_NAME", os.environ.get("TT_ARCH_NAME", "")).lower()
    return "wormhole_b0" in ARCH_NAME


def is_grayskull(device=None):
    if device is not None:
        return device.arch() == ttnn._ttnn.device.Arch.GRAYSKULL
    ARCH_NAME = os.environ.get("ARCH_NAME", os.environ.get("TT_ARCH_NAME", "")).lower()
    return "grayskull" in ARCH_NAME


def is_blackhole(device=None):
    if device is not None:
        return device.arch() == ttnn._ttnn.device.Arch.BLACKHOLE
    ARCH_NAME = os.environ.get("ARCH_NAME", os.environ.get("TT_ARCH_NAME", "")).lower()
    return "blackhole" in ARCH_NAME


SetDefaultDevice = ttnn._ttnn.device.SetDefaultDevice
GetDefaultDevice = ttnn._ttnn.device.GetDefaultDevice
format_input_tensor = ttnn._ttnn.device.format_input_tensor
format_output_tensor = ttnn._ttnn.device.format_output_tensor
pad_to_tile_shape = ttnn._ttnn.device.pad_to_tile_shape

SubDevice = ttnn._ttnn.device.SubDevice
SubDeviceId = ttnn._ttnn.device.SubDeviceId
SubDeviceManagerId = ttnn._ttnn.device.SubDeviceManagerId

DefaultQueueId = ttnn._ttnn.device.DefaultQueueId

__all__ = []
