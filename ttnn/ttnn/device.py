# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import contextlib
from typing import Optional, List
import os

import ttnn
from loguru import logger


def get_device_core_grid(device):
    compute_with_storage_grid_size = device.compute_with_storage_grid_size()
    return ttnn.CoreGrid(y=compute_with_storage_grid_size.y, x=compute_with_storage_grid_size.x)


# TODO: Device = ttnn._ttnn.Device
Device = ttnn._ttnn.multi_device.MeshDevice
Device.core_grid = property(get_device_core_grid)
DispatchCoreType = ttnn._ttnn.device.DispatchCoreType
DispatchCoreAxis = ttnn._ttnn.device.DispatchCoreAxis
Arch = ttnn._ttnn.device.Arch
DEFAULT_L1_SMALL_SIZE = ttnn._ttnn.device.DEFAULT_L1_SMALL_SIZE
DEFAULT_TRACE_REGION_SIZE = ttnn._ttnn.device.DEFAULT_TRACE_REGION_SIZE
get_max_worker_l1_unreserved_size = ttnn._ttnn.device.get_max_worker_l1_unreserved_size

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


synchronize_device = ttnn._ttnn.device.synchronize_device
GetDefaultDevice = ttnn._ttnn.device.GetDefaultDevice
SetDefaultDevice = ttnn._ttnn.device.SetDefaultDevice
GetPCIeDeviceID = ttnn._ttnn.device.GetPCIeDeviceID
GetNumPCIeDevices = ttnn._ttnn.device.GetNumPCIeDevices


def get_default_dispatch_core_type():
    eth_default_dispatch_clusters = [
        ttnn._ttnn.cluster.ClusterType.N300,
        ttnn._ttnn.cluster.ClusterType.T3K,
        ttnn._ttnn.cluster.ClusterType.N300_2x2,
    ]
    return (
        ttnn._ttnn.device.DispatchCoreType.ETH
        if ttnn._ttnn.cluster.get_cluster_type() in eth_default_dispatch_clusters
        else ttnn._ttnn.device.DispatchCoreType.WORKER
    )


class DispatchCoreConfig(ttnn._ttnn.device.DispatchCoreConfig):
    def __init__(self, type: DispatchCoreType = None, axis: DispatchCoreAxis = None):
        if type:
            try:
                type = DispatchCoreType(type)
            except ValueError:
                valid_values = [e.value for e in ttnn._ttnn.device.DispatchCoreType]
                raise ValueError(f"Invalid dispatch core type: {type}. Valid values are: {valid_values}")
        if axis:
            try:
                axis = DispatchCoreAxis(axis)
            except ValueError:
                valid_values = [e.value for e in ttnn._ttnn.device.DispatchCoreAxis]
                raise ValueError(f"Invalid dispatch core axis: {axis}. Valid values are: {valid_values}")
        if type and axis:
            # User provided both valid type and axis, check if they are compatible
            if type == DispatchCoreType.ETH and axis == DispatchCoreAxis.COL:
                raise ValueError("ETH dispatch core type cannot be used with COL axis")
            super().__init__(type, axis)
        elif type:
            # User provided only valid type
            super().__init__(type)
        elif axis:
            # User provided only valid axis
            if axis == DispatchCoreAxis.COL:
                # COL axis is not supported for ETH dispatch core type, default to WORKER
                type = DispatchCoreType.WORKER
            elif axis == DispatchCoreAxis.ROW:
                # ROW axis is supported for all dispatch core types, use default type for their system
                type = get_default_dispatch_core_type()
            super().__init__(type, axis)
        else:
            # User provided no valid type or axis, use default for their system
            type = get_default_dispatch_core_type()
            super().__init__(type)


def CreateDevice(
    device_id: int,
    num_command_queues: int = 1,
    l1_small_size: int = ttnn._ttnn.device.DEFAULT_L1_SMALL_SIZE,
    trace_region_size: int = ttnn._ttnn.device.DEFAULT_TRACE_REGION_SIZE,
    dispatch_core_config: DispatchCoreConfig = DispatchCoreConfig(),
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
    dispatch_core_config: DispatchCoreConfig = DispatchCoreConfig(),
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
EnablePersistentKernelCache = ttnn._ttnn.device.EnablePersistentKernelCache
DisablePersistentKernelCache = ttnn._ttnn.device.DisablePersistentKernelCache
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


def dump_device_memory_state(device, prefix=""):
    ttnn._ttnn.device.DumpDeviceMemoryState(device, prefix)


def get_memory_view(device, buffer_type):
    return ttnn._ttnn.device.GetMemoryView(device, buffer_type)


def is_wormhole_b0(device=None):
    if device is not None:
        return device.arch() == ttnn._ttnn.device.Arch.WORMHOLE_B0
    ARCH_NAME = ttnn.get_arch_name()
    return "wormhole_b0" in ARCH_NAME


def is_grayskull(device=None):
    if device is not None:
        return device.arch() == ttnn._ttnn.device.Arch.GRAYSKULL
    ARCH_NAME = ttnn.get_arch_name()
    return "grayskull" in ARCH_NAME


def is_blackhole(device=None):
    if device is not None:
        return device.arch() == ttnn._ttnn.device.Arch.BLACKHOLE
    ARCH_NAME = ttnn.get_arch_name()
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
