# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import contextlib
from typing import Optional, List
import os

import ttnn
from loguru import logger


def get_device_core_grid(device):
    compute_with_storage_grid_size = device.compute_with_storage_grid_size()
    return ttnn.types.CoreGrid(y=compute_with_storage_grid_size.y, x=compute_with_storage_grid_size.x)


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


def is_grayskull(device=None):
    if device is not None:
        return device.arch() == ttnn._ttnn.device.Arch.GRAYSKULL
    ARCH_NAME = ttnn._ttnn.device.get_arch_name()
    return "grayskull" in ARCH_NAME


def is_blackhole(device=None):
    if device is not None:
        return device.arch() == ttnn._ttnn.device.Arch.BLACKHOLE
    ARCH_NAME = ttnn._ttnn.device.get_arch_name()
    return "blackhole" in ARCH_NAME


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


def get_default_dispatch_core_axis(fabric_tensix_config=None):
    """Get default dispatch core axis, considering fabric tensix config if available."""
    if is_blackhole():
        # On Blackhole, if fabric tensix MUX is enabled, use ROW; otherwise use COL
        if fabric_tensix_config == ttnn.FabricTensixConfig.MUX:
            return DispatchCoreAxis.ROW
        else:
            return DispatchCoreAxis.COL
    else:
        # Non-Blackhole architectures default to ROW
        return DispatchCoreAxis.ROW


class DispatchCoreConfig(ttnn._ttnn.device.DispatchCoreConfig):
    def __init__(self, type: DispatchCoreType = None, axis: DispatchCoreAxis = None, fabric_tensix_config=None):
        # Validate user provided args
        if type:
            if not isinstance(type, DispatchCoreType):
                valid_values = [e for e in DispatchCoreType.__members__.values()]
                raise ValueError(f"Invalid dispatch core type: {type}. Valid values are: {valid_values}")
            if type == DispatchCoreType.ETH and axis == DispatchCoreAxis.COL:
                raise ValueError("COL axis is not supported for ETH dispatch core type")
        if axis:
            if not isinstance(axis, DispatchCoreAxis):
                valid_values = [e for e in DispatchCoreAxis.__members__.values()]
                raise ValueError(f"Invalid dispatch core axis: {axis}. Valid values are: {valid_values}")
            if axis == DispatchCoreAxis.ROW and is_blackhole() and fabric_tensix_config != ttnn.FabricTensixConfig.MUX:
                raise ValueError(
                    "ROW dispatch core axis is not supported for blackhole arch unless fabric tensix MUX is enabled"
                )
        if type and axis:
            # User provided both valid type and axis, check if they are compatible
            self.type = type
            self.axis = axis
        elif type:
            # User provided only valid type
            self.type = type
            self.axis = get_default_dispatch_core_axis(fabric_tensix_config)
            logger.debug(f"Using default dispatch core axis for this system: {self.axis}")
        elif axis:
            self.axis = axis
            # User provided only valid axis
            if self.axis == DispatchCoreAxis.COL:
                # COL axis is not supported for ETH dispatch core type, default to WORKER
                self.type = DispatchCoreType.WORKER
                logger.info(
                    f"{self.axis} axis is only supported on WORKER dispatch core type, defaulting to {self.type}"
                )
            elif self.axis == DispatchCoreAxis.ROW:
                # ROW axis is supported for all dispatch core types, use default type for their system
                self.type = get_default_dispatch_core_type()
                logger.debug(f"Using default dispatch core type for this system: {self.type}")
        else:
            # User provided no valid type or axis, use default for their system
            self.type = get_default_dispatch_core_type()
            logger.debug(f"Using default dispatch core type for this system: {self.type}")
            self.axis = get_default_dispatch_core_axis(fabric_tensix_config)
            logger.debug(f"Using default dispatch core axis for this system: {self.axis}")
        super().__init__(self.type, self.axis)


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
        dispatch_core_config or DispatchCoreConfig(),
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
        dispatch_core_config or DispatchCoreConfig(),
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


def dump_device_memory_state(device, prefix=""):
    ttnn._ttnn.device.DumpDeviceMemoryState(device, prefix)


def get_memory_view(device, buffer_type):
    return ttnn._ttnn.device.GetMemoryView(device, buffer_type)


pad_to_tile_shape = ttnn._ttnn.device.pad_to_tile_shape

SubDevice = ttnn._ttnn.device.SubDevice
SubDeviceId = ttnn._ttnn.device.SubDeviceId
SubDeviceManagerId = ttnn._ttnn.device.SubDeviceManagerId

__all__ = []
