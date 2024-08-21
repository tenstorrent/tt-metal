# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import contextlib
import os
from typing import Optional, List

import ttnn


def get_device_core_grid(device):
    compute_with_storage_grid_size = device.compute_with_storage_grid_size()
    return ttnn.CoreGrid(y=compute_with_storage_grid_size.y, x=compute_with_storage_grid_size.x)


# TODO: Device = ttnn._ttnn.Device
Device = ttnn._ttnn.device.Device
Device.core_grid = property(get_device_core_grid)
DispatchCoreType = ttnn._ttnn.device.DispatchCoreType
Arch = ttnn._ttnn.device.Arch
EPS_GS = ttnn._ttnn.device.EPS_GS
EPS_WHB0 = ttnn._ttnn.device.EPS_WHB0
EPS_BH = ttnn._ttnn.device.EPS_BH


def open_device(
    device_id: int,
    l1_small_size: int = ttnn._ttnn.device.DEFAULT_L1_SMALL_SIZE,
    trace_region_size: int = ttnn._ttnn.device.DEFAULT_TRACE_REGION_SIZE,
    dispatch_core_type: int = DispatchCoreType.WORKER,
):
    """
    open_device(device_id: int) -> ttnn.Device:

    Open a device with the given device_id. If the device is already open, return the existing device.
    """
    return ttnn._ttnn.device.open_device(
        device_id=device_id,
        l1_small_size=l1_small_size,
        trace_region_size=trace_region_size,
        dispatch_core_type=dispatch_core_type,
    )


def close_device(device):
    """
    close_device(device: ttnn.Device) -> None:

    Close the device and remove it from the device cache.
    """
    synchronize_device(device)
    ttnn._ttnn.device.close_device(device)


def enable_program_cache(device):
    ttnn._ttnn.device.enable_program_cache(device)


def disable_and_clear_program_cache(device):
    ttnn._ttnn.device.disable_and_clear_program_cache(device)


def synchronize_device(device: "ttnn.Device", queue_id: Optional[int] = None) -> None:
    """
    synchronize_device(device: ttnn.Device, queue_id: Optional[int] = None) -> None:

    Synchronize the device with host by waiting for all operations to complete.
    If queue_id is provided then only the operations associated with that queue_id are waited for,
    otherwise operations for all command queues are waited on.
    """
    ttnn._ttnn.device.Synchronize(device, queue_id)


def GetDefaultDevice():
    return ttnn._ttnn.device.GetDefaultDevice()


def SetDefaultDevice(device):
    ttnn._ttnn.device.SetDefaultDevice(device)


def GetPCIeDeviceID(device_id):
    return ttnn._ttnn.device.GetPCIeDeviceID(device_id)


def GetNumPCIeDevices():
    return ttnn._ttnn.device.GetNumPCIeDevices()


def CreateDevice(
    device_id: int,
    num_hw_cqs: int = 1,
    l1_small_size: int = ttnn._ttnn.device.DEFAULT_L1_SMALL_SIZE,
    trace_region_size: int = ttnn._ttnn.device.DEFAULT_TRACE_REGION_SIZE,
    dispatch_core_type: int = DispatchCoreType.WORKER,
):
    return ttnn._ttnn.device.CreateDevice(device_id, num_hw_cqs, l1_small_size, trace_region_size, dispatch_core_type)


def CreateDevices(
    device_ids: List[int],
    num_hw_cqs: int = 1,
    l1_small_size: int = ttnn._ttnn.device.DEFAULT_L1_SMALL_SIZE,
    trace_region_size: int = ttnn._ttnn.device.DEFAULT_TRACE_REGION_SIZE,
    dispatch_core_type: int = DispatchCoreType.WORKER,
):
    return ttnn._ttnn.device.CreateDevices(device_ids, num_hw_cqs, l1_small_size, trace_region_size, dispatch_core_type)


def CloseDevice(device):
    ttnn._ttnn.device.CloseDevice(device)


def CloseDevices(devices):
    ttnn._ttnn.device.CloseDevices(devices)


def DumpDeviceProfiler(device, last_dump: bool = False):
    ttnn._ttnn.device.DumpDeviceProfiler(device, last_dump)


def GetNumAvailableDevices():
    return ttnn._ttnn.device.GetNumAvailableDevices()


def EnablePersistentKernelCache():
    return ttnn._ttnn.device.EnablePersistentKernelCache()


def DisablePersistentKernelCache():
    return ttnn._ttnn.device.DisablePersistentKernelCache()


def EnableCompilationReports():
    return ttnn._ttnn.device.EnableCompilationReports()


def DisableCompilationReports():
    return ttnn._ttnn.device.DisableCompilationReports()


def EnableMemoryReports():
    return ttnn._ttnn.device.EnableMemoryReports()


def DisableMemoryReports():
    return ttnn._ttnn.device.DisableMemoryReports()


def SetLazyCommandQueueMode(lazy: bool):
    ttnn._ttnn.device.SetLazyCommandQueueMode(lazy)


def DeallocateBuffers(device):
    ttnn._ttnn.device.deallocate_buffers(device)


@contextlib.contextmanager
def manage_device(device_id: int):
    """
    manage_device(device_id: int) -> ttnn.Device:

    Context manager for opening and closing a device.
    """
    device = open_device(device_id=device_id)
    try:
        yield device
    finally:
        close_device(device)


def dump_device_memory_state(device, prefix=""):
    ttnn._ttnn.device.DumpDeviceMemoryState(device, prefix)


def is_wormhole_b0(device):
    return device.arch() == ttnn._ttnn.device.Arch.WORMHOLE_B0


def is_grayskull(device):
    return device.arch() == ttnn._ttnn.device.Arch.GRAYSKULL


__all__ = []
