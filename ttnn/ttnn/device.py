# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import contextlib

import tt_lib as ttl
import ttnn


def get_device_core_grid(device):
    compute_with_storage_grid_size = device.compute_with_storage_grid_size()
    return ttnn.CoreGrid(y=compute_with_storage_grid_size.y, x=compute_with_storage_grid_size.x)


Device = ttl.device.Device
Device.core_grid = property(get_device_core_grid)

DEVICES = {}


def open_device(*, device_id: int):
    """
    open_device(device_id: int) -> ttnn.Device:

    Open a device with the given device_id. If the device is already open, return the existing device.
    """
    if device_id in DEVICES:
        return DEVICES[device_id]
    device = ttl.device.CreateDevice(device_id)
    DEVICES[device_id] = device
    return device


def close_device(device):
    """
    close_device(device: ttnn.Device) -> None:

    Close the device and remove it from the device cache.
    """
    ttl.device.Synchronize(device)
    ttl.device.CloseDevice(device)
    del DEVICES[device.id()]


@contextlib.contextmanager
def manage_device(*, device_id: int):
    """
    manage_device(device_id: int) -> ttnn.Device:

    Context manager for opening and closing a device.
    """
    device = open_device(device_id=device_id)
    try:
        yield device
    finally:
        close_device(device)


def dump_device_memory_state(device):
    ttl.device.DumpDeviceMemoryState(device)


__all__ = []
