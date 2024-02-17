# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import tt_lib as ttl

import ttnn


def get_device_core_grid(device):
    compute_with_storage_grid_size = device.compute_with_storage_grid_size()
    return ttnn.CoreGrid(y=compute_with_storage_grid_size.y, x=compute_with_storage_grid_size.x)


Device = ttl.device.Device
Device.core_grid = property(get_device_core_grid)

DEVICES = {}


def open(device_id: int):
    if device_id in DEVICES:
        return DEVICES[device_id]
    device = ttl.device.CreateDevice(device_id)
    DEVICES[device_id] = device
    return device


def close(device):
    ttl.device.CloseDevice(device)
    del DEVICES[device.id()]


__all__ = []
