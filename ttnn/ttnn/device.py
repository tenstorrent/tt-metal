# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import tt_lib as ttl

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
