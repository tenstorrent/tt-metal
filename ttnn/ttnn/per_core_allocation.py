# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Experimental per-core L1 allocation APIs.

These APIs are not part of the stable ttnn public interface. They may change
or be removed without notice.
"""


def __getattr__(name):
    import ttnn

    _mod = ttnn._ttnn.per_core_allocation
    _exports = {
        "AllocatorMode": _mod.AllocatorMode,
        "CreateDevice": _mod.CreateDevice,
        "CreateDevices": _mod.CreateDevices,
        "open_mesh_device": _mod.open_mesh_device,
        "MemoryConfig": _mod.MemoryConfig,
        "per_core_buffer_address": _mod.per_core_buffer_address,
    }
    if name in _exports:
        return _exports[name]
    raise AttributeError(f"module 'ttnn.per_core_allocation' has no attribute {name!r}")
