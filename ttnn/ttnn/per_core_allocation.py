# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Experimental per-core allocation APIs."""

_NAMES = [
    "MemoryConfig",
    "set_per_core_allocation",
    "per_core_buffer_address",
]


def __getattr__(name):
    if name in _NAMES:
        import ttnn._ttnn.per_core_allocation as _mod

        val = getattr(_mod, name)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
