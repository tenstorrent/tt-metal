# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Plain-data serializers for memory and compute configs."""

from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING, Any


class _OptionalModule:
    """Attribute access reads a module already in sys.modules, else Any."""

    def __init__(self, name: str) -> None:
        self._name = name

    def __getattr__(self, attr: str) -> Any:
        module = sys.modules.get(self._name)
        if module is None:
            return Any
        return getattr(module, attr, Any)


if TYPE_CHECKING:
    import ttnn
else:
    # Hints stay resolvable without importing ttnn (cycle through decorators).
    ttnn = _OptionalModule("ttnn")


def memory_config_to_dict(memory_config: ttnn.MemoryConfig):
    # Convert to plain types for deterministic serialization.
    return {
        "memory_layout": str(memory_config.memory_layout),
        "buffer_type": str(memory_config.buffer_type),
        "shard_spec": str(memory_config.shard_spec),
        "is_sharded": bool(memory_config.is_sharded()),
        "interleaved": bool(memory_config.interleaved),
        "hash": int(memory_config.__hash__()),
    }


def compute_kernel_config_to_dict(compute_kernel_config: ttnn.WormholeComputeKernelConfig):
    return {
        "math_fidelity": str(compute_kernel_config.math_fidelity),
        "math_approx_mode": str(compute_kernel_config.math_approx_mode),
        "fp32_dest_acc_en": bool(compute_kernel_config.fp32_dest_acc_en),
        "packer_l1_acc": bool(compute_kernel_config.packer_l1_acc),
        "dst_full_sync_en": bool(compute_kernel_config.dst_full_sync_en),
        "throttle_level": str(compute_kernel_config.throttle_level),
    }


def program_config_to_dict(program_config):
    if hasattr(program_config, "to_json"):
        d = json.loads(program_config.to_json())
        d["type"] = type(program_config).__name__
        return d
    else:
        return {"type": type(program_config).__name__, "repr": repr(program_config)}
