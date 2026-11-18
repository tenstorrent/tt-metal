# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import ttnn
from .program_descriptor import create_program_descriptor


def _allocate_dram(device, rows, columns):
    return ttnn.allocate_tensor_on_device(
        ttnn.Shape([rows, columns]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )


def qwen_mlp_partial(h, w_norm, w_gate_up, w_down):
    rows, hidden = h.shape
    device = h.device()
    normalized = _allocate_dram(device, rows, hidden)
    activated = _allocate_dram(device, rows, w_down.shape[0])
    partial_y = _allocate_dram(device, rows, hidden)
    program_descriptor = create_program_descriptor(h, w_norm, w_gate_up, w_down, normalized, activated, partial_y)
    ttnn.generic_op([h, w_norm, w_gate_up, w_down, normalized, activated, partial_y], program_descriptor)
    return partial_y
