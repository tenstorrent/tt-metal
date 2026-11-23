# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""wq_b_linear: y = x @ W, x [1,1,160,1536] bf16, W [1536,4096] bf8_b -> y [1,1,160,4096] bf16 (L1).

One fused device program (ttnn.generic_op): an x-sender core reads x once from DRAM and
multicasts it to 64 worker cores; each worker streams its own W columns from DRAM, runs
matmul_block over the whole K, and writes its output block to the L1-interleaved y.
"""

from __future__ import annotations

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .wq_b_linear_program_descriptor import create_program_descriptor

X_SHAPE = [1, 1, 160, 1536]
W_SHAPE = [1536, 4096]

# 1. INPUT_TAGGERS — one shape, no shape-derived axis.
INPUT_TAGGERS = {}

# 2. SUPPORTED — equals the golden TARGET (one cell).
SUPPORTED = {
    "dtype": [ttnn.bfloat16],
    "weight_dtype": [ttnn.bfloat8_b],
    "layout": [ttnn.TILE_LAYOUT],
    "fp32_dest_acc_en": [False],
}

# 3. EXCLUSIONS
EXCLUSIONS = []


def default_compute_kernel_config() -> ttnn.ComputeConfigDescriptor:
    return ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        fp32_dest_acc_en=False,
        math_approx_mode=False,
    )


def _is_dram_interleaved(t) -> bool:
    mc = t.memory_config()
    return mc.buffer_type == ttnn.BufferType.DRAM and mc.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED


# 4. validate()
def validate(input_tensor, weight, *, compute_kernel_config=None):
    cfg = compute_kernel_config if compute_kernel_config is not None else default_compute_kernel_config()
    axes = {
        "dtype": input_tensor.dtype,
        "weight_dtype": weight.dtype,
        "layout": input_tensor.layout,
        "fp32_dest_acc_en": bool(cfg.fp32_dest_acc_en),
    }
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger((list(input_tensor.shape), list(weight.shape)), axes)

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"wq_b_linear: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")
    if weight.layout != ttnn.TILE_LAYOUT:
        raise UnsupportedAxisValue(f"wq_b_linear: weight layout {weight.layout!r} not TILE_LAYOUT")
    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"wq_b_linear: unsupported combination: {exc}")

    # Tensor-shape / placement contract.
    if list(input_tensor.shape) != X_SHAPE:
        raise ValueError(f"wq_b_linear: x shape {list(input_tensor.shape)} != {X_SHAPE}")
    if list(weight.shape) != W_SHAPE:
        raise ValueError(f"wq_b_linear: W shape {list(weight.shape)} != {W_SHAPE}")
    if not _is_dram_interleaved(input_tensor):
        raise ValueError("wq_b_linear: x must be DRAM interleaved")
    if not _is_dram_interleaved(weight):
        raise ValueError("wq_b_linear: W must be DRAM interleaved")


def wq_b_linear(
    input_tensor: ttnn.Tensor,
    weight: ttnn.Tensor,
    *,
    compute_kernel_config: ttnn.ComputeConfigDescriptor = None,
) -> ttnn.Tensor:
    validate(input_tensor, weight, compute_kernel_config=compute_kernel_config)
    cfg = compute_kernel_config if compute_kernel_config is not None else default_compute_kernel_config()

    device = input_tensor.device()
    out_shape = [*list(input_tensor.shape)[:-1], list(weight.shape)[-1]]
    output = ttnn.allocate_tensor_on_device(
        ttnn.Shape(out_shape),
        input_tensor.dtype,
        ttnn.TILE_LAYOUT,
        device,
        ttnn.L1_MEMORY_CONFIG,
    )
    descriptor = create_program_descriptor(input_tensor, weight, output, cfg)
    return ttnn.generic_op([input_tensor, weight, output], descriptor)
