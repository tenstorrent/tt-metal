# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""wq_b_linear: y = x @ W for the DSA-indexer wq_b projection, one fused device program."""

from __future__ import annotations

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .wq_b_linear_program_descriptor import MIN_CORES, create_program_descriptor

X_SHAPE = [1, 1, 160, 1536]
W_SHAPE = [1536, 4096]

INPUT_TAGGERS = {}

SUPPORTED = {
    "dtype": [ttnn.bfloat16],
    "weight_dtype": [ttnn.bfloat8_b],
    "layout": [ttnn.TILE_LAYOUT],
    "fp32_dest_acc_en": [False],
}

EXCLUSIONS = []


def default_compute_kernel_config():
    return ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        fp32_dest_acc_en=False,
        math_approx_mode=False,
    )


def _is_dram_interleaved(tensor):
    mc = tensor.memory_config()
    return mc.buffer_type == ttnn.BufferType.DRAM and mc.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED


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
    if weight.layout not in SUPPORTED["layout"]:
        raise UnsupportedAxisValue(
            f"wq_b_linear: weight layout={weight.layout!r} not in SUPPORTED {SUPPORTED['layout']}"
        )

    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"wq_b_linear: unsupported combination: {exc}")

    if list(input_tensor.shape) != X_SHAPE:
        raise ValueError(f"wq_b_linear: x shape {list(input_tensor.shape)} != {X_SHAPE}")
    if list(weight.shape) != W_SHAPE:
        raise ValueError(f"wq_b_linear: W shape {list(weight.shape)} != {W_SHAPE}")
    if not _is_dram_interleaved(input_tensor):
        raise ValueError("wq_b_linear: x must be DRAM interleaved")
    if not _is_dram_interleaved(weight):
        raise ValueError("wq_b_linear: W must be DRAM interleaved")

    grid = input_tensor.device().compute_with_storage_grid_size()
    if grid.x * grid.y < MIN_CORES:
        raise RuntimeError(f"wq_b_linear: needs >= {MIN_CORES} worker cores, device has {grid.x * grid.y}")
    return cfg


def wq_b_linear(
    input_tensor: ttnn.Tensor,
    weight: ttnn.Tensor,
    *,
    compute_kernel_config: ttnn.ComputeConfigDescriptor = None,
) -> ttnn.Tensor:
    cfg = validate(input_tensor, weight, compute_kernel_config=compute_kernel_config)

    device = input_tensor.device()
    output_shape = list(input_tensor.shape)[:-1] + [weight.shape[-1]]
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(output_shape),
        input_tensor.dtype,
        ttnn.TILE_LAYOUT,
        device,
        ttnn.L1_MEMORY_CONFIG,
    )
    program_descriptor = create_program_descriptor(input_tensor, weight, output_tensor, cfg)
    return ttnn.generic_op([input_tensor, weight, output_tensor], program_descriptor)
