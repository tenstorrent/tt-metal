# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""layer_norm over the last dim of a TILE tensor with gamma/beta, one generic_op dispatch per call."""

from __future__ import annotations

import math

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .layer_norm_tile_program_descriptor import (
    TILE,
    W_TILES_RESIDENT_MAX,
    affine_half_row_bytes,
    create_program_descriptor,
    select_blocking,
)


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS
# ---------------------------------------------------------------------------


def tag_alignment(inputs, axes):
    shape = inputs[0]
    if shape[-1] % TILE != 0:
        return "w_non_aligned"
    if shape[-2] % TILE != 0:
        return "h_non_aligned"
    return "tile_aligned"


INPUT_TAGGERS = {
    "alignment": tag_alignment,
}


# ---------------------------------------------------------------------------
# 2. SUPPORTED
# ---------------------------------------------------------------------------

SUPPORTED = {
    "dtype": [ttnn.bfloat16],
    "fp32_dest_acc_en": [True, False],
    "layout": [ttnn.TILE_LAYOUT],
    "alignment": ["tile_aligned", "h_non_aligned"],
    "affine_layout": [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
    "memory_layout": [ttnn.TensorMemoryLayout.INTERLEAVED],
}


# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------

EXCLUSIONS = []


PROPERTIES = {
    "multi_core": {"value": True, "source": "declared"},
    "bounded_cb": {"value": True, "source": "declared"},
}


def default_compute_kernel_config():
    return ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        math_approx_mode=False,
    )


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def _validate_affine(name, affine, width, input_layout_axes):
    if affine.dtype != ttnn.bfloat16:
        raise UnsupportedAxisValue(f"layer_norm: {name} dtype={affine.dtype!r} not in SUPPORTED [bfloat16]")
    shape = list(affine.shape)
    if math.prod(shape) != width or width % shape[-1] != 0:
        raise ValueError(f"layer_norm: {name} shape {shape} does not match input width {width}")
    if affine.layout == ttnn.TILE_LAYOUT and shape[-1] != width:
        raise ValueError(f"layer_norm: TILE {name} must carry W on its last dim, got shape {shape}")
    if affine.memory_config().memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
        raise UnsupportedAxisValue(f"layer_norm: {name} must be interleaved")
    if affine.layout == ttnn.ROW_MAJOR_LAYOUT:
        # Row-0 halves are read at stick offsets o and o + half_row; both must be NoC-aligned.
        alignment = (
            ttnn.get_dram_alignment()
            if affine.memory_config().buffer_type == ttnn.BufferType.DRAM
            else ttnn.get_l1_alignment()
        )
        if shape[-1] % TILE != 0 or affine_half_row_bytes(affine) % alignment != 0:
            raise UnsupportedAxisValue(
                f"layer_norm: ROW_MAJOR {name} needs last dim % 32 == 0 and 16-element halves aligned to "
                f"{alignment} B (shape {shape}, {affine.memory_config().buffer_type})"
            )
    input_layout_axes.append(affine.layout)


def validate(input_tensor, gamma, beta, *, compute_kernel_config):
    shape = list(input_tensor.shape)
    if len(shape) < 2:
        raise ValueError(f"layer_norm: input rank must be >= 2, got shape {shape}")
    width = shape[-1]

    affine_layouts = []
    _validate_affine("gamma", gamma, width, affine_layouts)
    _validate_affine("beta", beta, width, affine_layouts)

    axes = {
        "dtype": input_tensor.dtype,
        "layout": input_tensor.layout,
        "fp32_dest_acc_en": bool(compute_kernel_config.fp32_dest_acc_en),
        "memory_layout": input_tensor.memory_config().memory_layout,
    }
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger((shape,), axes)

    for axis, allowed in SUPPORTED.items():
        if axis == "affine_layout":
            for layout in affine_layouts:
                if layout not in allowed:
                    raise UnsupportedAxisValue(f"layer_norm: affine_layout={layout!r} not in SUPPORTED {allowed}")
            continue
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"layer_norm: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")

    for affine_layout in affine_layouts:
        cell = dict(axes, affine_layout=affine_layout)
        for exc in EXCLUSIONS:
            if all(cell.get(k) == v for k, v in exc.items()):
                raise ExcludedCell(f"layer_norm: unsupported combination (refinement candidate): {exc}")

    # Regime row-split-resident-w: a whole row of tiles must stay resident in L1.
    if -(-width // TILE) > W_TILES_RESIDENT_MAX:
        raise UnsupportedAxisValue(
            f"layer_norm: ceil(W/32)={-(-width // TILE)} exceeds W_TILES_RESIDENT_MAX={W_TILES_RESIDENT_MAX}"
        )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def layer_norm(
    input_tensor: ttnn.Tensor,
    gamma: ttnn.Tensor,
    beta: ttnn.Tensor,
    *,
    epsilon: float = 1e-5,
    compute_kernel_config: ttnn.ComputeConfigDescriptor = None,
) -> ttnn.Tensor:
    if compute_kernel_config is None:
        compute_kernel_config = default_compute_kernel_config()
    if input_tensor.dtype not in SUPPORTED["dtype"]:
        raise ValueError(f"layer_norm: input dtype {input_tensor.dtype!r} is not supported (bfloat16 only)")
    validate(input_tensor, gamma, beta, compute_kernel_config=compute_kernel_config)

    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(input_tensor.shape)),
        input_tensor.dtype,
        input_tensor.layout,
        input_tensor.device(),
        input_tensor.memory_config(),
    )

    blocking = select_blocking(input_tensor)
    program_descriptor = create_program_descriptor(
        input_tensor,
        gamma,
        beta,
        output_tensor,
        blocking,
        epsilon=epsilon,
        compute_kernel_config=compute_kernel_config,
    )
    return ttnn.generic_op([input_tensor, gamma, beta, output_tensor], program_descriptor)
