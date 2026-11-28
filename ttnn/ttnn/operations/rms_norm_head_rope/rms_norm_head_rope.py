# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""rms_norm_head_rope — per-head RMSNorm (head_dim 128) + optional weight + optional interleaved-pair RoPE.

One native dispatch (ttnn.generic_op) per call. See op_design.md for the blocking model.
"""

from __future__ import annotations

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .rms_norm_head_rope_program_descriptor import HEAD_DIM, create_program_descriptor


def default_compute_kernel_config():
    """Fresh default compute config (HiFi4, fp32 DEST accumulate, no approx) — single source of truth."""
    return ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        math_approx_mode=False,
    )


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS
# ---------------------------------------------------------------------------


def tag_alignment(inputs, axes):
    """W must be whole heads (multiple of 128), R a multiple of the tile height."""
    shape = inputs[0]
    if len(shape) < 2:
        return "h_non_aligned"
    if shape[-1] % HEAD_DIM != 0:
        return "w_non_aligned"
    if shape[-2] % 32 != 0:
        return "h_non_aligned"
    return "tile_aligned"


def tag_rank(inputs, axes):
    return int(len(inputs[0]))


INPUT_TAGGERS = {
    "alignment": tag_alignment,
    "rank": tag_rank,
}


# ---------------------------------------------------------------------------
# 2. SUPPORTED
# ---------------------------------------------------------------------------

SUPPORTED = {
    "dtype": [ttnn.bfloat16],
    # False (bf16 DEST) is accepted per the op spec; TARGET only sweeps True.
    "fp32_dest_acc_en": [True, False],
    "layout": [ttnn.TILE_LAYOUT],
    "alignment": ["tile_aligned"],
    "rank": [4],
    "affine": ["no_affine", "weight"],
    "affine_dtype": [ttnn.bfloat16, "none"],
    "affine_layout": [ttnn.TILE_LAYOUT, "none"],
    "rope": ["no_rope", "rope_bcast"],
    "rope_dtype": [ttnn.bfloat16, "none"],
    "memory_layout": [ttnn.TensorMemoryLayout.INTERLEAVED],
}


# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------

EXCLUSIONS = []


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def _axes_for_call(input_tensor, weight, rope_cos, rope_sin, compute_kernel_config):
    if weight is not None:
        affine, affine_dtype, affine_layout = "weight", weight.dtype, weight.layout
    else:
        affine, affine_dtype, affine_layout = "no_affine", "none", "none"
    if rope_cos is not None and rope_sin is not None:
        rope, rope_dtype = "rope_bcast", rope_cos.dtype
    else:
        rope, rope_dtype = "no_rope", "none"
    axes = {
        "dtype": input_tensor.dtype,
        "fp32_dest_acc_en": bool(compute_kernel_config.fp32_dest_acc_en),
        "layout": input_tensor.layout,
        "affine": affine,
        "affine_dtype": affine_dtype,
        "affine_layout": affine_layout,
        "rope": rope,
        "rope_dtype": rope_dtype,
        "memory_layout": input_tensor.memory_config().memory_layout,
    }
    shape = tuple(input_tensor.shape)
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger((shape,), axes)
    return axes


def _check_aux(name, t, shape):
    if tuple(t.shape) != tuple(shape):
        raise ValueError(f"rms_norm_head_rope: {name} shape {tuple(t.shape)} != expected {tuple(shape)}")
    if t.dtype != ttnn.bfloat16:
        raise ValueError(f"rms_norm_head_rope: {name} dtype must be bfloat16, got {t.dtype}")
    if t.layout != ttnn.TILE_LAYOUT:
        raise ValueError(f"rms_norm_head_rope: {name} must be TILE layout")
    if not ttnn.is_tensor_storage_on_device(t):
        raise ValueError(f"rms_norm_head_rope: {name} must be on device")
    if t.memory_config().memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
        raise ValueError(f"rms_norm_head_rope: {name} must be interleaved")


def validate(
    input_tensor,
    weight=None,
    *,
    rope_cos=None,
    rope_sin=None,
    transformation_mat=None,
    compute_kernel_config=None,
):
    if (rope_cos is None) != (rope_sin is None):
        raise ValueError("rms_norm_head_rope: rope_cos and rope_sin must be supplied together")

    cfg = compute_kernel_config if compute_kernel_config is not None else default_compute_kernel_config()
    axes = _axes_for_call(input_tensor, weight, rope_cos, rope_sin, cfg)

    # 1. SUPPORTED — per-axis
    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"rms_norm_head_rope: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")

    # 2. EXCLUSIONS — cell-level
    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"rms_norm_head_rope: unsupported combination (refinement candidate): {exc}")

    # Structural / interface checks beyond the registry axes.
    if input_tensor.memory_config().buffer_type != ttnn.BufferType.DRAM:
        raise ValueError("rms_norm_head_rope: input must be DRAM interleaved")
    shape = tuple(input_tensor.shape)
    if shape[0] != 1 or shape[1] != 1:
        raise ValueError(f"rms_norm_head_rope: input must be (1, 1, R, W), got {shape}")
    R, W = shape[-2], shape[-1]
    if weight is not None:
        _check_aux("weight", weight, (1, 1, 1, W))
    if rope_cos is not None:
        _check_aux("rope_cos", rope_cos, (1, 1, R, HEAD_DIM))
        _check_aux("rope_sin", rope_sin, (1, 1, R, HEAD_DIM))
    if transformation_mat is not None:
        if tuple(transformation_mat.shape) != (1, 1, 32, 32):
            raise ValueError("rms_norm_head_rope: transformation_mat must be (1, 1, 32, 32)")
        if transformation_mat.dtype != ttnn.bfloat16 or transformation_mat.layout != ttnn.TILE_LAYOUT:
            raise ValueError("rms_norm_head_rope: transformation_mat must be bfloat16 TILE")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def rms_norm_head_rope(
    input_tensor: ttnn.Tensor,
    weight: ttnn.Tensor = None,
    *,
    rope_cos: ttnn.Tensor = None,
    rope_sin: ttnn.Tensor = None,
    transformation_mat: ttnn.Tensor = None,
    epsilon: float = 1e-6,
    compute_kernel_config: ttnn.ComputeConfigDescriptor = None,
) -> ttnn.Tensor:
    validate(
        input_tensor,
        weight,
        rope_cos=rope_cos,
        rope_sin=rope_sin,
        transformation_mat=transformation_mat,
        compute_kernel_config=compute_kernel_config,
    )
    cfg = compute_kernel_config if compute_kernel_config is not None else default_compute_kernel_config()

    device = input_tensor.device()
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(input_tensor.shape)),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        device,
        ttnn.DRAM_MEMORY_CONFIG,
    )

    # transformation_mat is validated and ignored: the rotation tile is generated in L1 by the reader.
    program_descriptor = create_program_descriptor(
        input_tensor, output_tensor, weight, rope_cos, rope_sin, float(epsilon), cfg
    )

    io_tensors = [input_tensor]
    if weight is not None:
        io_tensors.append(weight)
    if rope_cos is not None:
        io_tensors += [rope_cos, rope_sin]
    io_tensors.append(output_tensor)  # output must be last
    return ttnn.generic_op(io_tensors, program_descriptor)
