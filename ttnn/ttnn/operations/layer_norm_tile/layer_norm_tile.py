# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""layer_norm: y = LayerNorm(x [+ residual]) * gamma + beta over the last dim, one generic_op dispatch per call."""

from __future__ import annotations

import math

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .layer_norm_tile_program_descriptor import TILE, create_program_descriptor, select_blocking


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
    "residual": ["residual", "no_residual"],
    "affine_dtype": [ttnn.bfloat16],
    "affine_layout": [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT],
    "memory_layout": [ttnn.TensorMemoryLayout.INTERLEAVED],
    "buffer_type": [ttnn.BufferType.DRAM, ttnn.BufferType.L1],
    "affine_buffer_type": [ttnn.BufferType.DRAM, ttnn.BufferType.L1],
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


def _check_contract(input_tensor, gamma, beta, residual, memory_config):
    """Malformed-call checks (ValueError), op_design.md validation steps 1-6."""
    if input_tensor.dtype != ttnn.bfloat16:
        raise ValueError(f"layer_norm: input dtype must be bfloat16, got {input_tensor.dtype}")
    if input_tensor.layout != ttnn.TILE_LAYOUT:
        raise ValueError(f"layer_norm: input layout must be TILE, got {input_tensor.layout}")
    shape = list(input_tensor.shape)
    if len(shape) < 2:
        raise ValueError(f"layer_norm: input rank must be >= 2, got shape {shape}")
    width = shape[-1]
    if gamma is None or beta is None:
        raise ValueError("layer_norm: gamma and beta are required")
    for name, affine in (("gamma", gamma), ("beta", beta)):
        if math.prod(list(affine.shape)) != width:
            raise ValueError(f"layer_norm: {name} shape {list(affine.shape)} does not hold W={width} elements")
    if gamma.dtype != beta.dtype or gamma.layout != beta.layout:
        raise ValueError("layer_norm: gamma and beta must share dtype and layout")
    if residual is not None and (
        list(residual.shape) != shape or residual.dtype != input_tensor.dtype or residual.layout != input_tensor.layout
    ):
        raise ValueError("layer_norm: residual must match the input's shape, dtype and layout")
    if memory_config is not None and memory_config.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
        raise ValueError("layer_norm: output memory_config must be INTERLEAVED")


def validate(input_tensor, gamma, beta, residual_input_tensor=None, *, compute_kernel_config, memory_config=None):
    _check_contract(input_tensor, gamma, beta, residual_input_tensor, memory_config)

    axes = {
        "dtype": input_tensor.dtype,
        "fp32_dest_acc_en": bool(compute_kernel_config.fp32_dest_acc_en),
        "layout": input_tensor.layout,
        "residual": "residual" if residual_input_tensor is not None else "no_residual",
        "affine_dtype": gamma.dtype,
        "affine_layout": gamma.layout,
        "memory_layout": input_tensor.memory_config().memory_layout,
        "buffer_type": input_tensor.memory_config().buffer_type,
        "affine_buffer_type": gamma.memory_config().buffer_type,
    }
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger((list(input_tensor.shape),), axes)

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"layer_norm: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")

    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"layer_norm: unsupported combination (refinement candidate): {exc}")

    # Placement of the non-axis tensors: the kernels only address interleaved pages.
    interleaved = ttnn.TensorMemoryLayout.INTERLEAVED
    others = [("gamma", gamma), ("beta", beta)]
    if residual_input_tensor is not None:
        others.append(("residual", residual_input_tensor))
    for name, tensor in others:
        if tensor.memory_config().memory_layout != interleaved:
            raise UnsupportedAxisValue(f"layer_norm: {name} must be interleaved")
    if gamma.memory_config().buffer_type != beta.memory_config().buffer_type:
        raise UnsupportedAxisValue("layer_norm: gamma and beta must share a buffer type")

    # Mechanism gates (step 8).
    if gamma.layout == ttnn.ROW_MAJOR_LAYOUT and list(gamma.shape)[-1] % TILE != 0:
        raise UnsupportedAxisValue("layer_norm: ROW_MAJOR gamma/beta last dim must be a multiple of 32")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def layer_norm(
    input_tensor: ttnn.Tensor,
    gamma: ttnn.Tensor,
    beta: ttnn.Tensor,
    residual_input_tensor: ttnn.Tensor = None,
    *,
    epsilon: float = 1e-5,
    compute_kernel_config: ttnn.ComputeConfigDescriptor = None,
    memory_config: ttnn.MemoryConfig = None,
) -> ttnn.Tensor:
    if compute_kernel_config is None:
        compute_kernel_config = default_compute_kernel_config()
    validate(
        input_tensor,
        gamma,
        beta,
        residual_input_tensor,
        compute_kernel_config=compute_kernel_config,
        memory_config=memory_config,
    )

    # The output is allocated before the L1 budget is queried: an L1 output occupies bank space.
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(input_tensor.shape)),
        input_tensor.dtype,
        input_tensor.layout,
        input_tensor.device(),
        memory_config if memory_config is not None else input_tensor.memory_config(),
    )

    blocking = select_blocking(
        input_tensor,
        residual=residual_input_tensor,
        compute_kernel_config=compute_kernel_config,
    )
    if blocking is None:
        raise NotImplementedError("layer_norm: row-resident footprint exceeds L1; regime R2 w_split_combine not built")

    program_descriptor = create_program_descriptor(
        input_tensor,
        gamma,
        beta,
        residual_input_tensor,
        output_tensor,
        blocking,
        epsilon=epsilon,
        compute_kernel_config=compute_kernel_config,
    )
    io_tensors = [input_tensor]
    if residual_input_tensor is not None:
        io_tensors.append(residual_input_tensor)
    io_tensors += [gamma, beta, output_tensor]
    return ttnn.generic_op(io_tensors, program_descriptor)
