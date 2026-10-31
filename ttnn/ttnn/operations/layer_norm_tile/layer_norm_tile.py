# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""layer_norm_tile: y = (x' - mean) * rsqrt(var + eps) * gamma + beta over the last dim, x' = x [+ residual].

One ttnn.generic_op dispatch per call (regime row_resident, op_design.md).
"""

from __future__ import annotations

import math

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .layer_norm_tile_program_descriptor import TILE, create_program_descriptor, derive_blocking


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
    "memory_layout": [ttnn.TensorMemoryLayout.INTERLEAVED],
    "residual": ["none", "present"],
    "affine_layout": [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT, "mixed"],
}


# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------

EXCLUSIONS = []


PROPERTIES = {
    "multi_core": {"value": True, "source": "declared"},
    "bounded_cb": {"value": True, "source": "declared"},
    "math_fidelity": {"value": ["LoFi", "HiFi2", "HiFi3", "HiFi4"], "source": "declared"},
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


def _check_universe(input_tensor, gamma, beta, residual_input_tensor):
    """Structural requirements; violations are ValueError (outside the op's universe, not refinements)."""
    tensors = {"input_tensor": input_tensor, "gamma": gamma, "beta": beta}
    if residual_input_tensor is not None:
        tensors["residual_input_tensor"] = residual_input_tensor
    for name, tensor in tensors.items():
        if tensor.dtype != ttnn.bfloat16:
            raise ValueError(f"layer_norm: {name} dtype must be bfloat16, got {tensor.dtype}")

    shape = list(input_tensor.shape)
    if input_tensor.layout != ttnn.TILE_LAYOUT:
        raise ValueError(f"layer_norm: input_tensor must be TILE_LAYOUT, got {input_tensor.layout}")
    if len(shape) != 4:
        raise ValueError(f"layer_norm: input_tensor must be rank 4, got shape {shape}")
    width = shape[-1]

    if residual_input_tensor is not None:
        if residual_input_tensor.layout != ttnn.TILE_LAYOUT:
            raise ValueError("layer_norm: residual_input_tensor must be TILE_LAYOUT")
        if list(residual_input_tensor.shape) != shape:
            raise ValueError(f"layer_norm: residual shape {list(residual_input_tensor.shape)} != input shape {shape}")

    for name, affine in (("gamma", gamma), ("beta", beta)):
        affine_shape = list(affine.shape)
        if math.prod(affine_shape) != width:
            raise ValueError(f"layer_norm: {name} volume must equal input width {width}, got shape {affine_shape}")
        if affine.layout == ttnn.TILE_LAYOUT and affine_shape[-1] != width:
            raise ValueError(f"layer_norm: TILE {name} last dim must be {width}, got shape {affine_shape}")


def _check_affine_sticks(gamma, beta):
    """Writer mechanism cap: a 32-element row-0 chunk must lie inside one ROW_MAJOR stick."""
    for name, affine in (("gamma", gamma), ("beta", beta)):
        stick = affine.shape[-1]
        if affine.layout == ttnn.ROW_MAJOR_LAYOUT and stick % TILE != 0:
            raise UnsupportedAxisValue(f"layer_norm: ROW_MAJOR {name} stick width {stick} is not a multiple of {TILE}")


def validate(input_tensor, gamma, beta, residual_input_tensor=None, *, compute_kernel_config):
    _check_universe(input_tensor, gamma, beta, residual_input_tensor)

    # The writer takes gamma's and beta's layouts as independent CT flags, so "mixed" is a real code path.
    affine_layout = gamma.layout if gamma.layout == beta.layout else "mixed"
    tensors = [input_tensor, gamma, beta] + ([residual_input_tensor] if residual_input_tensor is not None else [])
    memory_layouts = {t.memory_config().memory_layout for t in tensors}
    memory_layout = memory_layouts.pop() if len(memory_layouts) == 1 else "mixed"

    axes = {
        "dtype": input_tensor.dtype,
        "fp32_dest_acc_en": bool(compute_kernel_config.fp32_dest_acc_en),
        "layout": input_tensor.layout,
        "memory_layout": memory_layout,
        "residual": "present" if residual_input_tensor is not None else "none",
        "affine_layout": affine_layout,
    }
    shape = tuple(input_tensor.shape)
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger((shape,), axes)

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"layer_norm: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")

    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"layer_norm: unsupported combination (refinement candidate): {exc}")

    _check_affine_sticks(gamma, beta)


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
) -> ttnn.Tensor:
    if compute_kernel_config is None:
        compute_kernel_config = default_compute_kernel_config()
    validate(input_tensor, gamma, beta, residual_input_tensor, compute_kernel_config=compute_kernel_config)

    blocking = derive_blocking(
        input_tensor,
        has_residual=residual_input_tensor is not None,
        fp32_dest_acc_en=bool(compute_kernel_config.fp32_dest_acc_en),
        affine_in_l1=all(t.memory_config().buffer_type == ttnn.BufferType.L1 for t in (gamma, beta)),
    )
    if blocking is None:
        raise UnsupportedAxisValue(
            f"layer_norm: width {input_tensor.shape[-1]} exceeds the row_resident L1 cap (w_split_combine not built)"
        )

    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(input_tensor.shape)),
        input_tensor.dtype,
        ttnn.TILE_LAYOUT,
        input_tensor.device(),
        input_tensor.memory_config(),
    )

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
    io_tensors = [input_tensor, gamma, beta]
    if residual_input_tensor is not None:
        io_tensors.append(residual_input_tensor)
    io_tensors.append(output_tensor)
    return ttnn.generic_op(io_tensors, program_descriptor)
