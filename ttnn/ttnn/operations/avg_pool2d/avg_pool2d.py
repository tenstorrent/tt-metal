# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""avg_pool2d: global average pool over the flattened images [1, 1, N*H*W, C] -> [1, 1, N, C].

Specification: eval/specs/avg_pool2d/spec.py. Design: op_design.md (regime `col_split_stream`).
One native device program per call (`ttnn.generic_op` on a Metal 2.0 `ProgramSpec`).
"""

from __future__ import annotations

import ttnn

from eval import memory
from eval.specs.avg_pool2d.spec import AXES
from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .avg_pool2d_program_artifacts import create_program_artifacts

# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS: the specification's axis functions, never redefined here.
# ---------------------------------------------------------------------------

INPUT_TAGGERS = AXES

# ---------------------------------------------------------------------------
# 2. SUPPORTED: the first phase (the specification's PHASE0).
# ---------------------------------------------------------------------------

SUPPORTED = {
    "dtype":             [ttnn.bfloat16],
    "fp32_dest_acc_en":  [True, False],
    "layout":            [ttnn.TILE_LAYOUT],
    "memory_config":     ["DRAM", "L1"],
    "sharding":          ["interleaved", "width"],
    "output_dtype":      [ttnn.bfloat16],
    "output_layout":     [ttnn.TILE_LAYOUT],
    "window":            ["global"],
    "batch":             ["single"],
    "channel_alignment": ["tile_aligned"],
}

# ---------------------------------------------------------------------------
# 3. EXCLUSIONS: none.
# ---------------------------------------------------------------------------

EXCLUSIONS = []


# ---------------------------------------------------------------------------
# Compute config (rules R2-R5)
# ---------------------------------------------------------------------------


def default_compute_kernel_config():
    """What `compute_kernel_config=None` means (rule R2): HiFi4, a 32-bit accumulator, no approximation."""
    return ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        math_approx_mode=False,
    )


def _resolve_compute_config(compute_kernel_config):
    """A ttnn.ComputeConfigDescriptor from None (the default factory), a descriptor, or tt-metal's device
    compute kernel config (rule R4): the three fields are read as attributes; any other field
    (packer_l1_acc, throttle_level) is ignored. No field value is refused (rule R3)."""
    if compute_kernel_config is None:
        return default_compute_kernel_config()
    if isinstance(compute_kernel_config, ttnn.ComputeConfigDescriptor):
        return compute_kernel_config
    return ttnn.ComputeConfigDescriptor(
        math_fidelity=compute_kernel_config.math_fidelity,
        fp32_dest_acc_en=bool(compute_kernel_config.fp32_dest_acc_en),
        math_approx_mode=bool(compute_kernel_config.math_approx_mode),
    )


# ---------------------------------------------------------------------------
# Argument normalization (rule R6, rejections X1-X3)
# ---------------------------------------------------------------------------


def _int_tuple(name, value, lengths):
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"avg_pool2d: {name} must be a list or a tuple, got {type(value).__name__}")
    if len(value) not in lengths:
        raise ValueError(f"avg_pool2d: {name} must have {' or '.join(map(str, lengths))} values, got {len(value)}")
    if not all(isinstance(v, int) and not isinstance(v, bool) for v in value):
        raise TypeError(f"avg_pool2d: {name} values must be ints, got {value!r}")
    return tuple(value)


def _normalize_window(kernel_size, stride, padding):
    """(kernel, stride, (top, bottom, left, right)) as tuples, whatever form they came in (R6)."""
    kernel = _int_tuple("kernel_size", kernel_size, (2,))
    stride = _int_tuple("stride", stride, (2,))
    padding = _int_tuple("padding", padding, (2, 4))
    if len(padding) == 2:
        padding = (padding[0], padding[0], padding[1], padding[1])
    if min(kernel) < 1 or min(stride) < 1:
        raise ValueError(f"avg_pool2d: kernel_size {kernel} and stride {stride} must be >= 1")
    if min(padding) < 0:
        raise ValueError(f"avg_pool2d: padding {padding} must be >= 0")
    return kernel, stride, padding


def _check_shape(input_tensor, batch_size, input_h, input_w, channels):
    """X1: the tensor must be [1, 1, batch_size*input_h*input_w, channels]."""
    for name, v in (("batch_size", batch_size), ("input_h", input_h), ("input_w", input_w), ("channels", channels)):
        if not isinstance(v, int) or isinstance(v, bool) or v < 1:
            raise ValueError(f"avg_pool2d: {name} must be an int >= 1, got {v!r}")
    shape = tuple(input_tensor.shape)
    expected = (1, 1, batch_size * input_h * input_w, channels)
    if shape != expected:
        raise ValueError(f"avg_pool2d: input shape {shape} is not [1, 1, N*H*W, C] = {list(expected)}")
    return shape


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def _call_axes(input_tensor, batch_size, input_h, input_w, channels, kernel_size, stride, padding,
               compute_kernel_config, dtype, output_layout):
    shape = _check_shape(input_tensor, batch_size, input_h, input_w, channels)
    kernel, stride, padding = _normalize_window(kernel_size, stride, padding)
    config = _resolve_compute_config(compute_kernel_config)
    axes = {
        "dtype":            input_tensor.dtype,
        "fp32_dest_acc_en": bool(config.fp32_dest_acc_en),
        "layout":           input_tensor.layout,
        **memory.memory_axes(input_tensor),
        "output_dtype":     dtype,
        "output_layout":    output_layout,
    }
    inputs = (shape, batch_size, input_h, input_w, channels, kernel, stride, padding)
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger(inputs, axes)
    return axes, config


def validate(input_tensor, batch_size, input_h, input_w, channels, kernel_size, stride, padding, *,
             compute_kernel_config=None, dtype=ttnn.bfloat16, output_layout=ttnn.ROW_MAJOR_LAYOUT):
    """Refuse a malformed call (X1-X3) and anything outside SUPPORTED / inside EXCLUSIONS (R11, R12).
    Returns the call's axes and its compute config as a ttnn.ComputeConfigDescriptor."""
    axes, config = _call_axes(input_tensor, batch_size, input_h, input_w, channels, kernel_size, stride,
                              padding, compute_kernel_config, dtype, output_layout)
    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"avg_pool2d: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")
    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"avg_pool2d: unsupported combination (refinement candidate): {exc}")
    return axes, config


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def avg_pool2d(
    input_tensor: ttnn.Tensor,
    batch_size: int,
    input_h: int,
    input_w: int,
    channels: int,
    kernel_size,
    stride,
    padding,
    *,
    compute_kernel_config: ttnn.ComputeConfigDescriptor = None,
    dtype: ttnn.DataType = ttnn.bfloat16,
    output_layout: ttnn.Layout = ttnn.ROW_MAJOR_LAYOUT,
) -> ttnn.Tensor:
    """Global average pool of each image: out[n, c] = mean over (i, j) of in[(n*H + i)*W + j, c]."""
    _, config = validate(input_tensor, batch_size, input_h, input_w, channels, kernel_size, stride, padding,
                         compute_kernel_config=compute_kernel_config, dtype=dtype, output_layout=output_layout)

    # Rule R9: interleaved, in the input's buffer type, never sharded.
    out_memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED,
                                          input_tensor.memory_config().buffer_type)
    out_spec = ttnn.TensorSpec(ttnn.Shape([1, 1, batch_size, channels]), dtype, output_layout, out_memory_config)
    output = ttnn.allocate_tensor_on_device(out_spec, input_tensor.device())

    spec, run_args, tensor_indices = create_program_artifacts(
        input_tensor, output, positions=input_h * input_w, compute_config=config)
    return ttnn.generic_op([input_tensor, output], spec, run_args, tensor_indices)
