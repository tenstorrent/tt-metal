# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""to_layout: change a tensor's layout (TILE <-> ROW_MAJOR), values unchanged.

Specification: eval/specs/to_layout/spec.py. Design: op_design.md (this directory).

Every call dispatches exactly one device program (`ttnn.generic_op`), chosen by regime:
    identity_copy    input layout == requested layout (DM-only page copy)
    tilize_stream    ROW_MAJOR -> TILE
    untilize_stream  TILE -> ROW_MAJOR
The output is allocated in the input's buffer type, interleaved (rule R5).
"""

from __future__ import annotations

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS: the specification's axis functions, never redefined here.
# ---------------------------------------------------------------------------

from eval.specs.to_layout.spec import AXES  # noqa: E402
from eval.memory import memory_axes  # noqa: E402

INPUT_TAGGERS = AXES

# ---------------------------------------------------------------------------
# 2. SUPPORTED (first phase = the specification's PHASE0)
# ---------------------------------------------------------------------------

SUPPORTED = {
    "dtype":         [ttnn.bfloat16],
    "layout":        [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
    "target_layout": [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
    "memory_config": ["DRAM", "L1"],
    "sharding":      ["interleaved"],
    "alignment":     ["tile_aligned", "h_non_aligned"],
    "rank":          [3, 4],
}

# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------

EXCLUSIONS = []

# Block-float dtypes: stored in tiles only, no row-major form (rule R2).
_BLOCK_FLOAT_DTYPES = tuple(
    d for d in (getattr(ttnn, "bfloat8_b", None), getattr(ttnn, "bfloat4_b", None)) if d is not None
)

# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def validate(input_tensor, layout):
    # Rejection X1: `layout` must be a ttnn.Layout.
    if not isinstance(layout, ttnn.Layout):
        raise TypeError(f"to_layout: `layout` must be a ttnn.Layout, got {type(layout).__name__} ({layout!r})")
    if not isinstance(input_tensor, ttnn.Tensor):
        raise TypeError(f"to_layout: `input_tensor` must be a ttnn.Tensor, got {type(input_tensor).__name__}")
    if not ttnn.is_tensor_storage_on_device(input_tensor):
        raise NotImplementedError("to_layout: the input tensor must be on device")

    axes = {
        "dtype": input_tensor.dtype,
        "layout": input_tensor.layout,
        "target_layout": layout,
        **memory_axes(input_tensor),  # memory_config, sharding (rule R6)
    }
    shape = tuple(input_tensor.shape)
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger((shape,), axes)

    # Rule R2: a block-float tensor has no row-major form.
    if axes["dtype"] in _BLOCK_FLOAT_DTYPES and layout == ttnn.ROW_MAJOR_LAYOUT:
        raise UnsupportedAxisValue(
            f"to_layout: a block-float tensor ({axes['dtype']}) has no ROW_MAJOR form (rule R2)"
        )

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"to_layout: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")

    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"to_layout: unsupported combination (refinement candidate): {exc}")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def to_layout(
    input_tensor: ttnn.Tensor,
    layout: ttnn.Layout,
) -> ttnn.Tensor:
    validate(input_tensor, layout)

    from .to_layout_program_artifacts import create_program_artifacts

    # R5: the output lives in the input's buffer, interleaved.
    memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, input_tensor.memory_config().buffer_type)
    out_spec = ttnn.TensorSpec(input_tensor.shape, input_tensor.dtype, layout, memory_config)
    output = ttnn.allocate_tensor_on_device(out_spec, input_tensor.device())

    spec, run_args, tensor_indices = create_program_artifacts(input_tensor, output)
    return ttnn.generic_op([input_tensor, output], spec, run_args, tensor_indices)
