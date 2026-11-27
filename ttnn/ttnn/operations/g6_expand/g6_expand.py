# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_expand: x.expand(*sizes).contiguous() for a 4-D tiled bf16 tensor whose dim 2 is 1.

One device program (regime R1 of op_design.md): column-major linear split of the output
tiles over the grid; each core fetches row 0 of every input page its range touches,
replicates it in place into a full broadcast tile, and writes that tile to its column
segment. Pure byte movement, bit-exact.
"""

from __future__ import annotations

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .g6_expand_program_descriptor import get_program


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS
# ---------------------------------------------------------------------------


def _out_shape(shape, sizes):
    return [int(d) if int(s) == -1 else int(s) for d, s in zip(shape, sizes)]


def tag_alignment(inputs, axes):
    """Tag on the OUTPUT shape (sizes with -1 replaced by the input's dim)."""
    scenario = inputs[0]
    out = _out_shape(scenario["shape"], scenario["sizes"])
    if len(out) < 2:
        return "tile_aligned"
    if out[-1] % 32:
        return "w_non_aligned"
    if out[-2] % 32:
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
    "layout": [ttnn.TILE_LAYOUT],
    "alignment": ["tile_aligned", "w_non_aligned"],
}


# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------

EXCLUSIONS = []


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def validate(input_tensor, sizes):
    scenario = {"shape": list(input_tensor.shape), "sizes": list(sizes)}
    axes = {
        "dtype": input_tensor.dtype,
        "layout": input_tensor.layout,
    }
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger((scenario,), axes)

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"g6_expand: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")

    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"g6_expand: unsupported combination (refinement candidate): {exc}")


def _check_inputs(input_tensor, sizes):
    if not ttnn.is_tensor_storage_on_device(input_tensor):
        raise ValueError("g6_expand: input_tensor must be on device")
    mc = input_tensor.memory_config()
    if mc.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED or mc.buffer_type != ttnn.BufferType.DRAM:
        raise ValueError(f"g6_expand: input_tensor must be DRAM interleaved, got {mc}")
    shape = list(input_tensor.shape)
    if len(shape) != 4:
        raise ValueError(f"g6_expand: input rank must be 4, got {len(shape)}")
    if len(sizes) != 4:
        raise ValueError(f"g6_expand: sizes must have length 4 (rank of input), got {len(sizes)}")
    for d, (dim, s) in enumerate(zip(shape, sizes)):
        if s != -1 and s < 1:
            raise ValueError(f"g6_expand: sizes[{d}]={s} is invalid (must be -1 or >= 1)")
        if dim != 1 and s not in (-1, dim):
            raise ValueError(f"g6_expand: sizes[{d}]={s} differs from non-1 input dim {dim}")
        if dim == 1 and d != 2 and s not in (-1, 1):
            raise ValueError(f"g6_expand: expanding dim {d} is outside the contract (only dim 2)")
    if shape[2] != 1:
        raise ValueError(f"g6_expand: input dim 2 must be 1, got {shape[2]}")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

last_plan = {}


def g6_expand(input_tensor: ttnn.Tensor, sizes) -> ttnn.Tensor:
    validate(input_tensor, sizes)
    sizes = [int(s) for s in sizes]
    _check_inputs(input_tensor, sizes)

    out_shape = _out_shape(list(input_tensor.shape), sizes)
    device = input_tensor.device()
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(out_shape),
        input_tensor.dtype,
        input_tensor.layout,
        device,
        input_tensor.memory_config(),
    )

    program_descriptor, plan = get_program(input_tensor, output_tensor)
    last_plan.clear()
    last_plan.update(plan)
    return ttnn.generic_op([input_tensor, output_tensor], program_descriptor)
