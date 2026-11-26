# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_repeat: out = x.repeat(reps), materialized bit for bit in one device program.

Phase 0 contract: x (1, 1, 1, W), W % 32 == 0, reps [1, 1, R, 1], 1 <= R < 32,
bfloat16, TILE, DRAM interleaved. out is (1, 1, R, W): one partial tile row.
"""

from __future__ import annotations

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .g6_repeat_program_descriptor import get_program, plan_report as _plan_report

TILE_DIM = 32


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS
# ---------------------------------------------------------------------------


def tag_alignment(inputs, axes):
    """Alignment of the OUTPUT shape (x's shape times reps, dim by dim)."""
    scenario = inputs[0]
    out = [int(d) * int(r) for d, r in zip(scenario["shape"], scenario["reps"])]
    if out[-1] % TILE_DIM:
        return "w_non_aligned"
    if out[-2] % TILE_DIM:
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
    "alignment": ["h_non_aligned"],
}


# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------

EXCLUSIONS = []


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def _refuse(message):
    raise ValueError(f"g6_repeat: {message}")


def _check_contract(input_tensor, reps):
    """Phase 0 contract checks; they raise ValueError and run before the SUPPORTED
    gate so that e.g. reps[2] = 32 (tagged "tile_aligned") is refused as outside the
    contract rather than as a missing feature."""
    if not isinstance(reps, (list, tuple)):
        _refuse(f"reps must be a list or tuple of ints, got {type(reps).__name__}")
    try:
        reps = [int(r) for r in reps]
    except (TypeError, ValueError):
        _refuse(f"reps must hold ints, got {reps!r}")
    shape = list(input_tensor.shape)
    if len(shape) != 4:
        _refuse(f"input must have rank 4, got rank {len(shape)} (shape {shape})")
    if len(reps) != len(shape):
        _refuse(f"reps {reps} must have the rank of the input ({len(shape)})")
    if any(r <= 0 for r in reps):
        _refuse(f"every repetition count must be positive, got {reps}")
    if not ttnn.is_tensor_storage_on_device(input_tensor):
        _refuse("input must be on a device")
    mc = input_tensor.memory_config()
    if mc.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED or mc.buffer_type != ttnn.BufferType.DRAM:
        _refuse(f"input must be DRAM interleaved, got {mc.memory_layout} in {mc.buffer_type}")
    if shape[:3] != [1, 1, 1]:
        _refuse(f"input shape must be (1, 1, 1, W), got {tuple(shape)}")
    if reps[0] != 1 or reps[1] != 1 or reps[3] != 1:
        _refuse(f"only dim 2 may be repeated, got reps {reps}")
    if reps[2] >= TILE_DIM:
        _refuse(f"reps[2] must be < {TILE_DIM} (out must stay one partial tile row), got {reps[2]}")
    return reps


def validate(input_tensor, reps):
    reps = _check_contract(input_tensor, reps)

    axes = {
        "dtype": input_tensor.dtype,
        "layout": input_tensor.layout,
    }
    scenario = {"shape": list(input_tensor.shape), "reps": reps}
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger((scenario,), axes)

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"g6_repeat: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")

    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"g6_repeat: unsupported combination (refinement candidate): {exc}")

    return reps


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def g6_repeat(input_tensor: ttnn.Tensor, reps) -> ttnn.Tensor:
    """out = input_tensor.repeat(*reps), a new tensor in input_tensor's dtype,
    layout and memory placement, every logical element bit-identical."""
    reps = validate(input_tensor, reps)

    out_shape = [int(d) * r for d, r in zip(input_tensor.shape, reps)]
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(out_shape),
        input_tensor.dtype,
        input_tensor.layout,
        input_tensor.device(),
        input_tensor.memory_config(),
    )

    program_descriptor = get_program(input_tensor, output_tensor, reps)
    return ttnn.generic_op([input_tensor, output_tensor], program_descriptor)


def plan_report(input_tensor: ttnn.Tensor, reps) -> dict:
    """The plan a call with these inputs selects (BB-08): regime, cores, block size,
    and each core's (start_tile_col, core_w_tiles) range — every active core has role
    "repeat"."""
    reps = validate(input_tensor, reps)
    return _plan_report(input_tensor, reps)
