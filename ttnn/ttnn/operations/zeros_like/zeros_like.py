# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""zeros_like: a new tensor with the input's shape, dtype, layout and memory config, every element +0.0.

Specification: eval/specs/zeros_like/spec.py. Design: op_design.md (beside this file).
One ttnn.generic_op dispatch of a Metal 2.0 ProgramSpec (zeros_like_program_artifacts.py): a writer-only
program that zeroes one page in each core's L1 and writes it to every output page that core owns. The
input's buffer is never read (R1); only its spec is.
"""

from __future__ import annotations

import math

import ttnn
from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from eval import memory
from eval.specs.zeros_like.spec import AXES

from .zeros_like_program_artifacts import create_program_artifacts

# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS — the specification's AXES, never redefined here.
# ---------------------------------------------------------------------------

INPUT_TAGGERS = AXES

# ---------------------------------------------------------------------------
# 2. SUPPORTED — equals the specification's PHASE0 after the first phase.
# ---------------------------------------------------------------------------

SUPPORTED = {
    "dtype":         [ttnn.bfloat16],
    "layout":        [ttnn.TILE_LAYOUT],
    "memory_config": ["DRAM"],
    "sharding":      ["interleaved"],
    "alignment":     ["tile_aligned", "w_non_aligned", "h_non_aligned"],
    "rank":          [2, 3, 4],
}

# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------

EXCLUSIONS = []

# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def validate(tensor):
    """Refuse (NotImplementedError) anything outside SUPPORTED / inside EXCLUSIONS (R5, R6)."""
    shape = tuple(tensor.shape)
    axes = {
        "dtype": tensor.dtype,
        "layout": tensor.layout,
        **memory.memory_axes(tensor),
    }
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger((shape,), axes)

    # Name every axis outside SUPPORTED, not only the first in dict order: a sharded L1 input must be refused by
    # its `sharding` value (R5) even while `memory_config` L1 is itself unsupported.
    refused = [f"{axis}={axes[axis]!r} not in SUPPORTED {allowed}"
               for axis, allowed in SUPPORTED.items() if axes[axis] not in allowed]
    if refused:
        raise UnsupportedAxisValue("zeros_like: " + "; ".join(refused))

    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"zeros_like: unsupported combination (refinement candidate): {exc}")

    # A zero-volume tensor has no page to write, so there is no program to dispatch (one-dispatch rule).
    if math.prod(shape) == 0:
        raise UnsupportedAxisValue(f"zeros_like: zero-volume tensor {list(shape)} is not supported")


# ---------------------------------------------------------------------------
# Public entry point — the specification's stub: zeros_like(tensor) -> Tensor
# ---------------------------------------------------------------------------


def zeros_like(tensor: ttnn.Tensor) -> ttnn.Tensor:
    validate(tensor)
    # A new buffer (R2) with the input's shape, dtype, layout (R4) and memory config (R3).
    out = ttnn.allocate_tensor_on_device(tensor.spec, tensor.device())
    spec, run_args, tensor_indices = create_program_artifacts(out)
    return ttnn.generic_op([out], spec, run_args, tensor_indices)
