# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""pad: a row-major tensor grown by `(before, after)` elements per dimension, every added element `value`.

Registry declarations (INPUT_TAGGERS, SUPPORTED, EXCLUSIONS, validate) + the public entry point. The device
program (one `ttnn.generic_op` dispatch of one `ttnn.ProgramSpec`) is built in pad_program_artifacts.py.
"""

from __future__ import annotations

from collections.abc import Sequence

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from eval.memory import memory_axes
from eval.specs.pad.spec import AXES  # noqa: E402

from .pad_program_artifacts import create_program_artifacts, fold_to_rank4, output_shape

# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS — the specification's axis functions, never redefined here.
# ---------------------------------------------------------------------------

INPUT_TAGGERS = AXES

# ---------------------------------------------------------------------------
# 2. SUPPORTED — the first phase equals the specification's PHASE0.
# ---------------------------------------------------------------------------

SUPPORTED = {
    "dtype": [ttnn.bfloat16],
    "layout": [ttnn.ROW_MAJOR_LAYOUT],
    "memory_config": ["DRAM"],
    "sharding": ["interleaved"],
    "pad_value": [0.0],
    "padding_side": ["after"],
    "alignment": ["tile_aligned", "w_non_aligned", "h_non_aligned"],
    "rank": [4],
}

# ---------------------------------------------------------------------------
# 3. EXCLUSIONS — none.
# ---------------------------------------------------------------------------

EXCLUSIONS = []


# ---------------------------------------------------------------------------
# Padding parse (R3, X1-X4): malformed calls, refused before the support gate.
# ---------------------------------------------------------------------------


def _is_int(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def parse_padding(padding, rank: int):
    """`padding` as a tuple of `(before, after)` int pairs, one per dimension, first dimension first.

    TypeError: not a sequence of pairs of ints (X4: torch's flat form). ValueError: not one pair per dimension
    (X1, X2) or a negative amount (X3).
    """
    if isinstance(padding, (str, bytes)) or not isinstance(padding, Sequence):
        raise TypeError(f"pad: padding must be a sequence of (before, after) pairs, got {type(padding).__name__}")
    pairs = []
    for item in padding:
        if isinstance(item, (str, bytes)) or not isinstance(item, Sequence) or len(item) != 2:
            raise TypeError(f"pad: every item of padding must be a (before, after) pair, got {item!r}")
        before, after = item
        if not (_is_int(before) and _is_int(after)):
            raise TypeError(f"pad: padding amounts must be ints, got {item!r}")
        pairs.append((int(before), int(after)))
    if len(pairs) != rank:
        raise ValueError(f"pad: padding has {len(pairs)} pairs for a tensor of rank {rank}; one pair per dimension")
    if any(before < 0 or after < 0 for before, after in pairs):
        raise ValueError(f"pad: padding amounts must be >= 0, got {pairs}")
    return tuple(pairs)


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def _axes(input_tensor, padding, value):
    shape = tuple(int(d) for d in input_tensor.shape)
    axes = {
        "dtype": input_tensor.dtype,
        "layout": input_tensor.layout,
        **memory_axes(input_tensor),
        "pad_value": value,
    }
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger((shape, padding), axes)
    return axes


def validate(input_tensor, padding, value):
    """Refuse a malformed `padding` (X1-X4), then gate on SUPPORTED and EXCLUSIONS (R6, R7)."""
    padding = parse_padding(padding, len(input_tensor.shape))
    axes = _axes(input_tensor, padding, value)
    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"pad: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")
    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"pad: unsupported combination (refinement candidate): {exc}")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _run(input_tensor, padding, value, **probe):
    """Build and dispatch the one device program. `probe` reaches create_program_artifacts (tests only:
    thread plan, block_rows override); a test may call this with a value validate() would refuse, since the
    kernels are value-general."""
    padding = parse_padding(padding, len(input_tensor.shape))
    shape4, pairs4 = fold_to_rank4(tuple(int(d) for d in input_tensor.shape), padding)
    out_shape = output_shape(tuple(int(d) for d in input_tensor.shape), padding)
    buffer_type = input_tensor.memory_config().buffer_type  # R4: the output lives where the input does
    out = ttnn.allocate_tensor_on_device(
        ttnn.TensorSpec(ttnn.Shape(out_shape), input_tensor.dtype, ttnn.ROW_MAJOR_LAYOUT, buffer_type),
        input_tensor.device(),
    )
    spec, run_args, tensor_indices = create_program_artifacts(input_tensor, out, shape4, pairs4, value, **probe)
    return ttnn.generic_op([input_tensor, out], spec, run_args, tensor_indices)


def pad(input_tensor: ttnn.Tensor, padding, value: float) -> ttnn.Tensor:
    validate(input_tensor, padding, value)
    return _run(input_tensor, padding, value)
