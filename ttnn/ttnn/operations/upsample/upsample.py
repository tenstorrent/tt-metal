# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""upsample: nearest-neighbour enlargement of a channels-last image [N, H, W, C] by whole factors (sh, sw).

Specification: eval/specs/upsample/spec.py. Design: op_design.md (Blocking Model: `stick_stream`).

One `ttnn.generic_op` dispatch per call. The program is a reader -> ring -> writer pipeline with no compute
kernel (the op moves bytes and computes nothing, spec rule R1): every input stick (the C values of one
position [n, h, w], one row-major page) is read once into a cluster's L1 ring and written sh*sw times.
"""

from __future__ import annotations

import ttnn

from eval.memory import memory_axes
from eval.specs.upsample.spec import AXES
from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .upsample_program_artifacts import create_program_artifacts, stick_stream_fits


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS: the specification's axis functions, never redefined here.
# ---------------------------------------------------------------------------

INPUT_TAGGERS = AXES


# ---------------------------------------------------------------------------
# 2. SUPPORTED: the first phase is the specification's PHASE0, axis for axis.
# ---------------------------------------------------------------------------

SUPPORTED = {
    "dtype":             [ttnn.bfloat16],
    "layout":            [ttnn.ROW_MAJOR_LAYOUT],
    "memory_config":     ["DRAM", "L1"],
    "sharding":          ["interleaved"],
    "mode":              ["nearest"],
    "channel_alignment": ["tile_aligned"],
    "scale":             ["uniform"],
}


# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------

EXCLUSIONS = []


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def _whole_factor(value) -> bool:
    # bool is an int in Python; True is not a scale factor.
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def normalize_scale_factor(scale_factor):
    """An int or a pair of whole numbers >= 1 as (scale_h, scale_w); ValueError otherwise (X1).

    The form is never gated (rule R2): 2 and (2, 2) both become (2, 2).
    """
    if _whole_factor(scale_factor):
        return int(scale_factor), int(scale_factor)
    if isinstance(scale_factor, (tuple, list)) and len(scale_factor) == 2 and all(map(_whole_factor, scale_factor)):
        return int(scale_factor[0]), int(scale_factor[1])
    raise ValueError(f"upsample: scale_factor must be a whole number >= 1 or a pair of them, got {scale_factor!r}")


def validate(input_tensor, scale_factor, mode="nearest"):
    # Malformed calls (REJECTIONS X1-X3) first: they are errors, not support refusals.
    shape = tuple(input_tensor.shape)
    if len(shape) != 4:
        raise ValueError(f"upsample: the input must be of rank 4 [N, H, W, C], got shape {list(shape)}")
    normalize_scale_factor(scale_factor)
    if mode != "nearest":
        raise ValueError(f"upsample: mode must be \"nearest\", got {mode!r}")
    if not ttnn.is_tensor_storage_on_device(input_tensor):
        raise ValueError("upsample: the input tensor must be on a device")

    axes = {
        "dtype": input_tensor.dtype,
        "layout": input_tensor.layout,
        "mode": mode,
        **memory_axes(input_tensor),  # memory_config, sharding (rule R6)
    }
    inputs = (shape, scale_factor)
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger(inputs, axes)

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"upsample: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")

    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"upsample: unsupported combination (refinement candidate): {exc}")

    # Regime predicate of the Blocking Model: a ring of RING_ENTRIES one-stick entries must fit the ring cap;
    # a wider stick needs the deferred `channel_chunk` regime.
    if not stick_stream_fits(input_tensor):
        raise NotImplementedError(
            f"upsample: a stick of C={shape[-1]} does not fit the stick_stream ring (deferred channel_chunk regime)"
        )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def upsample(
    input_tensor: ttnn.Tensor,
    scale_factor,
    mode: str = "nearest",
) -> ttnn.Tensor:
    validate(input_tensor, scale_factor, mode)
    scale_h, scale_w = normalize_scale_factor(scale_factor)
    return _dispatch(input_tensor, scale_h, scale_w)


def _dispatch(input_tensor, scale_h, scale_w, **probe):
    """The one device dispatch. `probe` (reader_thread_mask / writer_thread_mask) is for tests only."""
    n, h, w, c = tuple(input_tensor.shape)
    out_shape = ttnn.Shape([n, h * scale_h, w * scale_w, c])
    # Rule R5: the output lives in the input's buffer, interleaved.
    out_spec = ttnn.TensorSpec(out_shape, input_tensor.dtype, input_tensor.layout, input_tensor.memory_config())
    output = ttnn.allocate_tensor_on_device(out_spec, input_tensor.device())
    spec, run_args, tensor_indices = create_program_artifacts(input_tensor, output, scale_h, scale_w, **probe)
    return ttnn.generic_op([input_tensor, output], spec, run_args, tensor_indices)
