# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""add_: `a <- a + b` (with activations=[RELU]: `a <- max(a + b, 0)`), written into input_tensor_a's own
buffer and returned as that tensor. One native device program per call; no tensor allocated.

Specification: eval/specs/add_/spec.py. Design: op_design.md (regime `sharded_resident`).
"""

from __future__ import annotations

import ttnn

from eval.memory import memory_axes
from eval.specs.add_.spec import AXES, canonical_activations
from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .add__program_artifacts import TILE, create_program_artifacts, tensor_tiles

# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS: the specification's axis functions, never redefined here.
# ---------------------------------------------------------------------------

INPUT_TAGGERS = AXES

# ---------------------------------------------------------------------------
# 2. SUPPORTED: the first phase (the specification's PHASE0), one entry per TARGET axis.
# ---------------------------------------------------------------------------

SUPPORTED = {
    "dtype":            [ttnn.bfloat16],
    "fp32_dest_acc_en": [True],
    "layout":           [ttnn.TILE_LAYOUT],
    "memory_config":    ["L1"],
    "sharding":         ["height"],
    "activation":       ["relu"],
    "alignment":        ["tile_aligned", "h_non_aligned"],
    "rank":             [4],
    "broadcast":        ["none"],
}

# ---------------------------------------------------------------------------
# 3. EXCLUSIONS: none inside the first-phase rectangle. (Rule R1's {float32, fp32_dest_acc_en=False}
# joins this list when float32 enters SUPPORTED.)
# ---------------------------------------------------------------------------

EXCLUSIONS = []


def default_compute_kernel_config():
    """What `compute_kernel_config=None` means (rule R2): the one place the default lives."""
    return ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        math_approx_mode=False,
    )


def _resolve_config(compute_kernel_config):
    """`compute_kernel_config`, or the default when it is None (rule R2: resolved through the factory only)."""
    return compute_kernel_config if compute_kernel_config is not None else default_compute_kernel_config()


def _broadcasts_to(b_shape, a_shape) -> bool:
    """Whether b broadcasts to the shape of a (the result has a's shape: it is written into a)."""
    if len(b_shape) > len(a_shape):
        return False
    padded = (1,) * (len(a_shape) - len(b_shape)) + tuple(b_shape)
    return all(bd == ad or bd == 1 for bd, ad in zip(padded, a_shape))


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def validate(input_tensor_a, input_tensor_b, *, activations=None, compute_kernel_config=None):
    """The runtime gate. Malformed calls raise ValueError (X1); valid calls outside the current support
    raise UnsupportedAxisValue / ExcludedCell (NotImplementedError, rule R9). Never reads math_fidelity or
    math_approx_mode (rule R3)."""
    config = _resolve_config(compute_kernel_config)
    a_shape, b_shape = tuple(input_tensor_a.shape), tuple(input_tensor_b.shape)

    # X1: b must broadcast to a.
    if not _broadcasts_to(b_shape, a_shape):
        raise ValueError(f"add_: the shape of input_tensor_b {list(b_shape)} does not broadcast to the shape of "
                         f"input_tensor_a {list(a_shape)}")

    # R7: one dtype, one layout, one placement for both operands.
    if input_tensor_b.dtype != input_tensor_a.dtype:
        raise UnsupportedAxisValue(f"add_: input_tensor_b dtype {input_tensor_b.dtype} is not input_tensor_a's "
                                   f"({input_tensor_a.dtype})")
    if input_tensor_b.layout != input_tensor_a.layout:
        raise UnsupportedAxisValue(f"add_: input_tensor_b layout {input_tensor_b.layout} is not input_tensor_a's "
                                   f"({input_tensor_a.layout})")
    placement = memory_axes(input_tensor_a)
    if memory_axes(input_tensor_b) != placement:
        raise UnsupportedAxisValue(f"add_: input_tensor_b is placed {memory_axes(input_tensor_b)}, input_tensor_a "
                                   f"{placement}; both operands must be placed alike")

    axes = {
        "dtype":            input_tensor_a.dtype,
        "layout":           input_tensor_a.layout,
        "fp32_dest_acc_en": bool(config.fp32_dest_acc_en),
        "activation":       canonical_activations(activations, {}),
        **placement,
    }
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger((a_shape, b_shape), axes)

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"add_: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")
    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"add_: unsupported combination (refinement candidate): {exc}")

    # Regime `sharded_resident`: b's ring must sit on a's cores, over a's shard layout.
    if axes["sharding"] == "height":
        a_spec = input_tensor_a.memory_config().shard_spec
        b_spec = input_tensor_b.memory_config().shard_spec
        if a_spec is None or b_spec is None:
            raise UnsupportedAxisValue("add_: a height-sharded operand has no shard spec")
        if (a_spec.grid != b_spec.grid or tuple(a_spec.shape) != tuple(b_spec.shape)
                or a_spec.orientation != b_spec.orientation):
            raise UnsupportedAxisValue("add_: both operands must have one shard spec (grid, shard shape, orientation)")
        _, col_tiles = tensor_tiles(a_shape)
        if a_spec.shape[0] % TILE != 0 or a_spec.shape[1] != col_tiles * TILE:
            raise UnsupportedAxisValue(f"add_: shard shape {list(a_spec.shape)} is not whole tile rows of the full "
                                       f"padded width {col_tiles * TILE}")
    return axes


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def add_(
    input_tensor_a: ttnn.Tensor,
    input_tensor_b: ttnn.Tensor,
    *,
    activations=None,
    compute_kernel_config: ttnn.ComputeConfigDescriptor = None,
) -> ttnn.Tensor:
    axes = validate(input_tensor_a, input_tensor_b, activations=activations, compute_kernel_config=compute_kernel_config)
    spec, run_args, tensor_args = create_program_artifacts(
        input_tensor_a,
        input_tensor_b,
        pack_relu=axes["activation"] == "relu",
        compute_kernel_config=_resolve_config(compute_kernel_config),
    )
    # One dispatch; a is the in-place output (last). Its handle is discarded: the caller's tensor is returned.
    ttnn.generic_op([input_tensor_b, input_tensor_a], spec, run_args, tensor_args)
    return input_tensor_a
