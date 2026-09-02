# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""rms_norm — root-mean-square normalization over the last dimension.

    out[..., w] = x[..., w] * rsqrt( (1/W) * sum_w' x[..., w']^2 + eps ) * gamma[w]

Registry-model op file: the four declarations (INPUT_TAGGERS / SUPPORTED /
EXCLUSIONS / validate) plus the public entry point.  The kernels and the
ProgramDescriptor live in rms_norm_program_descriptor.py + kernels/.

Phase 0 scheme (op_design.md section 1.5) — row-parallel, multi-core,
coarse-blocked, dual-path on an explicit fits-in-L1 predicate:

  * the independent `row` axis (all leading dims folded, incl. H) is split
    across the FULL compute grid with split_work_to_cores(..., row_wise=True);
  * each core walks its assignment in the coarsest whole-row block that fits
    L1 (BLOCK_ROWS);
  * the dependent `width` axis stays inside a core, taken in ONE chunk
    (WT_CHUNK == Wt) whenever the working set fits, and chunked only as an L1
    fallback (the STREAM regime).

Both layouts are native (no host-side to_layout / tilize / untilize / pad /
slice), and H and/or W need not be multiples of 32.
"""

from __future__ import annotations

from typing import Any, Optional

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .rms_norm_program_descriptor import create_program_descriptor

TILE_DIM = 32


# ---------------------------------------------------------------------------
# Compute-config contract (op_design.md section 2.1)
# ---------------------------------------------------------------------------
#
# Single source of truth for what `compute_kernel_config=None` means.  The
# golden axis-tagger imports this same factory, so the default must never be
# inlined anywhere else.


def default_compute_kernel_config() -> "ttnn.ComputeConfigDescriptor":
    """Phase 0 default: the maxed-out precision corner."""
    return ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        math_approx_mode=False,
    )


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS
# ---------------------------------------------------------------------------


def tag_alignment(inputs, axes):
    """Three-value tile-alignment split (feature_spec.py:14-21).

    w_non_aligned dominates h_non_aligned: a non-tile-aligned W drives the
    masked-reduce path (partial scaler + logical-W divisor), which is a
    genuinely different kernel path from H row padding.
    """
    shape = inputs[0]
    if shape[-1] % TILE_DIM != 0:
        return "w_non_aligned"
    if shape[-2] % TILE_DIM != 0:
        return "h_non_aligned"
    return "tile_aligned"


def tag_rank(inputs, axes):
    return len(inputs[0])


INPUT_TAGGERS = {
    "alignment": tag_alignment,
    "rank": tag_rank,
}


# ---------------------------------------------------------------------------
# 2. SUPPORTED
# ---------------------------------------------------------------------------
#
# Every finite axis the golden feature_spec TARGET enumerates gets an entry so
# out-of-rectangle cells refuse cleanly instead of over-claiming.
#
# "none" on gamma_dtype / gamma_layout is the "no weight tensor" sentinel and
# is ALWAYS legal (see eval/prompts/rms_norm.txt).

SUPPORTED = {
    # Refinement 1: the full float precision surface.  Every CB's data_format is
    # derived from the dtype of the tensor it carries (activation / gamma /
    # output) in rms_norm_program_descriptor.py, so block-float rides the same
    # path as bf16 -- see that file's D5 note.
    "dtype": [ttnn.float32, ttnn.bfloat16, ttnn.bfloat8_b],
    # Both DEST accumulation modes.  {float32, False} stays an EXCLUSION below.
    "fp32_dest_acc_en": [True, False],
    "layout": [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
    "alignment": ["tile_aligned", "w_non_aligned", "h_non_aligned"],
    "rank": [2, 3, 4],
    "gamma_mode": ["gamma", "no_gamma"],
    "gamma_dtype": [ttnn.float32, ttnn.bfloat16, ttnn.bfloat8_b, "none"],
    "gamma_layout": [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT, "none"],
    # Refinement 2: all four placements.  HEIGHT_SHARDED is the Lamp-L3 knob-turn
    # (the shard cuts the independent `row` axis, so the shard IS the per-core
    # block and the reduce stays local -- zero-copy CBs, no NoC read for x);
    # WIDTH/BLOCK_SHARDED are the Lamp-L4 scheme-change (the shard cuts the
    # dependent `width` axis, so per-core partial sums are gathered to each
    # group's root, finalized there and multicast back).  See op_design.md 5.3
    # and the SCHEME_* map in rms_norm_program_descriptor.py.
    #
    # Refinement 2b completed the layout x placement rectangle: a ROW_MAJOR shard
    # cutting the width axis has a sub-tile edge, so no core holds a whole width
    # TILE -- but the combine sums per-row PARTIALS elementwise and never needs
    # one, so each core reduces the BAND it already holds, staged from its own L1
    # (descriptor D10).  Every layout x memory_layout pair is now native.
    "memory_layout": [
        ttnn.TensorMemoryLayout.INTERLEAVED,
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
    ],
}


# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------
#
# float32 activations without fp32 DEST accumulation would silently downcast
# the sum-of-squares to bf16 — refused natively rather than answered wrongly
# (references/precision_convention.md).  Now load-bearing rather than
# documentary: Refinement 1 put False in SUPPORTED["fp32_dest_acc_en"], so this
# is the one cell of that axis the op still refuses.  It stays refused even after
# Refinement 1b cut the wide-W accumulation error (descriptor D7): the objection
# is that a float32 CALLER asked for fp32 and would get a 16-bit accumulator, not
# that the error is large.

EXCLUSIONS = [
    {"dtype": ttnn.float32, "fp32_dest_acc_en": False},
]

# Refinement 2b added NO exclusions and REMOVED the two Refinement 2 had parked
# here ({ROW_MAJOR, WIDTH_SHARDED} and {ROW_MAJOR, BLOCK_SHARDED}).  An RM shard
# that cuts the width axis has a sub-tile edge, so no core holds a whole width
# TILE -- but the cross-core combine sums per-row PARTIALS elementwise and never
# needs one: each core stages the band it already holds out of its own L1, in the
# tensor's GLOBAL tile frame, and joins the unchanged combine.  See _plan_band in
# rms_norm_program_descriptor.py.  gamma works at BOTH layouts there, because the
# global tile frame keeps every gamma fetch on a tile column.

# Refinement 1 added NO exclusions.  Two corners were expected to need one and
# both were measured clean, so claiming them would have under-reported support:
#
#  * {gamma_dtype: bfloat8_b, alignment: *_non_aligned} -- op_design.md 9.2
#    predicted a bf8b gamma would be perturbed on a non-tile-aligned W, since 16
#    weights share one exponent and the straddling block mixes pad lanes with
#    real weights.  It does not: gamma's tile padding is ZERO, and a zero never
#    raises a block's shared exponent, so the real weights in that block are
#    untouched.  Measured PCC 0.99997 / rel RMS 0.008 on (1,1,32,72) and
#    (1,1,50,128); pinned by test_rms_norm_precision_mixed_gamma_dtype.
#  * {dtype: bfloat8_b, ...} generally -- every CB already derives its format
#    from the dtype of the tensor it carries, so block-float needed no new path.


# ---------------------------------------------------------------------------
# 3b. PROPERTIES — non-axis capabilities
# ---------------------------------------------------------------------------

PROPERTIES = {
    # Phase 0 splits the row axis over device.compute_with_storage_grid_size().
    "multi_core": {"value": True, "source": "declared"},
    # Every CB page count derives from BLOCK_ROWS / WT_CHUNK / a depth knob,
    # each bounded by the L1 budget predicate in the program descriptor.
    "bounded_cb": {"value": True, "source": "declared"},
    # math_fidelity / math_approx_mode are never gated.
    "math_fidelity": {"value": ["LoFi", "HiFi2", "HiFi3", "HiFi4"], "source": "declared"},
}


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def validate(
    input_tensor,
    *,
    gamma=None,
    epsilon: float = 1e-6,
    compute_kernel_config=None,
    memory_config=None,
    program_config=None,
):
    """Runtime support gate. Raises before any device work."""
    # --- structural checks first: the taggers below index shape[-2] ---------
    if len(input_tensor.shape) < 2:
        raise ValueError(
            f"rms_norm: input rank must be >= 2 (reduction needs a row axis and a width axis); "
            f"got rank {len(input_tensor.shape)} for shape {list(input_tensor.shape)}"
        )
    if gamma is not None and gamma.shape[-1] != input_tensor.shape[-1]:
        raise ValueError(
            f"rms_norm: gamma last dim {gamma.shape[-1]} must match the input's last dim " f"{input_tensor.shape[-1]}"
        )
    if epsilon <= 0.0:
        raise ValueError(f"rms_norm: epsilon must be > 0, got {epsilon}")

    cfg = compute_kernel_config if compute_kernel_config is not None else default_compute_kernel_config()

    has_gamma = gamma is not None
    axes = {
        "dtype": input_tensor.dtype,
        "layout": input_tensor.layout,
        "fp32_dest_acc_en": bool(getattr(cfg, "fp32_dest_acc_en", True)),
        "gamma_mode": "gamma" if has_gamma else "no_gamma",
        "gamma_dtype": gamma.dtype if has_gamma else "none",
        "gamma_layout": gamma.layout if has_gamma else "none",
        "memory_layout": input_tensor.memory_config().memory_layout,
    }
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger((list(input_tensor.shape),), axes)

    # 1. SUPPORTED — per-axis
    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"rms_norm: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")

    # 2. EXCLUSIONS — cell-level inside SUPPORTED
    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"rms_norm: unsupported combination (refinement candidate): {exc}")

    # The requested output placement must also be one we implement.
    if memory_config is not None and memory_config.memory_layout not in SUPPORTED["memory_layout"]:
        raise UnsupportedAxisValue(
            f"rms_norm: memory_config.memory_layout={memory_config.memory_layout!r} not in "
            f"SUPPORTED {SUPPORTED['memory_layout']}"
        )

    return axes


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def rms_norm(
    input_tensor: "ttnn.Tensor",
    *,
    gamma: Optional["ttnn.Tensor"] = None,
    epsilon: float = 1e-6,
    compute_kernel_config: Optional["ttnn.ComputeConfigDescriptor"] = None,
    memory_config: Optional["ttnn.MemoryConfig"] = None,
    program_config: Optional[Any] = None,
) -> "ttnn.Tensor":
    """RMSNorm over the last dimension.

    Args:
        input_tensor: on-device tensor, rank >= 2, TILE or ROW_MAJOR layout.
        gamma: optional per-channel scale of shape (1, 1, 1, W); its dtype and
            layout are independent of the input's.
        epsilon: added to the mean square before the rsqrt.
        compute_kernel_config: resolved through default_compute_kernel_config()
            when None, then passed through unmodified.
        memory_config: output placement (defaults to the input's).
        program_config: reserved; ignored when None.
    """
    validate(
        input_tensor,
        gamma=gamma,
        epsilon=epsilon,
        compute_kernel_config=compute_kernel_config,
        memory_config=memory_config,
        program_config=program_config,
    )

    if compute_kernel_config is None:
        compute_kernel_config = default_compute_kernel_config()

    device = input_tensor.device()
    output_memory_config = memory_config if memory_config is not None else input_tensor.memory_config()

    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(input_tensor.shape)),
        input_tensor.dtype,
        input_tensor.layout,
        device,
        output_memory_config,
    )

    program_descriptor = create_program_descriptor(
        input_tensor,
        output_tensor,
        gamma=gamma,
        epsilon=epsilon,
        compute_kernel_config=compute_kernel_config,
    )

    tensors = [input_tensor] + ([gamma] if gamma is not None else []) + [output_tensor]
    return ttnn.generic_op(tensors, program_descriptor)
