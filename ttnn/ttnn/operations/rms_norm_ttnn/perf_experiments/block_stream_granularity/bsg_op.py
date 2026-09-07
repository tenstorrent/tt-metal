# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""rms_norm_ttnn — root-mean-square normalization over the last dimension.

    t = x + residual_input_tensor                       (optional)
    y = t * rsqrt( (1/W) * sum_w t[..., w]^2 + eps )
    y = y * weight                                      (optional, per-channel)
    y = y + bias                                        (optional, per-channel)

Registry-model op file: the four declarations (INPUT_TAGGERS / SUPPORTED /
EXCLUSIONS / validate) plus the public entry point.  The kernels and the
ProgramDescriptor live in rms_norm_ttnn_program_descriptor.py + kernels/.

DERIVATIVE OF THE DESIGNATED SEED `ttnn/ttnn/operations/rms_norm/`.  Every
scheme, regime, knob, CB and kernel of the seed is preserved; this file adds
capability at the seams (op_design.md's A1..A13 delta table):

  A1  residual_input_tensor -- a second full-shape activation added BEFORE the
      statistics.
  A2  bias -- a per-channel shift applied AFTER the scale.
  A3  ranks 0, 1 and 5 (no rank floor, no rank ceiling).
  A4  zero-volume inputs (one degenerate device program).
  A5  program_config consumed: two variants, validated, `subblock_w` honoured,
      `inplace` honoured.
  A6  two compute-config object types accepted (one adapter at the door).
  A7  the default compute config becomes HiFi4 / approx / 16-bit DEST.
  A8  {float32, fp32_dest_acc_en=False} becomes a supported cell.
  A9  epsilon = 0.0 accepted (no domain check).
  A10 the per-channel shape rule is a LOGICAL FLOOR, and the blocked (Wt, 32)
      ROW_MAJOR physical form is accepted.
  A11 a host-resident input raises at the entry point.
  A12 the public surface is renamed and takes `weight` rather than `gamma`.
  A13 a torch reference consuming EVERY operand is exported alongside the op.

Phase 0 scheme (unchanged from the seed) -- row-parallel, multi-core,
coarse-blocked, with a cross-core width combine where the row axis cannot fill
the grid:

  * the independent `row` axis (all leading dims folded, incl. H) is split
    across the FULL compute grid with split_work_to_cores(..., row_wise=True);
  * each core walks its assignment in the coarsest whole-row block that fits
    L1 (BLOCK_ROWS);
  * the dependent `width` axis stays inside a core, taken in ONE chunk
    (WT_CHUNK == Wt) whenever the working set fits, and chunked only as an L1
    fallback (the ROW_RESIDENT / STREAM regimes);
  * where `row` leaves the grid under-filled, `width` is additionally split
    across a group of cores and the partials are gathered / finalized /
    multicast back.

Both layouts are native (no host-side to_layout / tilize / untilize / pad /
slice), and H and/or W need not be multiples of 32.
"""

from __future__ import annotations

from typing import Any, Optional

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from bsg_descriptor import (
    _div_up,
    create_program_descriptor,
    dest_tile_limit,
    per_channel_form,
    resolve_program_config,
)

TILE_DIM = 32

#: dtypes accepted for a per-channel operand (weight / bias).  X-08: the SAME
#: set at BOTH layouts, checked at both.
#:
#: This is the op's KERNEL CAPABILITY -- the widest set the per-channel reader
#: and its CB format derivation can carry -- and it is deliberately a SUPERSET
#: invariant over the registry axis, asserted below.  The two refusals are
#: different in kind and must stay different in TYPE:
#:
#:   * a dtype outside THIS set is an input-contract violation ("this op does
#:     not accept a bfloat4_b weight at all") and raises ValueError, which is
#:     what `eval/prompts/rms_norm_ttnn.txt` "## Validation" requires and what
#:     `test_validation.py::test_refuses_per_channel_dtype_outside_the_accepted_set`
#:     asserts;
#:   * a dtype inside this set but outside `SUPPORTED["gamma_dtype"]` is a
#:     registry support refusal and raises `UnsupportedAxisValue`
#:     (a NotImplementedError), which is what the golden suite's xfail-strict
#:     gate requires.
#:
#: Today the two sets coincide, so only the first refusal is reachable.  The
#: superset assertion is what keeps the second reachable if a future refinement
#: ever narrows the axis: `_check_per_channel` runs BEFORE validate()'s
#: SUPPORTED loop, so a narrowed axis value must pass this gate to reach the
#: loop that is supposed to refuse it.
PER_CHANNEL_DTYPES = (ttnn.float32, ttnn.bfloat16, ttnn.bfloat8_b)


# ---------------------------------------------------------------------------
# Compute-config contract (op_design.md A6 / A7)
# ---------------------------------------------------------------------------
#
# Single source of truth for what `compute_kernel_config=None` means.  The
# golden axis-tagger imports this same factory, so the default must never be
# inlined anywhere else.


def default_compute_kernel_config() -> "ttnn.ComputeConfigDescriptor":
    """A7: HiFi4 math, approximate SFPU, 16-bit DEST accumulation.

    A real precision AND performance point, not a placeholder: it is the cell a
    caller who omits the config gets at every input dtype, which is exactly why
    {float32, fp32_dest_acc_en=False} is a SUPPORTED cell (A8) rather than an
    exclusion -- refusing it would make the op's own default unreachable for a
    float32 input.

    A fresh object per call: the descriptor is mutable, so a shared constant
    would let one caller's edit leak into the next call's default.
    """
    return ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=False,
        math_approx_mode=True,
    )


#: Fields the adapter carries across from a DEVICE compute-kernel config onto
#: the generic-op descriptor.  `packer_l1_acc` is deliberately absent: X-04
#: says accept the field and read nothing off it, so it stays on the caller's
#: object and configures nothing here.
_COMPUTE_CONFIG_FIELDS = ("math_fidelity", "math_approx_mode", "fp32_dest_acc_en", "dst_full_sync_en")


def normalize_compute_kernel_config(cfg) -> "ttnn.ComputeConfigDescriptor":
    """A6: accept EITHER config object type, at the door.

    `ttnn.ComputeConfigDescriptor` (what generic_op's ComputeKernelDescriptor
    takes) passes through untouched -- so a descriptor-carrying call builds
    byte-identically to the seed.  A device compute-kernel config
    (`init_device_compute_kernel_config`, i.e. Wormhole/Grayskull/Blackhole
    ComputeKernelConfig) is copied field-for-field onto a fresh descriptor.
    This is an adapter, not a change of program-construction API: everything
    downstream still sees one descriptor.
    """
    if cfg is None:
        return default_compute_kernel_config()
    if isinstance(cfg, ttnn.ComputeConfigDescriptor):
        return cfg
    out = default_compute_kernel_config()
    for field in _COMPUTE_CONFIG_FIELDS:
        value = getattr(cfg, field, None)
        if value is not None and hasattr(out, field):
            setattr(out, field, value)
    return out


# ---------------------------------------------------------------------------
# A13. Torch reference — consumes EVERY operand
# ---------------------------------------------------------------------------


def torch_rms_norm_ttnn(input_tensor, *, epsilon: float = 1e-12, weight=None, bias=None, residual_input_tensor=None):
    """The five stages, in fp32, in order, returned in the input's dtype.

    A reference that accepts an operand and drops it reports agreement for a
    call it never modelled, so every operand this op takes is consumed here:
    the residual enters BEFORE the statistics and the bias AFTER the scale.

    Rank 0 has no reduced dimension -- mean(t^2) over a one-element row is t^2,
    so the scalar case is t / sqrt(t^2 + epsilon) and a zero scalar comes out
    ZERO rather than NaN.  A zero-volume input is returned unchanged.
    """
    import torch

    original_dtype = input_tensor.dtype
    t = input_tensor.to(torch.float32)
    if residual_input_tensor is not None:
        t = t + residual_input_tensor.to(torch.float32)
    if t.numel() == 0:
        return t.to(original_dtype)
    mean_sq = t * t if t.dim() == 0 else torch.mean(t * t, dim=-1, keepdim=True)
    y = t / torch.sqrt(mean_sq + epsilon)
    if weight is not None:
        y = y * weight.to(torch.float32).reshape(-1)
    if bias is not None:
        y = y + bias.to(torch.float32).reshape(-1)
    return y.to(original_dtype)


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS
# ---------------------------------------------------------------------------


def tag_alignment(inputs, axes):
    """Three-value tile-alignment split (feature_spec.py:42-45).

    w_non_aligned dominates h_non_aligned: a non-tile-aligned W drives the
    masked-reduce path (partial scaler + logical-W divisor), which is a
    genuinely different kernel path from H row padding.

    A3/X-10: tolerant of rank < 2.  A tensor with no last extent has W = 1 (the
    rank-0 reduction is over one element) and a tensor with no second-to-last
    extent is not MIS-aligned, so H reads as one tile height.
    """
    shape = inputs[0]
    width = shape[-1] if len(shape) >= 1 else 1
    height = shape[-2] if len(shape) >= 2 else TILE_DIM
    if width % TILE_DIM != 0:
        return "w_non_aligned"
    if height % TILE_DIM != 0:
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
# "none" on gamma_dtype / gamma_layout is the "no per-channel operand"
# sentinel and is ALWAYS legal (see eval/prompts/rms_norm_ttnn.txt).

SUPPORTED = {
    # The full float precision surface.  Every CB's data_format is derived from
    # the dtype of the tensor it carries (activation / weight / bias / output)
    # in rms_norm_ttnn_program_descriptor.py, so block-float rides the same
    # path as bf16 -- see that file's D5 note.
    "dtype": [ttnn.float32, ttnn.bfloat16, ttnn.bfloat8_b],
    # A8: BOTH DEST accumulation modes at EVERY dtype, float32 included.  The
    # seed refused {float32, False} on policy; A7 makes it the op's own default
    # cell, so refusing it would make the default unreachable.
    "fp32_dest_acc_en": [True, False],
    "layout": [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
    "alignment": ["tile_aligned", "w_non_aligned", "h_non_aligned"],
    # A3: no rank floor, no rank ceiling.  0 and 1 fold onto the ROW_MAJOR path
    # at R = 1; 5 is rank 4 with one more leading factor.
    "rank": [0, 1, 2, 3, 4, 5],
    # A1 + A2: the optional-operand presence axis.  Each value is a distinct
    # COMPILED PROGRAM -- optionality is compile-time specialization, never a
    # materialized identity operand.
    "gamma_mode": [
        "no_gamma",
        "gamma",
        "gamma_bias",
        "bias",
        "residual",
        "gamma_bias_residual",
    ],
    "gamma_dtype": [ttnn.float32, ttnn.bfloat16, ttnn.bfloat8_b, "none"],
    "gamma_layout": [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT, "none"],
    # All four placements.  HEIGHT_SHARDED is the knob-turn (the shard cuts the
    # independent `row` axis, so the shard IS the per-core block and the reduce
    # stays local -- zero-copy CBs, no NoC read for x or the residual);
    # WIDTH/BLOCK_SHARDED are the scheme-change (the shard cuts the dependent
    # `width` axis, so per-core partial sums are gathered to each group's root,
    # finalized there and multicast back).  A ROW_MAJOR shard cutting the width
    # axis takes the BAND scheme.  See the SCHEME_* map in the descriptor.
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
# EMPTY.  The seed's only exclusion -- {float32, fp32_dest_acc_en=False} -- is
# REMOVED by A8: the cell already worked (no kernel change was needed), the
# seed refused it on policy, and A7 makes it the cell the op's own default
# compute config produces.  It is scored at what a 16-bit accumulator can
# deliver (helpers.TOLERANCE_OVERRIDES), which is the honest contract.

EXCLUSIONS: list = []


# The PER_CHANNEL_DTYPES superset invariant, checked at import so it cannot rot.
# `_check_per_channel` runs BEFORE validate()'s SUPPORTED loop, so any value the
# registry axis still lists must clear the kernel-capability gate to reach the
# loop that would refuse it -- otherwise a narrowed axis would surface as a
# ValueError where the golden suite's xfail-strict gate expects a
# NotImplementedError (verify_supported would report `xfail_wrong_mode`).
assert set(SUPPORTED["gamma_dtype"]) - {"none"} <= set(PER_CHANNEL_DTYPES), (
    "rms_norm_ttnn: SUPPORTED['gamma_dtype'] lists a dtype outside PER_CHANNEL_DTYPES; "
    "widen PER_CHANNEL_DTYPES (the kernel-capability set) or narrow the axis"
)


# ---------------------------------------------------------------------------
# 3b. PROPERTIES — non-axis capabilities
# ---------------------------------------------------------------------------

PROPERTIES = {
    # The row axis is split over device.compute_with_storage_grid_size(), and a
    # width split adds a second axis where the row axis under-fills the grid.
    #
    # "verified", not "declared": the Phase-0 golden run records `device_num_cores`
    # per cell (eval/profiling.py reads the core_count of the dominant program in
    # the op's profiler window -- i.e. the grid the program ACTUALLY used, which is
    # exactly the evidence eval/op_template.py names for this field).  Over 23 340
    # measured cells the maximum is 110 = the whole 11x10 Blackhole compute grid,
    # and the grid-filling shapes sit at 64-110 (see verification_report.md's
    # occupancy table).  The single-core cells are the small shapes where the row
    # axis is one tile-row and the width split is correctly gated off.
    "multi_core": {"value": True, "source": "verified"},
    # Every CB page count derives from BLOCK_ROWS / WT_CHUNK / a depth knob,
    # each bounded by the L1 budget predicate in the program descriptor.
    "bounded_cb": {"value": True, "source": "declared"},
    # math_fidelity / math_approx_mode are never gated.
    "math_fidelity": {"value": ["LoFi", "HiFi2", "HiFi3", "HiFi4"], "source": "declared"},
}


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def _is_on_device(tensor) -> bool:
    """A11 / X-07: is this tensor's storage on a device?

    Checked by asking the tensor rather than by try/except-ing `.device()`, so
    the refusal names the storage instead of surfacing whatever the accessor
    faults with.
    """
    device_storage = getattr(ttnn.StorageType, "DEVICE", None)
    if device_storage is None:  # pragma: no cover - binding shape changed
        return True
    return tensor.storage_type() == device_storage


def _check_per_channel(name, operand, input_tensor, width):
    """A10 / X-08 / X-09: one contract for `weight` and for `bias`.

    The shape rule is a LOGICAL FLOOR, not an equality, and it is applied to
    the CHANNEL EXTENT the operand carries -- which is its logical last dim in
    the flat (1, 1, 1, W) form and Wt * 32 in the blocked (Wt, 32) ROW_MAJOR
    one.  Reading the blocked form's trailing 32 as its channel count would
    wrongly refuse every blocked operand whose W exceeds 32.
    """
    if operand.dtype not in PER_CHANNEL_DTYPES:
        # ValueError, not a support refusal: a dtype outside the op's accepted
        # per-channel set is an input-contract violation, which the prompt's
        # "## Validation" list requires be raised as ValueError / RuntimeError.
        # A dtype INSIDE the set but outside SUPPORTED["gamma_dtype"] is the other
        # refusal, and validate()'s SUPPORTED loop raises UnsupportedAxisValue for
        # it -- see PER_CHANNEL_DTYPES' superset invariant.
        raise ValueError(
            f"rms_norm_ttnn: {name} dtype {operand.dtype!r} is not in the accepted "
            f"per-channel set {list(PER_CHANNEL_DTYPES)} (the same set at BOTH layouts; "
            f"got layout {operand.layout})"
        )
    if not _is_on_device(operand):
        raise ValueError(f"rms_norm_ttnn: {name} must be resident on a device; got storage {operand.storage_type()!r}")
    _blocked, channel_extent = per_channel_form(operand, width)
    if channel_extent < width:
        raise ValueError(
            f"rms_norm_ttnn: {name} covers {channel_extent} channels, which is fewer than the "
            f"input's last dimension {width}; a per-channel operand must cover every channel "
            f"(shape {list(operand.shape)}, layout {operand.layout})"
        )
    if operand.layout == ttnn.TILE_LAYOUT:
        padded = list(operand.padded_shape)
        # The reader fetches tile columns 0 .. ceil(W/32)-1 of the operand, so the
        # requirement is that those columns EXIST: the operand's padded width must
        # cover the input's TILE-padded width.
        #
        # Compared against `ceil32(W)` and not against `input_tensor.padded_shape`,
        # because a ROW_MAJOR input has no tile padding at all -- its padded last
        # dim IS its logical W -- while the operand it is handed is a legal TILE
        # tensor padded to 64.  Reading the input's own padded dim there refused
        # every {ROW_MAJOR input, TILE per-channel operand} cell at a non-aligned
        # W (48 golden cells at (1,1,32,50) alone).
        #
        # A FLOOR, like the logical rule above and for the same reason: an operand
        # WIDER than the input is legal at this layout too (tile ids are row-major
        # over the padded grid, so columns 0..Wt-1 are the first Wt tiles whatever
        # the total width) and refusing it would be an equality rule dressed up as
        # a coverage rule.
        tile_padded_width = _div_up(width, TILE_DIM) * TILE_DIM
        if padded[-1] < tile_padded_width:
            raise ValueError(
                f"rms_norm_ttnn: a TILE-layout {name}'s padded last dim {padded[-1]} does not cover the "
                f"input's tile-padded last dim {tile_padded_width} (logical W = {width})"
            )
        if len(padded) >= 2 and padded[-2] != TILE_DIM:
            raise ValueError(
                f"rms_norm_ttnn: a TILE-layout {name}'s padded second-to-last dim must be one tile "
                f"height ({TILE_DIM}); got {padded[-2]} from shape {list(operand.shape)}"
            )


def _check_residual(residual, input_tensor):
    """The residual matches the input EXACTLY -- no broadcast, no promotion."""
    if not _is_on_device(residual):
        raise ValueError(
            f"rms_norm_ttnn: residual_input_tensor must be resident on a device; "
            f"got storage {residual.storage_type()!r}"
        )
    if residual.dtype != input_tensor.dtype:
        raise ValueError(
            f"rms_norm_ttnn: residual_input_tensor dtype {residual.dtype!r} must equal the input's "
            f"{input_tensor.dtype!r}"
        )
    if residual.layout != input_tensor.layout:
        raise ValueError(
            f"rms_norm_ttnn: residual_input_tensor layout {residual.layout!r} must equal the input's "
            f"{input_tensor.layout!r}"
        )
    if list(residual.shape) != list(input_tensor.shape):
        raise ValueError(
            f"rms_norm_ttnn: residual_input_tensor shape {list(residual.shape)} must equal the input's "
            f"{list(input_tensor.shape)}"
        )
    if list(residual.padded_shape) != list(input_tensor.padded_shape):
        raise ValueError(
            f"rms_norm_ttnn: residual_input_tensor padded shape {list(residual.padded_shape)} must equal "
            f"the input's {list(input_tensor.padded_shape)}"
        )
    rc, xc = residual.memory_config(), input_tensor.memory_config()
    if rc.memory_layout != xc.memory_layout or (rc.shard_spec is None) != (xc.shard_spec is None):
        raise ValueError(
            f"rms_norm_ttnn: residual_input_tensor placement {rc.memory_layout!r} must equal the input's "
            f"{xc.memory_layout!r} (shard spec included)"
        )
    if rc.shard_spec is not None and (
        tuple(rc.shard_spec.shape) != tuple(xc.shard_spec.shape)
        or rc.shard_spec.grid != xc.shard_spec.grid
        or rc.shard_spec.orientation != xc.shard_spec.orientation
    ):
        raise ValueError(
            f"rms_norm_ttnn: residual_input_tensor shard spec {rc.shard_spec} must equal the input's "
            f"{xc.shard_spec}"
        )


def validate(
    input_tensor,
    *,
    epsilon: float = 1e-12,
    weight=None,
    bias=None,
    residual_input_tensor=None,
    memory_config=None,
    program_config=None,
    compute_kernel_config=None,
):
    """Runtime support gate. Raises before any device work.

    A9: `epsilon` carries NO domain check -- 0.0 is accepted, and the finalize
    keeps it inside the denominator so a cancelled row comes out zero rather
    than NaN.
    A3/X-10: no rank floor and no rank ceiling.
    """
    # --- A11 / X-07: storage first.  Everything below reads the tensor's
    # placement, and a host tensor has no device to obtain.
    if not _is_on_device(input_tensor):
        raise ValueError(
            f"rms_norm_ttnn: input_tensor must be resident on a device; got storage "
            f"{input_tensor.storage_type()!r}.  Move it with ttnn.to_device / pass device= to "
            f"ttnn.from_torch."
        )

    shape = list(input_tensor.shape)
    width = shape[-1] if len(shape) >= 1 else 1

    if weight is not None:
        _check_per_channel("weight", weight, input_tensor, width)
    if bias is not None:
        _check_per_channel("bias", bias, input_tensor, width)
    if weight is not None and bias is not None and weight.layout != bias.layout:
        raise ValueError(
            f"rms_norm_ttnn: weight and bias must share a layout when both are present; got weight "
            f"layout {weight.layout!r} and bias layout {bias.layout!r}"
        )
    if residual_input_tensor is not None:
        _check_residual(residual_input_tensor, input_tensor)

    cfg = normalize_compute_kernel_config(compute_kernel_config)

    per_channel = weight if weight is not None else bias
    has_weight = weight is not None
    has_bias = bias is not None
    has_residual = residual_input_tensor is not None
    gamma_mode = "_".join(
        name for name, present in (("gamma", has_weight), ("bias", has_bias), ("residual", has_residual)) if present
    )
    axes = {
        "dtype": input_tensor.dtype,
        "layout": input_tensor.layout,
        "fp32_dest_acc_en": bool(getattr(cfg, "fp32_dest_acc_en", False)),
        "gamma_mode": gamma_mode or "no_gamma",
        "gamma_dtype": per_channel.dtype if per_channel is not None else "none",
        "gamma_layout": per_channel.layout if per_channel is not None else "none",
        "memory_layout": input_tensor.memory_config().memory_layout,
    }
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger((shape,), axes)

    # 1. SUPPORTED — per-axis
    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"rms_norm_ttnn: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")

    # 2. EXCLUSIONS — cell-level inside SUPPORTED
    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"rms_norm_ttnn: unsupported combination (refinement candidate): {exc}")

    # The requested output placement must also be one we implement.
    if memory_config is not None and memory_config.memory_layout not in SUPPORTED["memory_layout"]:
        raise UnsupportedAxisValue(
            f"rms_norm_ttnn: memory_config.memory_layout={memory_config.memory_layout!r} not in "
            f"SUPPORTED {SUPPORTED['memory_layout']}"
        )

    # 3. A5 — the caller's program config.  Resolved (and refused) HERE so the
    # entry point's first line is the whole gate; the resolved record is
    # rebuilt by the entry point from the same pure function.
    resolve_program_config(
        input_tensor,
        program_config=program_config,
        memory_config=memory_config,
        dest_limit=dest_tile_limit(cfg),
    )

    return axes


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def rms_norm_ttnn(
    input_tensor: "ttnn.Tensor",
    *,
    epsilon: float = 1e-12,
    weight: Optional["ttnn.Tensor"] = None,
    bias: Optional["ttnn.Tensor"] = None,
    residual_input_tensor: Optional["ttnn.Tensor"] = None,
    memory_config: Optional["ttnn.MemoryConfig"] = None,
    program_config: Optional[Any] = None,
    compute_kernel_config: Optional[Any] = None,
) -> "ttnn.Tensor":
    """RMSNorm over the last dimension, in exactly ONE device program per call.

    Args:
        input_tensor: on-device tensor, any rank >= 0, TILE or ROW_MAJOR layout,
            INTERLEAVED or any *_SHARDED placement.
        epsilon: added to the mean square before the rsqrt.  No domain check;
            0.0 is accepted.
        weight: optional per-channel scale, flat (1, 1, 1, Wg >= W) at either
            layout or blocked (Wt, 32) ROW_MAJOR.  Its dtype and layout are
            independent of the input's.
        bias: optional per-channel shift applied AFTER the scale.  Same shape
            rules as `weight`; must share `weight`'s layout when both are
            present, and its dtype is independent of `weight`'s.
        residual_input_tensor: optional second activation added BEFORE the
            statistics.  Matches the input exactly.
        memory_config: output placement (defaults to the input's).
        program_config: the caller's own blocking (see A5); `subblock_w` is
            honoured, `inplace` returns the INPUT tensor object.
        compute_kernel_config: `ttnn.ComputeConfigDescriptor` or a device
            compute-kernel config; resolved through
            default_compute_kernel_config() when None.
    """
    validate(
        input_tensor,
        epsilon=epsilon,
        weight=weight,
        bias=bias,
        residual_input_tensor=residual_input_tensor,
        memory_config=memory_config,
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
    )

    compute_kernel_config = normalize_compute_kernel_config(compute_kernel_config)
    resolved_pc = resolve_program_config(
        input_tensor,
        program_config=program_config,
        memory_config=memory_config,
        dest_limit=dest_tile_limit(compute_kernel_config),
    )

    device = input_tensor.device()
    output_memory_config = memory_config if memory_config is not None else input_tensor.memory_config()

    if resolved_pc.inplace:
        # A5 / X-03: `inplace` is a contract about what SURVIVES the call --
        # the caller reads the input tensor back afterwards -- so the output IS
        # the input object and cb_output_tiles aliases its buffer.  A
        # disagreeing memory_config was already refused in validate().
        output_tensor = input_tensor
    else:
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
        weight=weight,
        bias=bias,
        residual=residual_input_tensor,
        epsilon=epsilon,
        compute_kernel_config=compute_kernel_config,
        program_config=resolved_pc,
    )

    tensors = [input_tensor]
    for operand in (weight, bias, residual_input_tensor):
        if operand is not None:
            tensors.append(operand)
    tensors.append(output_tensor)
    result = ttnn.generic_op(tensors, program_descriptor)

    if resolved_pc.inplace:
        # A5 / X-03: `inplace` is a contract about the OBJECT, not just about the
        # bytes.  generic_op hands back `io_tensors.back()`, which is a fresh
        # Python Tensor over the same buffer -- correct values, but a caller who
        # sets `inplace` and then reads their own tensor back is relying on
        # IDENTITY, so returning the copy would quietly break exactly the thing
        # the flag is for.  The device program already wrote through
        # cb_output_tiles, which aliases this buffer.
        return input_tensor
    return result
