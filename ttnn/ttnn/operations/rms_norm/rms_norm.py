# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""rms_norm — root-mean-square normalization along the last dimension.

    rms_norm(x) = x * rsqrt(mean(x**2, dim=-1, keepdim=True) + epsilon) * gamma

Registry-model op file: the four declarations (INPUT_TAGGERS, SUPPORTED,
EXCLUSIONS, validate) plus the public entry point.  The kernel schedule and its
blocking model live in `op_design.md` / `l1_ledger.md` next to this file; the
program is built by `rms_norm_program_descriptor.create_program_descriptor`.

Support rectangle (see SUPPORTED below):
  * bfloat16 / float32 / bfloat8_b activations, TILE and ROW_MAJOR, INTERLEAVED
    or HEIGHT/WIDTH/BLOCK sharded (a resident shard is consumed in place).
  * gamma optional, at its own dtype/layout ("none" sentinel when absent).
  * fp32_dest_acc_en at both settings, except for float32 activations
    (see EXCLUSIONS).
"""

from __future__ import annotations

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .rms_norm_program_descriptor import create_program_descriptor


# ---------------------------------------------------------------------------
# Phase 0 compute-kernel-config default — ONE factory, no inlined duplicate.
# ---------------------------------------------------------------------------


def default_compute_kernel_config() -> "ttnn.ComputeConfigDescriptor":
    """The Phase 0 default compute config (a fresh descriptor per call).

    Phase 0 is the maxed-out precision corner: fp32 DEST accumulation on.
    `math_fidelity` / `math_approx_mode` are NOT gated — the values here are
    only the defaults used when the caller passes None.
    """
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi4
    cfg.fp32_dest_acc_en = True
    cfg.math_approx_mode = False
    return cfg


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS
# ---------------------------------------------------------------------------


def tag_alignment(inputs, axes):
    """Three-value alignment split (matches eval/golden_tests/rms_norm)."""
    shape = inputs[0]
    if len(shape) < 2:
        # rank<2 never reaches the kernel (validate raises first); keep the
        # tagger total so a stray call cannot IndexError.
        return "w_non_aligned" if shape[-1] % 32 else "tile_aligned"
    w_aligned = shape[-1] % 32 == 0
    h_aligned = shape[-2] % 32 == 0
    if w_aligned and h_aligned:
        return "tile_aligned"
    if not w_aligned:
        return "w_non_aligned"
    return "h_non_aligned"


def tag_rank(inputs, axes):
    return int(len(inputs[0]))


INPUT_TAGGERS = {
    "alignment": tag_alignment,
    "rank": tag_rank,
}


# ---------------------------------------------------------------------------
# 2. SUPPORTED
# ---------------------------------------------------------------------------

SUPPORTED = {
    "dtype": [ttnn.float32, ttnn.bfloat16, ttnn.bfloat8_b],
    # Refinement 1 opened the bf16-DEST corner. The whole `cb_stat_*` path stays
    # fp32 in L1 at BOTH settings, so only the in-DEST accumulation narrows.
    # {float32, fp32_dest_acc_en=False} stays refused (see EXCLUSIONS).
    "fp32_dest_acc_en": [True, False],
    "layout": [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
    "alignment": ["tile_aligned", "w_non_aligned", "h_non_aligned"],
    "rank": [2, 3, 4],
    "gamma_mode": ["gamma", "no_gamma"],
    # "none" is the absent-gamma sentinel and is ALWAYS legal.
    "gamma_dtype": [ttnn.float32, ttnn.bfloat16, ttnn.bfloat8_b, "none"],
    "gamma_layout": [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT, "none"],
    # Refinement 2: all four placements. The three sharded schemes are placement
    # unlocks of the scheme Phase 0 already built (op_design.md lamp S1) — the
    # cross-core combine exists, so the shard spec SUPPLIES the block geometry
    # instead of `_select_regime` choosing it, and the resident shard is consumed
    # in place through a zero-copy CB pinned over its L1 buffer.
    #   HEIGHT ⇒ cuts the independent `row` axis  ⇒ w_group_size == 1, local reduce
    #   WIDTH  ⇒ cuts the dependent `hidden` axis ⇒ the shard grid is ONE group
    #   BLOCK  ⇒ cuts both ⇒ one grid row of the shard rectangle is one group
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
# Cells inside cartesian(SUPPORTED) refused for now. {float32,
# fp32_dest_acc_en=False} is the canonical precision exclusion, and since
# Refinement 1 put False in SUPPORTED it is REACHABLE and actively enforced:
# asking for fp32 activations while narrowing DEST to 16 bits contradicts the
# only reason to pay for an fp32 activation tensor. Callers who want the narrow
# DEST datapath pass bfloat16 or bfloat8_b activations.
#
# NOTHING ELSE is excluded. Every other cell Refinement 1 opened was measured
# green: bfloat16/bfloat8_b at fp32_dest_acc_en=False, dtype=bfloat8_b and
# gamma_dtype=bfloat8_b, including the mask-before-square ragged-hidden path
# under a 16-bit DEST accumulator. bfloat8_b x ROW_MAJOR and bfloat8_b x
# non-tile-aligned are NOT listed: they are structurally impossible (ttnn itself
# refuses to build a ROW_MAJOR block-float tensor) and live in the golden
# suite's feature_spec.INVALID, so they are skipped, not refused.

EXCLUSIONS = [
    {"dtype": ttnn.float32, "fp32_dest_acc_en": False},
]


# ---------------------------------------------------------------------------
# 3b. PROPERTIES
# ---------------------------------------------------------------------------

PROPERTIES = {
    # Verified: the program descriptor partitions device.compute_with_storage_grid_size()
    # into reduction groups and emits per-core runtime args for every active core.
    "multi_core": {"value": True, "source": "verified"},
    # Declared: every CB page count is an expression in block_row_tiles /
    # core_w_tiles / w_group_size, all bounded by the L1 residency solve.
    "bounded_cb": {"value": True, "source": "declared"},
    "math_fidelity": {
        "value": ["LoFi", "HiFi2", "HiFi3", "HiFi4"],
        "source": "declared",
    },
}


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def validate(input_tensor, *, gamma=None, epsilon=1e-6, compute_kernel_config=None, **_):
    """Runtime support gate. Raises before any device work."""
    shape = list(input_tensor.shape)

    # Structural preconditions first — these are ValueError (a caller mistake),
    # not a support refusal. The error text is a contract: it must name what it
    # rejects ("rank" / "gamma") so a caller can tell them apart.
    if len(shape) < 2:
        raise ValueError(f"rms_norm: input tensor rank must be >= 2, got rank {len(shape)} (shape {tuple(shape)})")
    if gamma is not None:
        gamma_shape = list(gamma.shape)
        if gamma_shape[-1] != shape[-1]:
            raise ValueError(
                f"rms_norm: gamma last dimension ({gamma_shape[-1]}) must match the input's "
                f"last dimension ({shape[-1]})"
            )
        if any(d != 1 for d in gamma_shape[:-1]):
            raise ValueError(f"rms_norm: gamma leading dimensions must all be 1, got gamma shape {tuple(gamma_shape)}")

    cfg = compute_kernel_config if compute_kernel_config is not None else default_compute_kernel_config()

    axes = {
        "dtype": input_tensor.dtype,
        "fp32_dest_acc_en": bool(getattr(cfg, "fp32_dest_acc_en", True)),
        "layout": input_tensor.layout,
        "gamma_mode": "gamma" if gamma is not None else "no_gamma",
        "gamma_dtype": gamma.dtype if gamma is not None else "none",
        "gamma_layout": gamma.layout if gamma is not None else "none",
        "memory_layout": input_tensor.memory_config().memory_layout,
    }
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger((shape,), axes)

    # 1. SUPPORTED — per axis.
    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"rms_norm: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")

    # 2. EXCLUSIONS — cell-level inside SUPPORTED.
    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"rms_norm: unsupported combination (refinement candidate): {exc}")

    if epsilon is None or epsilon <= 0.0:
        raise ValueError(f"rms_norm: epsilon must be > 0, got {epsilon!r}")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def rms_norm(
    input_tensor: "ttnn.Tensor",
    *,
    gamma=None,
    epsilon: float = 1e-6,
    compute_kernel_config=None,
    memory_config=None,
) -> "ttnn.Tensor":
    """Root-mean-square normalization over the last dimension.

    Args:
        input_tensor: rank >= 2 activation tensor on device (TILE or ROW_MAJOR).
        gamma: optional per-column scale with last dim == input's last dim.
        epsilon: numerical-stability term added inside the rsqrt.
        compute_kernel_config: ttnn.ComputeConfigDescriptor; None resolves
            through default_compute_kernel_config().
        memory_config: output memory config; defaults to the input's.

    Returns:
        A tensor with the input's shape, dtype and layout.
    """
    validate(
        input_tensor,
        gamma=gamma,
        epsilon=epsilon,
        compute_kernel_config=compute_kernel_config,
    )

    cfg = compute_kernel_config if compute_kernel_config is not None else default_compute_kernel_config()

    device = input_tensor.device()
    out_memory_config = memory_config if memory_config is not None else input_tensor.memory_config()

    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(input_tensor.shape)),
        input_tensor.dtype,
        input_tensor.layout,
        device,
        out_memory_config,
    )

    program_descriptor = create_program_descriptor(
        input_tensor,
        gamma,
        output_tensor,
        epsilon=epsilon,
        compute_kernel_config=cfg,
    )

    io_tensors = [input_tensor]
    if gamma is not None:
        io_tensors.append(gamma)
    io_tensors.append(output_tensor)  # output MUST be last

    return ttnn.generic_op(io_tensors, program_descriptor)
