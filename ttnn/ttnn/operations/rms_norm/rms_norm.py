# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""RMSNorm operation (registry model).

    RMSNorm(x) = x / sqrt(mean(x^2, dim=-1, keepdim=True) + epsilon) * gamma

Phase-1 scheme (see op_design.md §1): row-parallel, bounded two-pass streaming
reduce over the last dim W, multi-core from day 1. Independent tile-rows
(`R = NC * ceil(H/32)`) are spread across the whole grid; within a core each
tile-row streams its W twice through fixed-size CBs. Every block knob
(BLOCK_SIZE, DEPTH, grid) is a parameter at its trivial value — never an inlined
constant.

Native RM + TILE input, native non-tile-aligned H/W. No host-side
layout/pad workarounds — the kernels do the RM↔tile conversion and the
partial-W / partial-H masking natively.
"""

from __future__ import annotations

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .rms_norm_program_descriptor import create_program_descriptor


# ---------------------------------------------------------------------------
# Compute-config default (single source of truth)
# ---------------------------------------------------------------------------
# Phase-0 is the maxed-out precision corner: fp32_dest_acc_en=True, HiFi4.
# validate() and the golden axis-tagger both read this factory — never inline
# the default elsewhere (see references/precision_convention.md).


def default_compute_kernel_config():
    return ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        math_approx_mode=False,
    )


def _resolve_config(compute_kernel_config):
    return compute_kernel_config if compute_kernel_config is not None else default_compute_kernel_config()


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS  (shape-derived axes; gamma/precision axes are read directly)
# ---------------------------------------------------------------------------


def tag_alignment(inputs, axes):
    """Three-value split matching feature_spec:
    tile_aligned  — both H(-2) and W(-1) divisible by 32.
    w_non_aligned — W not divisible by 32 (H may or may not be).
    h_non_aligned — W aligned, H not aligned.
    """
    shape = inputs[0]
    H, W = int(shape[-2]), int(shape[-1])
    if W % 32 != 0:
        return "w_non_aligned"
    if H % 32 != 0:
        return "h_non_aligned"
    return "tile_aligned"


def tag_rank(inputs, axes):
    return len(inputs[0])


INPUT_TAGGERS = {
    "alignment": tag_alignment,
    "rank": tag_rank,
}


# ---------------------------------------------------------------------------
# 2. SUPPORTED  (phase-1: narrow, but one entry per TARGET axis)
# ---------------------------------------------------------------------------
# Every axis the golden feature_spec TARGET enumerates gets an entry so
# out-of-rectangle cells xfail cleanly instead of over-claiming.
#
#   * dtype            — bf16/f32/bf8b native. bf8b is a block-float format with
#                        no RM representation and is only exercised tile-aligned
#                        (bf8b+RM and bf8b+non_aligned are INVALID in
#                        feature_spec, so no EXCLUSION is needed for those).
#   * fp32_dest_acc_en — True (maxed corner, HiFi4) OR False (bf16 DEST accum,
#                        the perf-loose config). {f32,False} is lossy → EXCLUSION.
#   * layout           — TILE and ROW_MAJOR, both native.
#   * alignment        — tile / W-nonaligned / H-nonaligned, all native.
#   * rank             — 2/3/4.
#   * gamma_mode       — optional scale (present / absent).
#   * gamma_dtype      — real dtype when present (bf16/f32/bf8b), "none" absent.
#   * gamma_layout     — RM gamma (phase-1 contract) OR TILE gamma (native tiled
#                        reader, Refinement 2); "none" when absent. bf8b gamma
#                        implies TILE gamma (block-float has no RM form), so TILE
#                        gamma also unlocks the bf8b-gamma cells.
#   * memory_layout    — INTERLEAVED (the sharded schemes are §1 lamps).

SUPPORTED = {
    "dtype": [ttnn.float32, ttnn.bfloat16, ttnn.bfloat8_b],
    "fp32_dest_acc_en": [True, False],
    "layout": [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
    "alignment": ["tile_aligned", "w_non_aligned", "h_non_aligned"],
    "rank": [2, 3, 4],
    "gamma_mode": ["gamma", "no_gamma"],
    "gamma_dtype": [ttnn.float32, ttnn.bfloat16, ttnn.bfloat8_b, "none"],
    "gamma_layout": [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT, "none"],
    "memory_layout": [
        ttnn.TensorMemoryLayout.INTERLEAVED,
        # Local per-core reduction (Refinement 5): HEIGHT sharding splits rows across
        # cores, each core keeps FULL-W rows, so the RMS reduce stays LOCAL per core.
        # The row-shard is resident in each core's L1 -> cb_x_in/cb_out backed zero-copy
        # on the sharded buffers (no NoC read/write), reusing the interleaved row-parallel
        # compute unchanged (op_design.md §1 lamp 3 — a knob-turn, not a cross-core scheme).
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        # Cross-core W-split (Refinement 4): the hidden dim is pre-placed across a
        # group of cores; each reduces its local W-slice to a partial Σx², then one
        # cross-core round (gather -> fold -> broadcast 1/RMS) precedes normalize.
        # Slices consumed locally via zero-copy sharded CBs (op_design.md §1 lamp 2).
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
    ],
}


# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------
# {float32, fp32_dest_acc_en=False} is the design's legal-but-refused precision
# corner (references/precision_convention.md): fp32 activations with non-fp32
# DEST accumulation is lossy/nonsensical. Now that fp32_dest_acc_en=False is in
# SUPPORTED (for the bf16 perf-loose config), this cell is inside the SUPPORTED
# rectangle and must be refused cell-level so {f32,False} stays xfail-strict.

EXCLUSIONS = [
    {"dtype": ttnn.float32, "fp32_dest_acc_en": False},
    # Refinement 5 lands HEIGHT_SHARDED for TILE input (TILE gamma, RM gamma, no gamma)
    # via the zero-copy resident-shard row-parallel path. RM INPUT + HEIGHT still needs the
    # tilize/untilize-on-resident-shard plumbing (a local-reduction analog of the R4b RM
    # sub-scheme) — deferred to R5a. Refused cell-level so it stays xfail-strict:
    #   * RM INPUT + HEIGHT: covers RM+no-gamma and RM+RM-gamma (RM+TILE-gamma is INVALID).
    {"layout": ttnn.ROW_MAJOR_LAYOUT, "memory_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED},
    # Refinement 4a lands RM-gamma on the cross-core path (reader reads the gamma
    # W-slice as row-major sticks, compute tilizes them into cb_gamma before the
    # pass-2 ·gamma). Refinement 4b lands RM-INPUT + WIDTH/BLOCK sharded: the reader
    # loopback-repacks the resident RM shard's arbitrary-width sticks into ceil(sw/32)
    # padded tiles, EVERY core zeros its last tile's sub-tile lanes via a per-core
    # partial scaler, the shared cross-core combine sums the (associative) partials,
    # and the writer untilizes back to the resident RM shard. Both sets of
    # {gamma_layout/layout: ROW_MAJOR, memory_layout: *_SHARDED} exclusions are removed.
]


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def validate(input_tensor, *, gamma=None, epsilon=1e-6, compute_kernel_config=None, memory_config=None):
    # --- Hard input errors (contract): raise ValueError before the axis gate. ---
    if len(input_tensor.shape) < 2:
        raise ValueError(f"rms_norm: input must have rank >= 2, got {len(input_tensor.shape)}")
    if gamma is not None and int(gamma.shape[-1]) != int(input_tensor.shape[-1]):
        raise ValueError(
            f"rms_norm: gamma last dim {int(gamma.shape[-1])} does not match "
            f"input last dim {int(input_tensor.shape[-1])}"
        )

    config = _resolve_config(compute_kernel_config)
    has_gamma = gamma is not None

    # Build the axes dict the same way the golden harness does: tensor
    # properties + compute-config precision + gamma facets + shape taggers.
    axes = {
        "dtype": input_tensor.dtype,
        "layout": input_tensor.layout,
        "fp32_dest_acc_en": bool(getattr(config, "fp32_dest_acc_en", True)),
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


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def rms_norm(
    input_tensor: ttnn.Tensor,
    *,
    gamma: "ttnn.Tensor | None" = None,
    epsilon: float = 1e-6,
    compute_kernel_config: "ttnn.ComputeConfigDescriptor | None" = None,
    memory_config: "ttnn.MemoryConfig | None" = None,
) -> ttnn.Tensor:
    """RMS-normalize `input_tensor` along its last dim, with optional scale `gamma`.

    Layout and shape of the output match the input (RM→RM, TILE→TILE); no
    host-side layout/pad transform is applied.
    """
    validate(
        input_tensor,
        gamma=gamma,
        epsilon=epsilon,
        compute_kernel_config=compute_kernel_config,
        memory_config=memory_config,
    )

    config = _resolve_config(compute_kernel_config)
    device = input_tensor.device()
    out_mem = memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG

    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(input_tensor.shape)),
        input_tensor.dtype,
        input_tensor.layout,
        device,
        out_mem,
    )

    program_descriptor = create_program_descriptor(
        input_tensor,
        output_tensor,
        gamma=gamma,
        epsilon=epsilon,
        compute_kernel_config=config,
    )

    # Output tensor MUST be last; gamma (optional) is referenced by the reader's
    # second TensorAccessor.
    tensors = [input_tensor]
    if gamma is not None:
        tensors.append(gamma)
    tensors.append(output_tensor)
    return ttnn.generic_op(tensors, program_descriptor)
