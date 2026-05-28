# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
layer_norm_rm — Per-row (final-dim) LayerNorm on a ROW_MAJOR fp32 tensor.

Phase-0 implementation of the spec in op_design.md.

Pipeline (per tile-row work item, see op_design.md for the full reduce/sub/
reduce chain):

    reader   : DRAM RM sticks → cb_input_rm (tile-paged)
             + one-shot replicate-32× of optional gamma/beta sticks
             + one-shot scaler tile (1/W) for both reductions
    compute  : tilize → reduce(SUM, REDUCE_ROW) [mean]
             → sub<COL> [centered]
             → square [(x - mean)^2]
             → reduce(SUM, REDUCE_ROW) + (+eps, rsqrt) post-op [inv_std]
             → mul<COL> [normalized]
             → optional mul_in_place<ROW> by gamma
             → optional add_in_place<ROW> by beta
             → untilize → cb_output_tiles
    writer   : cb_output_tiles → DRAM RM sticks

Output shape, dtype (fp32), and layout (ROW_MAJOR) match input. Tilize and
untilize happen entirely in-kernel — the entry point does NOT cast to TILE.

Phase-0 envelope (SUPPORTED below):
- input dtype = float32; input layout = ROW_MAJOR_LAYOUT
- rank ∈ {2, 3, 4}; H % 32 == 0; W % 32 == 0; H ≥ 32; W ≥ 32
- optional gamma / beta — fp32, ROW_MAJOR, total element count == W
- epsilon strictly > 0

Memory budget: the per-core L1 CB footprint scales ~10 × Wt × 4 KB with
gamma+beta and ~6 × Wt × 4 KB without. Wide shapes overshoot the 1.5 MB
budget — empirically W ≥ 1024 with gamma+beta and W ≥ ~~1500~~ without.
Such cells fail with an `OOM` at program creation; they are not gated in
validate() (the failure mode is the signal) and are queued in
op_requirements.md as the W-axis chunking refinement.

EXCLUSIONS: gamma_only/gamma_beta + affine_layout=TILE — gamma/beta are
unconditionally read as RM sticks; tile-layout affine inputs require a
tilize-bypass reader and are a later refinement. The canonical
(affine=no_affine, affine_layout=TILE) cell stays inside the SUPPORTED
rectangle because the kernel ignores the affine_layout value entirely
when neither gamma nor beta is supplied.
"""

import ttnn

from .layer_norm_rm_program_descriptor import create_program_descriptor


_DEFAULT_EPSILON = 1e-5


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS
# ---------------------------------------------------------------------------
#
# Shape-derived categorical axes (per registry model):
# - `alignment`: three-way split per feature_spec.py expectation —
#     * tile_aligned   (both H and W % 32 == 0)
#     * w_non_aligned  (W % 32 != 0)
#     * h_non_aligned  (W % 32 == 0 and H % 32 != 0)
# - `rank`: input tensor rank.


def tag_alignment(inputs, axes):
    shape = inputs[0]
    H = shape[-2]
    W = shape[-1]
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
# 2. SUPPORTED
# ---------------------------------------------------------------------------
#
# Phase-0 envelope. Both layouts appear under `affine_layout` so the
# canonical (no_affine, TILE) cell — the only no_affine cell after
# feature_spec.py's INVALID canonicalization — falls inside the rectangle.
# Affine-bearing TILE cells are then carved out via EXCLUSIONS below.

SUPPORTED = {
    "dtype": [ttnn.float32],
    "layout": [ttnn.ROW_MAJOR_LAYOUT],
    "alignment": ["tile_aligned"],
    "rank": [2, 3, 4],
    "affine": ["gamma_beta", "gamma_only", "no_affine"],
    "affine_dtype": [ttnn.float32],
    "affine_layout": [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
}


# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------
#
# When gamma or beta is actually present, the kernel reads them as RM
# sticks (the reader replicates the single stick 32× into the gamma_rm /
# beta_rm CBs, then the compute side runs tilize). A TILE-layout affine
# tensor would need a different reader path. Phase 0 refuses the cell;
# a future refinement adds tilize-bypass.

EXCLUSIONS = [
    {"affine": "gamma_only", "affine_layout": ttnn.TILE_LAYOUT},
    {"affine": "gamma_beta", "affine_layout": ttnn.TILE_LAYOUT},
]


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def validate(input_tensor, gamma=None, beta=None, *, epsilon=1e-5):
    # --- Build axes dict (tagger-aware, matches the harness) -------------
    if gamma is None and beta is None:
        affine = "no_affine"
    elif beta is None:
        affine = "gamma_only"
    else:
        affine = "gamma_beta"

    # affine_dtype / affine_layout follow gamma (or beta when only beta is
    # supplied); canonicalize to (float32, TILE) for no_affine — mirrors
    # feature_spec.py's INVALID canonicalization (which marks the
    # ROW_MAJOR variant of no_affine as the redundant cell).
    if gamma is not None:
        affine_dtype = gamma.dtype
        affine_layout = gamma.layout
    elif beta is not None:
        affine_dtype = beta.dtype
        affine_layout = beta.layout
    else:
        affine_dtype = ttnn.float32
        affine_layout = ttnn.TILE_LAYOUT

    axes = {
        "dtype": input_tensor.dtype,
        "layout": input_tensor.layout,
        "affine": affine,
        "affine_dtype": affine_dtype,
        "affine_layout": affine_layout,
    }
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger((tuple(input_tensor.shape),), axes)

    # 1. SUPPORTED — per-axis
    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise NotImplementedError(f"layer_norm_rm: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")

    # 2. EXCLUSIONS — cell-level
    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise NotImplementedError(f"layer_norm_rm: unsupported combination (refinement candidate): {exc}")

    # --- Non-axis structural guards (kernel-correctness preconditions) ---
    shape = list(input_tensor.shape)
    if len(shape) < 2:
        raise NotImplementedError(f"layer_norm_rm: input rank {len(shape)} not in SUPPORTED [rank >= 2]")
    H = shape[-2]
    W = shape[-1]
    if H < 32:
        raise NotImplementedError(f"layer_norm_rm: H={H} not in SUPPORTED [H >= 32 and H % 32 == 0]")
    if W < 32:
        raise NotImplementedError(f"layer_norm_rm: W={W} not in SUPPORTED [W >= 32 and W % 32 == 0]")

    # NOTE: no explicit W cap here. The per-core L1 budget (~10 × Wt × 4 KB
    # with gamma+beta) bounds the workable W, but the kernel will surface
    # an `OOM` at CB-allocation time rather than a NotImplementedError —
    # we let the failure mode be the refinement signal (per the verifier
    # protocol).

    # Affine tensor numel check — total element count must match input W.
    for name, t in (("gamma", gamma), ("beta", beta)):
        if t is None:
            continue
        affine_numel = 1
        for d in list(t.shape):
            affine_numel *= d
        if affine_numel != W:
            raise NotImplementedError(f"layer_norm_rm: {name} numel {affine_numel} != input W {W}")

    if not (epsilon > 0):
        raise NotImplementedError(f"layer_norm_rm: epsilon={epsilon} not in SUPPORTED [epsilon > 0]")


def layer_norm_rm(
    input_tensor: ttnn.Tensor,
    gamma: ttnn.Tensor = None,
    beta: ttnn.Tensor = None,
    *,
    epsilon: float = _DEFAULT_EPSILON,
    memory_config: ttnn.MemoryConfig = None,
) -> ttnn.Tensor:
    """
    Per-row LayerNorm over the final dimension.

    Args:
        input_tensor: fp32 ROW_MAJOR tensor (rank ≥ 2, tile-aligned H/W).
        gamma: optional fp32 ROW_MAJOR scale, total element count == input W.
        beta:  optional fp32 ROW_MAJOR shift, total element count == input W.
        epsilon: numerical stability term added before rsqrt; default 1e-5.
        memory_config: output memory config; default DRAM_MEMORY_CONFIG.

    Returns:
        Output tensor with the same shape, dtype, and layout as input.
    """
    validate(input_tensor, gamma, beta, epsilon=epsilon)

    device = input_tensor.device()
    output_memory_config = memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG

    # Output has identical shape, dtype, and layout to the input.
    output_shape = list(input_tensor.shape)
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(output_shape),
        input_tensor.dtype,
        input_tensor.layout,
        device,
        output_memory_config,
    )

    program_descriptor = create_program_descriptor(
        input_tensor,
        output_tensor,
        gamma=gamma,
        beta=beta,
        epsilon=epsilon,
    )

    # Output tensor MUST be last in the IO tensor list.
    io_tensors = [input_tensor]
    if gamma is not None:
        io_tensors.append(gamma)
    if beta is not None:
        io_tensors.append(beta)
    io_tensors.append(output_tensor)

    return ttnn.generic_op(io_tensors, program_descriptor)
