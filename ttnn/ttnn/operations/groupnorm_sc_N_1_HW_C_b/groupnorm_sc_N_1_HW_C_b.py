# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
groupnorm_sc_N_1_HW_C_b — GroupNorm over an (N, 1, HW, C) channels-last activation.

One `ttnn.generic_op` dispatch per call. Two-pass statistics (Σx, Σx² per channel) on a
row-split, multi-core grid; per-core partials are collapsed to group lanes, gathered to a
per-unit root, finalized to [rstd, mean], multicast back, expanded to per-channel
a = rstd·γ and b' = mean·a − β, and applied as y = x·a − b'. Design: op_design.md.
"""

from __future__ import annotations

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue


# ---------------------------------------------------------------------------
# Tuning overrides (module-level, mutated in place; never rebound)
# ---------------------------------------------------------------------------
#
# None = use the derived default. Read at every call and folded into the program
# descriptor (CB sizes, RT args), so a monkeypatched value always takes effect.
TUNING_OVERRIDES = {
    "max_resident_chunks": None,
    "chunk_tiles_target": None,
    "max_cores": None,
}


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS (verbatim from the task statement)
# ---------------------------------------------------------------------------


def _placement(inputs):
    return inputs[2] if len(inputs) > 2 else None


def _ceil(n, m):
    return -(-n // m)


def tag_alignment(inputs, axes):
    shape = inputs[0]
    HW, C = shape[-2], shape[-1]
    if HW % 32 == 0 and C % 32 == 0:
        return "tile_aligned"
    if C % 32 != 0:
        return "c_non_aligned"
    return "hw_non_aligned"


def tag_memory_layout(inputs, axes):
    if _placement(inputs) is None:
        return ttnn.TensorMemoryLayout.INTERLEAVED
    return ttnn.TensorMemoryLayout.BLOCK_SHARDED


def tag_shard_fit(inputs, axes):
    p = _placement(inputs)
    if p is None:
        return "none"
    gx, gy = p
    N, _, HW, C = inputs[0]
    rows = N * HW
    hg, wg = (32, 32) if axes["layout"] == ttnn.TILE_LAYOUT else (1, 8)
    sh = _ceil(_ceil(rows, gy), hg) * hg
    sw = _ceil(_ceil(C, gx), wg) * wg
    if (gy - 1) * sh >= rows or (gx - 1) * sw >= C:
        return "illegal"
    if sh * gy == rows and sw * gx == C:
        return "even"
    return "ragged"


INPUT_TAGGERS = {
    "alignment": tag_alignment,
    "memory_layout": tag_memory_layout,
    "shard_fit": tag_shard_fit,
}


# ---------------------------------------------------------------------------
# 2. SUPPORTED (Phase 0)
# ---------------------------------------------------------------------------

SUPPORTED = {
    "dtype": [ttnn.bfloat16],
    "layout": [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
    "alignment": ["tile_aligned"],
    "affine": ["gamma_beta", "gamma_only", "no_affine"],
    "affine_dtype": [ttnn.bfloat16, "none"],
    "affine_layout": [ttnn.ROW_MAJOR_LAYOUT, "none"],
    "memory_layout": [ttnn.TensorMemoryLayout.INTERLEAVED],
    "shard_fit": ["none"],
    "in_place": [False],
}


# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------

EXCLUSIONS = []


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def _shard_placement(input_tensor):
    """(grid_x, grid_y) of a sharded input's shard grid, or None for an interleaved one."""
    if not input_tensor.is_sharded():
        return None
    spec = input_tensor.shard_spec()
    if spec is None:
        return None
    bbox = spec.grid.bounding_box()
    return (bbox.end.x - bbox.start.x + 1, bbox.end.y - bbox.start.y + 1)


def _affine_axes(gamma, beta):
    if gamma is not None and beta is not None:
        affine = "gamma_beta"
    elif gamma is not None:
        affine = "gamma_only"
    elif beta is not None:
        # Outside the TARGET universe (no beta-only cell); refused below via SUPPORTED.
        affine = "beta_only"
    else:
        affine = "no_affine"
    ref = gamma if gamma is not None else beta
    if ref is None:
        return affine, "none", "none"
    return affine, ref.dtype, ref.layout


def validate(input_tensor, num_groups, *, gamma=None, beta=None, eps=1e-5, in_place=False):
    # (1) Structural ValueErrors.
    shape = tuple(input_tensor.shape)
    if len(shape) != 4:
        raise ValueError(f"groupnorm_sc_N_1_HW_C_b: input rank must be 4 (N, 1, HW, C), got rank {len(shape)}")
    if shape[1] != 1:
        raise ValueError(f"groupnorm_sc_N_1_HW_C_b: input shape[1] must be 1, got shape {shape}")
    N, _, HW, C = shape
    if not isinstance(num_groups, int) or num_groups < 1 or C % num_groups != 0:
        raise ValueError(f"groupnorm_sc_N_1_HW_C_b: num_groups={num_groups} must be >= 1 and divide C={C}")
    for name, t in (("gamma", gamma), ("beta", beta)):
        if t is not None and tuple(t.shape) != (1, 1, 1, C):
            raise ValueError(f"groupnorm_sc_N_1_HW_C_b: {name} must have shape (1, 1, 1, {C}), got {tuple(t.shape)}")
    if not (eps > 0):
        raise ValueError(f"groupnorm_sc_N_1_HW_C_b: eps must be > 0, got {eps}")
    placement = _shard_placement(input_tensor)
    if in_place and placement is None:
        raise ValueError("groupnorm_sc_N_1_HW_C_b: in_place=True requires a BLOCK_SHARDED (L1) input")

    # (2) Axes.
    affine, affine_dtype, affine_layout = _affine_axes(gamma, beta)
    axes = {
        "dtype": input_tensor.dtype,
        "layout": input_tensor.layout,
        "affine": affine,
        "affine_dtype": affine_dtype,
        "affine_layout": affine_layout,
        "in_place": bool(in_place),
    }
    inputs = (shape, num_groups) if placement is None else (shape, num_groups, placement)
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger(inputs, axes)
    # memory_layout is read from the tensor itself (a non-block sharding is not a BLOCK_SHARDED cell).
    axes["memory_layout"] = input_tensor.memory_config().memory_layout
    if gamma is not None and beta is not None and (gamma.dtype != beta.dtype or gamma.layout != beta.layout):
        raise UnsupportedAxisValue(
            "groupnorm_sc_N_1_HW_C_b: gamma and beta must share dtype and layout "
            f"(gamma {gamma.dtype}/{gamma.layout}, beta {beta.dtype}/{beta.layout})"
        )

    # (3) SUPPORTED / EXCLUSIONS.
    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"groupnorm_sc_N_1_HW_C_b: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")
    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"groupnorm_sc_N_1_HW_C_b: unsupported combination (refinement candidate): {exc}")

    # Shape-class gate: a feasible column-slice count must exist (cross_column_group_combine regime).
    from .groupnorm_sc_N_1_HW_C_b_program_descriptor import plan_or_raise

    return plan_or_raise(input_tensor, num_groups, has_gamma=gamma is not None, has_beta=beta is not None)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def groupnorm_sc_N_1_HW_C_b(input_tensor, num_groups, *, gamma=None, beta=None, eps=1e-5, in_place=False):
    plan = validate(input_tensor, num_groups, gamma=gamma, beta=beta, eps=eps, in_place=in_place)

    from .groupnorm_sc_N_1_HW_C_b_program_descriptor import create_program_descriptor

    device = input_tensor.device()
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(input_tensor.shape)),
        input_tensor.dtype,
        ttnn.TILE_LAYOUT,
        device,
        ttnn.DRAM_MEMORY_CONFIG,
    )
    program_descriptor = create_program_descriptor(plan, input_tensor, output_tensor, gamma=gamma, beta=beta, eps=eps)
    io_tensors = [input_tensor]
    if gamma is not None:
        io_tensors.append(gamma)
    if beta is not None:
        io_tensors.append(beta)
    io_tensors.append(output_tensor)
    return ttnn.generic_op(io_tensors, program_descriptor)
