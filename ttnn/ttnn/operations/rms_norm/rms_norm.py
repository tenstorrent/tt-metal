# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""rms_norm — root-mean-square normalization over the last dimension.

    out[..., r, c] = x[..., r, c] * rsqrt(mean(x[..., r, :]^2) + epsilon) * gamma[c]

Registry-model op: the four declarations (INPUT_TAGGERS / SUPPORTED /
EXCLUSIONS / validate) plus the public entry point.  The kernel schedule and
its blocking model live in `op_design.md`; the program descriptor that realizes
them is `rms_norm_program_descriptor.py`.
"""

from __future__ import annotations

from typing import Optional

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from ttnn.operations.rms_norm.rms_norm_program_descriptor import create_program_descriptor

TILE_DIM = 32


# ---------------------------------------------------------------------------
# Phase 0 compute-kernel-config default — the ONLY definition
# ---------------------------------------------------------------------------
#
# `None` resolves through this factory, and the golden axis tagger reads the
# same factory, so the Phase 0 precision corner is spelled exactly once.
# It is a FACTORY (fresh descriptor per call), never a shared constant.


def default_compute_kernel_config() -> "ttnn.ComputeConfigDescriptor":
    """Phase 0 default: maxed-out precision corner."""
    return ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        math_approx_mode=False,
    )


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS
# ---------------------------------------------------------------------------


def tag_alignment(inputs, axes):
    """Three-value split over the last two dims (see feature_spec.py).

    tile_aligned  — H and W both multiples of 32
    w_non_aligned — W not a multiple of 32 (H may or may not be)
    h_non_aligned — W aligned, H not aligned
    """
    shape = inputs[0]
    w = shape[-1]
    h = shape[-2] if len(shape) >= 2 else TILE_DIM
    if w % TILE_DIM != 0:
        return "w_non_aligned"
    if h % TILE_DIM != 0:
        return "h_non_aligned"
    return "tile_aligned"


def tag_rank(inputs, axes):
    return int(len(inputs[0]))


INPUT_TAGGERS = {
    "alignment": tag_alignment,
    "rank": tag_rank,
}


# ---------------------------------------------------------------------------
# 2. SUPPORTED  (Phase 0 — narrow on placement only)
# ---------------------------------------------------------------------------
#
# `rank` is checked first so a rank-1 tensor is refused before any axis that
# would need a second dimension to be meaningful.

SUPPORTED = {
    "rank": [2, 3, 4],
    "dtype": [ttnn.bfloat16, ttnn.float32, ttnn.bfloat8_b],
    # Both accumulation widths are supported.  The stat CBs stay float32
    # regardless (see rms_norm_program_descriptor.py) — the axis only selects
    # the DEST accumulation width, never the L1 statistic format.
    # {float32, fp32_dest_acc_en=False} is refused below in EXCLUSIONS.
    "fp32_dest_acc_en": [True, False],
    "layout": [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
    "alignment": ["tile_aligned", "w_non_aligned", "h_non_aligned"],
    "gamma_mode": ["gamma", "no_gamma"],
    # "none" is the absent-gamma sentinel and is ALWAYS legal.
    "gamma_dtype": [ttnn.bfloat16, ttnn.float32, ttnn.bfloat8_b, "none"],
    "gamma_layout": [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT, "none"],
    # All four placements are the SAME logical scheme (op_design.md §Lamps,
    # "Physical shard placement"): HEIGHT cuts the independent row axis
    # (num_hidden_slices == 1), WIDTH cuts the dependent hidden axis (the Phase 0
    # gather + broadcast combine), BLOCK cuts both.  A shard is consumed
    # NATIVELY — the CB is bound to the caller's resident L1 buffer — never
    # re-read through a TensorAccessor.
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
# `fp32_dest_acc_en=False` is now inside SUPPORTED, so the one cell the design
# calls out as a PERMANENT refusal is finally expressible as an EXCLUSION:
# a caller who hands us float32 activations is asking for float32 arithmetic,
# and silently accumulating them at reduced DEST width would be a lie about the
# precision they paid for.  This is not a refinement candidate.

EXCLUSIONS = [
    {"dtype": ttnn.float32, "fp32_dest_acc_en": False},
]


# ---------------------------------------------------------------------------
# 3b. PROPERTIES
# ---------------------------------------------------------------------------

PROPERTIES = {
    "multi_core": {"value": True, "source": "declared"},
    "bounded_cb": {"value": True, "source": "declared"},
    "math_fidelity": {"value": ["LoFi", "HiFi2", "HiFi3", "HiFi4"], "source": "declared"},
}


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def _structural_checks(input_tensor, gamma):
    """Shape contracts that are not registry axes (raise ValueError)."""
    shape = list(input_tensor.shape)
    if len(shape) < 2:
        raise ValueError(f"rms_norm: input_tensor must have rank >= 2 (at least 2 dimensions); got rank {len(shape)}")
    if gamma is not None:
        gamma_shape = list(gamma.shape)
        if gamma_shape[-1] != shape[-1]:
            raise ValueError(
                f"rms_norm: gamma last dim (width) {gamma_shape[-1]} must match " f"input_tensor last dim {shape[-1]}"
            )
        if any(d != 1 for d in gamma_shape[:-1]):
            raise ValueError(f"rms_norm: gamma must have shape (1, ..., 1, W); got {gamma_shape}")


def _output_placement_checks(input_tensor, memory_config):
    """The output inherits the input's placement; a mismatch is a caller error.

    rms_norm is elementwise in placement terms — every core writes exactly the
    block it read — so the output's shard geometry must equal the input's.  A
    different geometry would be a re-layout, which is `ttnn.to_memory_config`'s
    job, not this op's.
    """
    if memory_config is None:
        return
    src = input_tensor.memory_config()
    if memory_config.memory_layout != src.memory_layout:
        raise ValueError(
            f"rms_norm: output memory_layout {memory_config.memory_layout} must match the "
            f"input's {src.memory_layout} (the op does not re-place its result)"
        )
    if memory_config.shard_spec is not None or src.shard_spec is not None:
        if memory_config.shard_spec is None or src.shard_spec is None:
            raise ValueError("rms_norm: output shard_spec must match the input's")
        if list(memory_config.shard_spec.shape) != list(src.shard_spec.shape) or (
            memory_config.shard_spec.grid != src.shard_spec.grid
        ):
            raise ValueError(
                f"rms_norm: output shard spec {list(memory_config.shard_spec.shape)} must match the "
                f"input's {list(src.shard_spec.shape)} on the same grid"
            )


def validate(input_tensor, *, gamma=None, epsilon=1e-6, compute_kernel_config=None, memory_config=None):
    """Runtime support gate. Called as the entry point's first statement."""
    _structural_checks(input_tensor, gamma)
    _output_placement_checks(input_tensor, memory_config)

    cfg = compute_kernel_config if compute_kernel_config is not None else default_compute_kernel_config()
    has_gamma = gamma is not None

    axes = {
        "dtype": input_tensor.dtype,
        "fp32_dest_acc_en": bool(getattr(cfg, "fp32_dest_acc_en", True)),
        "layout": input_tensor.layout,
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
    input_tensor: "ttnn.Tensor",
    *,
    gamma: Optional["ttnn.Tensor"] = None,
    epsilon: float = 1e-6,
    compute_kernel_config: "ttnn.ComputeConfigDescriptor" = None,
    memory_config: Optional["ttnn.MemoryConfig"] = None,
) -> "ttnn.Tensor":
    """RMSNorm over the last dimension.

    No host-side layout / shape workaround: both TILE_LAYOUT and
    ROW_MAJOR_LAYOUT inputs (and non-tile-aligned H/W) are handled natively by
    the kernels, at INTERLEAVED or any of the three sharded placements.

    `memory_config` selects the output placement; it defaults to the input's, so
    a sharded input yields a matching sharded output (the norm convention).
    """
    validate(
        input_tensor,
        gamma=gamma,
        epsilon=epsilon,
        compute_kernel_config=compute_kernel_config,
        memory_config=memory_config,
    )

    if compute_kernel_config is None:
        compute_kernel_config = default_compute_kernel_config()

    output_tensor = ttnn.allocate_tensor_on_device(
        input_tensor.shape,
        input_tensor.dtype,
        input_tensor.layout,
        input_tensor.device(),
        memory_config if memory_config is not None else input_tensor.memory_config(),
    )

    program_descriptor = create_program_descriptor(
        input_tensor,
        output_tensor,
        gamma=gamma,
        epsilon=epsilon,
        compute_kernel_config=compute_kernel_config,
    )

    io_tensors = [input_tensor]
    if gamma is not None:
        io_tensors.append(gamma)
    io_tensors.append(output_tensor)

    return ttnn.generic_op(io_tensors, program_descriptor)
