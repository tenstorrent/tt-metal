# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""RMSNorm over the last dimension (registry model).

    out[..., h, w] = x[..., h, w] * rsqrt(mean(x[..., h, :]**2) + eps) * gamma[w]

Four registry declarations (INPUT_TAGGERS / SUPPORTED / EXCLUSIONS / validate)
plus the public entry point, per `eval/op_template.py`.  INVALID lives in
`eval/golden_tests/rms_norm/feature_spec.py`, never here.

The kernel is a generic-op (ProgramDescriptor) implementation; see
`rms_norm_program_descriptor.py` for the blocking model and
`op_design.md` for the design it realizes.
"""

from __future__ import annotations

from typing import Any, Optional

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from ttnn.operations.rms_norm.rms_norm_program_descriptor import create_program_descriptor


# ---------------------------------------------------------------------------
# Compute-kernel-config default (single source of truth)
# ---------------------------------------------------------------------------
#
# Phase 0 is the maxed-out precision corner. The golden axis tagger
# (eval/golden_tests/rms_norm/axes.py) reads THIS factory, so the default must
# never be inlined anywhere else. A fresh descriptor per call — never a shared
# mutable constant.


def default_compute_kernel_config() -> "ttnn.ComputeConfigDescriptor":
    return ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        math_approx_mode=False,
    )


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS
# ---------------------------------------------------------------------------
#
# Shape-derived categorical axes. Both taggers read only inputs[0] (the
# activation shape); gamma_dtype / gamma_layout are NOT taggers — the harness
# reads them straight off the gamma tensor (axes.py), exactly like dtype/layout.


def tag_alignment(inputs, axes):
    """Three-value alignment split. W-not-divisible-by-32 wins over H."""
    shape = inputs[0]
    if shape[-1] % 32 != 0:
        return "w_non_aligned"
    if shape[-2] % 32 != 0:
        return "h_non_aligned"
    return "tile_aligned"


def tag_rank(inputs, axes):
    return int(len(inputs[0]))


INPUT_TAGGERS = {
    "alignment": tag_alignment,
    "rank": tag_rank,
}


# ---------------------------------------------------------------------------
# 2. SUPPORTED
# ---------------------------------------------------------------------------
#
# One entry per axis the golden feature_spec TARGET enumerates, so an
# out-of-rectangle cell xfails cleanly instead of being over-claimed.
#
# "none" on gamma_dtype / gamma_layout is the absent-weight sentinel and is
# ALWAYS legal (see the prompt's "Optional weight (gamma) axes" contract).

SUPPORTED = {
    "dtype": [ttnn.float32, ttnn.bfloat16, ttnn.bfloat8_b],
    # Refinement 1: the full precision surface. `fp32_dest_acc_en` is read off
    # the caller's compute_kernel_config (default True — see
    # default_compute_kernel_config, which is NOT changed by this refinement).
    "fp32_dest_acc_en": [True, False],
    "layout": [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
    "alignment": ["tile_aligned", "w_non_aligned", "h_non_aligned"],
    "rank": [2, 3, 4],
    "gamma_mode": ["gamma", "no_gamma"],
    "gamma_dtype": [ttnn.float32, ttnn.bfloat16, ttnn.bfloat8_b, "none"],
    "gamma_layout": [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT, "none"],
    # Refinement 2: the cross-core W-split makes the two W-splitting placements
    # native — each core's shard is consumed in place from its own L1 (a
    # zero-copy CB, no NoC read) and the dependent reduce is combined across the
    # shard's cores. HEIGHT_SHARDED is Refinement 4.
    "memory_layout": [
        ttnn.TensorMemoryLayout.INTERLEAVED,
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
    ],
}


# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------
#
# float32 + fp32_dest_acc_en=False is refused permanently (an fp32 input must
# not be silently accumulated through bfloat16 DEST) — see
# references/precision_convention.md. It is an op-side refusal, never an
# INVALID cell. Kept explicit so it stays load-bearing the moment
# fp32_dest_acc_en=False joins SUPPORTED for bfloat16 (Lamp L4).

EXCLUSIONS = [
    {"dtype": ttnn.float32, "fp32_dest_acc_en": False},
    # A ROW_MAJOR shard is not a tile block: eval.sharding's RM granule is
    # (1 row, L1_align/elem_bytes columns), so a shard is e.g. [1, 128] or
    # [64, 8] — a single stick, or 8 of the 32 columns of a tile. The kernels
    # tilize a 32-stick x 32-column block in place, which a core physically does
    # not hold under those shards; forming one would need a cross-core stick
    # gather, a different scheme from this refinement's W-split. Structural gap,
    # refused rather than left failing.
    {"layout": ttnn.ROW_MAJOR_LAYOUT, "memory_layout": ttnn.TensorMemoryLayout.WIDTH_SHARDED},
    {"layout": ttnn.ROW_MAJOR_LAYOUT, "memory_layout": ttnn.TensorMemoryLayout.BLOCK_SHARDED},
]


# ---------------------------------------------------------------------------
# 3b. PROPERTIES
# ---------------------------------------------------------------------------

PROPERTIES = {
    # The independent tile-row axis is split over the full compute grid from
    # phase 1 (ttnn.split_work_to_cores over device.compute_with_storage_grid_size()).
    "multi_core": {"value": True, "source": "declared"},
    # Every CB page count is a function of the block knobs (HT_BLOCK / WT_CHUNK
    # / the buffer depths), never of a whole-op dimension. The two Wt-sized CBs
    # are predicate-guarded residents with a streaming fallback, and the host
    # asserts the final per-core CB total against L1_CB_BUDGET_BYTES.
    "bounded_cb": {"value": True, "source": "declared"},
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
    """Runtime gate. Argument errors first, then SUPPORTED, then EXCLUSIONS."""

    # --- argument errors (ValueError / RuntimeError, message substrings are
    # matched by the acceptance test) ---
    if len(input_tensor.shape) < 2:
        raise ValueError(f"rms_norm: input_tensor must have rank >= 2, got rank {len(input_tensor.shape)}")
    if gamma is not None and int(gamma.shape[-1]) != int(input_tensor.shape[-1]):
        raise ValueError(
            f"rms_norm: gamma last dim {int(gamma.shape[-1])} must match "
            f"input_tensor last dim {int(input_tensor.shape[-1])}"
        )
    if epsilon <= 0:
        raise ValueError(f"rms_norm: epsilon must be > 0, got {epsilon}")

    # --- support refusals that are not axis values ---
    if program_config is not None:
        raise UnsupportedAxisValue("rms_norm: program_config is not supported yet")
    if memory_config is not None and memory_config.memory_layout not in SUPPORTED["memory_layout"]:
        raise UnsupportedAxisValue(
            f"rms_norm: output memory_layout={memory_config.memory_layout!r} "
            f"not in SUPPORTED {SUPPORTED['memory_layout']}"
        )

    cfg = compute_kernel_config if compute_kernel_config is not None else default_compute_kernel_config()

    axes = {
        "dtype": input_tensor.dtype,
        "layout": input_tensor.layout,
        "fp32_dest_acc_en": bool(getattr(cfg, "fp32_dest_acc_en", True)),
        "gamma_mode": "gamma" if gamma is not None else "no_gamma",
        "gamma_dtype": gamma.dtype if gamma is not None else "none",
        "gamma_layout": gamma.layout if gamma is not None else "none",
        "memory_layout": input_tensor.memory_config().memory_layout,
    }
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger((list(input_tensor.shape),), axes)

    # 1. SUPPORTED — per axis
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
    compute_kernel_config: Optional["ttnn.ComputeConfigDescriptor"] = None,
    memory_config: Optional["ttnn.MemoryConfig"] = None,
    program_config: Optional[Any] = None,
) -> "ttnn.Tensor":
    """RMSNorm over the last dimension.

    No host-side layout/padding workarounds: TILE and ROW_MAJOR inputs, and
    non-tile-aligned H and/or W, are all handled natively by the kernels.
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
    out_memory_config = memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG

    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(input_tensor.shape)),
        input_tensor.dtype,
        input_tensor.layout,
        device,
        out_memory_config,
    )

    program_descriptor = create_program_descriptor(
        input_tensor,
        output_tensor,
        gamma=gamma,
        epsilon=epsilon,
        compute_kernel_config=compute_kernel_config,
        device=device,
    )

    io_tensors = [input_tensor]
    if gamma is not None:
        io_tensors.append(gamma)
    io_tensors.append(output_tensor)

    return ttnn.generic_op(io_tensors, program_descriptor)
