# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""rms_norm — root-mean-square layer normalization over the last dim.

    RMSNorm(x) = x / sqrt(mean(x^2, dim=-1, keepdim=True) + epsilon) * gamma

Registry-model op file: four declarations (INPUT_TAGGERS, SUPPORTED, EXCLUSIONS,
validate) + the public entry point. `gamma` is an optional [1,1,1,W] ROW_MAJOR
scale; output shape/dtype/layout match the input. Both TILE and ROW_MAJOR inputs
are handled natively (no host-side layout conversion).
"""

from __future__ import annotations

from typing import Optional

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .rms_norm_program_descriptor import create_program_descriptor


# ---------------------------------------------------------------------------
# Phase-0 compute-kernel-config default (single exported factory).
# ---------------------------------------------------------------------------
def default_compute_kernel_config() -> ttnn.ComputeConfigDescriptor:
    """The phase-0 maxed-out precision corner. `None` resolves through this, and
    the golden axis-tagger reads the same factory — don't inline the default."""
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi4
    cfg.fp32_dest_acc_en = True
    cfg.math_approx_mode = False
    return cfg


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS — project the input shape onto categorical axes.
# ---------------------------------------------------------------------------
def tag_alignment(inputs, axes):
    """Three-value split (matches feature_spec):
    tile_aligned / w_non_aligned / h_non_aligned."""
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
# 2. SUPPORTED — per-axis accepted values (phase-0: narrow but axis-complete).
# ---------------------------------------------------------------------------
SUPPORTED = {
    # R2 added bfloat8_b (TILE input only — bf8b + ROW_MAJOR is structurally
    # INVALID, skipped by the harness). bf8b's low-precision reduce rides the
    # R1-fixed reduce datapath via bf16 intermediate CBs (see program descriptor).
    "dtype": [ttnn.float32, ttnn.bfloat16, ttnn.bfloat8_b],
    # Two-axis precision model (dtype x fp32_dest_acc_en). R2 dropped the
    # {bf16, False} EXCLUSION (bf16 without fp32 dest-acc is a supported target);
    # {float32, False} stays EXCLUDED (permanent — fp32 requires fp32 accumulation).
    "fp32_dest_acc_en": [True, False],
    "layout": [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
    "alignment": ["tile_aligned", "w_non_aligned", "h_non_aligned"],
    "rank": [2, 3, 4],
    "gamma_mode": ["gamma", "no_gamma"],
    # "none" sentinel == "no weight" and is ALWAYS legal. R2 added bfloat8_b
    # (block-quantized gamma; TILE-only since bf8b + ROW_MAJOR is INVALID).
    "gamma_dtype": [ttnn.float32, ttnn.bfloat16, ttnn.bfloat8_b, "none"],
    # R2 added TILE_LAYOUT: gamma read as tiles directly (no RM->tilize) — a
    # second gamma reader leg alongside the RM one, required for bf8b gamma and
    # independently useful for mixed-precision (bf16 acts + fp32/bf8b TILE weights).
    "gamma_layout": [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT, "none"],
    # R4 added HEIGHT_SHARDED: the Row-axis knob-turn. Each core owns a contiguous
    # span of tile-rows (pinned by the shard spec) and computes their RMS locally —
    # the reduction stays LOCAL per core, no cross-core communication.
    # R5 added WIDTH_SHARDED + BLOCK_SHARDED: the dependent-axis scheme-change. The
    # hidden W is split across cores, so each core computes a partial Σx² over its
    # W-slice; the partials are combined cross-core (reduce-root gather + mcast
    # broadcast-back of 1/rms). See rms_norm_program_descriptor.py + the sharded
    # reader/compute kernels.
    "memory_layout": [
        ttnn.TensorMemoryLayout.INTERLEAVED,
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
    ],
}


# ---------------------------------------------------------------------------
# 3. EXCLUSIONS — cells inside SUPPORTED refused for now (refinement candidates).
# ---------------------------------------------------------------------------
EXCLUSIONS = [
    # float32 + False: legal config, permanently refused (fp32 requires fp32
    # accumulation — see references/precision_convention.md). NOT a refinement.
    {"dtype": ttnn.float32, "fp32_dest_acc_en": False},
    # NOTE: R2 anticipated a {bfloat8_b, non-tile-aligned} exclusion, but on-device
    # verification (golden check_output, PCC>=0.99 & rel-RMS<=0.10) showed every
    # bf8b non-tile-aligned shape passes: bf8b input is TILE-only, so ttnn's
    # zero-padding keeps the block-float shared exponent clean, the masked
    # partial-W reduce zeros the W-tail, and H-padding rows reduce to 0 and are
    # dropped by the writer. So bf8b is FULLY supported — no exclusion added.
    # R5a landed ROW_MAJOR + WIDTH/BLOCK_SHARDED cross-core reduction: each core reads
    # its OWN resident [Hs, Ws] shard directly from local L1, zero-pads the sub-tile W
    # (and H) tail to whole tiles, tilizes, and the SAME R5 reduce-root gather + mcast
    # broadcast combine runs unchanged. One structural gap remains, carved out below:
    #
    # (5b landed) RM + WIDTH_SHARDED + non-tile-aligned W. auto_shard_config splits a
    #     non-aligned W into a RAGGED grid (ncores != nx*ny), so the WIDTH reduction
    #     group is not a rectangle the mcast broadcast can address. R5b replaces the
    #     mcast broadcast-back with a UNICAST broadcast (root -> each group member +
    #     a per-member ready flag, mirroring the already-unicast gather leg) for ragged
    #     WIDTH groups only; rectangular WIDTH/BLOCK groups keep the mcast fast path.
    #     So {RM, WIDTH, w_non_aligned} is now SUPPORTED (no exclusion).
    #
    # (5c landed) RM cross-core + TILE gamma (fp32/bf16). Each core owns a sub-tile W-slice
    #     at a sub-tile global column offset (w_col_start = i*Ws), so a TILE-stored gamma
    #     can't be read as whole tiles aligned to the core's LOCAL column 0. The reader now
    #     extracts the containing global gamma tile(s)' ROW-0 sub-columns (face-aware L1 byte
    #     copy) into cb_gamma_rm and reuses the RM-gamma compute tilize leg — so fp32/bf16
    #     TILE gamma is SUPPORTED (no exclusion). See rms_norm_sharded_reader.cpp
    #     (GAMMA_TILE_EXTRACT) + _build_cross_core_descriptor.
    #
    # (5d landed) RM cross-core + TILE gamma, gamma_dtype=bf8b. A bf8b tile is block-float
    #     (16 elements share an 8-bit exponent; per-element bytes are meaningless without it),
    #     so the row-0 sub-column extraction is not a byte copy — the reader now DEQUANTS each
    #     row-0 datum (block-float decode, GAMMA_TILE_EXTRACT==2 / bfp8b_datum_to_f32_bits)
    #     into the float cb_gamma_rm and reuses the SAME RM-gamma compute tilize leg. The
    #     dequant matches the hardware unpacker (bf8b->float lossless == the R2 INTERLEAVED
    #     bf8b-gamma FPU-unpack path), so bf8b TILE gamma is now SUPPORTED (no exclusion).
]


PROPERTIES = {
    # R4: row work is distributed across the grid (INTERLEAVED via split_work_to_cores;
    # HEIGHT_SHARDED via the shard spec) — embarrassingly parallel, no cross-core comms.
    "multi_core": {"value": True, "source": "declared"},
    "bounded_cb": {"value": True, "source": "declared"},
    "math_fidelity": {"value": ["LoFi", "HiFi2", "HiFi3", "HiFi4"], "source": "declared"},
}


# ---------------------------------------------------------------------------
# 4. validate() — runtime gate. Structural errors first (ValueError), then the
#    registry SUPPORTED / EXCLUSIONS checks (typed NotImplementedError).
# ---------------------------------------------------------------------------
def validate(input_tensor, *, gamma=None, compute_kernel_config=None):
    # Structural checks (design Validation table).
    shape = tuple(input_tensor.shape)
    if len(shape) < 2:
        raise ValueError(f"rms_norm: input must have rank >= 2 (got rank {len(shape)})")
    if gamma is not None and int(gamma.shape[-1]) != int(shape[-1]):
        raise ValueError(f"rms_norm: gamma last dim {int(gamma.shape[-1])} != input last dim {int(shape[-1])}")

    cfg = compute_kernel_config if compute_kernel_config is not None else default_compute_kernel_config()

    # Build the axes dict the same way the golden harness does: tensor
    # properties + kwargs + every tagger.
    axes = {
        "dtype": input_tensor.dtype,
        "fp32_dest_acc_en": bool(cfg.fp32_dest_acc_en),
        "layout": input_tensor.layout,
        "gamma_mode": "gamma" if gamma is not None else "no_gamma",
        "gamma_dtype": gamma.dtype if gamma is not None else "none",
        "gamma_layout": gamma.layout if gamma is not None else "none",
        "memory_layout": input_tensor.memory_config().memory_layout,
    }
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger((shape,), axes)

    # 1. SUPPORTED — per-axis.
    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"rms_norm: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")

    # 2. EXCLUSIONS — cell-level inside SUPPORTED.
    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"rms_norm: unsupported combination (refinement candidate): {exc}")


# ---------------------------------------------------------------------------
# Public entry point.
# ---------------------------------------------------------------------------
def rms_norm(
    input_tensor: ttnn.Tensor,
    *,
    gamma: Optional[ttnn.Tensor] = None,
    epsilon: float = 1e-6,
    compute_kernel_config: ttnn.ComputeConfigDescriptor = None,
    memory_config: ttnn.MemoryConfig = None,
) -> ttnn.Tensor:
    validate(input_tensor, gamma=gamma, compute_kernel_config=compute_kernel_config)

    cfg = compute_kernel_config if compute_kernel_config is not None else default_compute_kernel_config()
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
        compute_kernel_config=cfg,
    )

    tensors = [input_tensor, output_tensor] if gamma is None else [input_tensor, gamma, output_tensor]
    return ttnn.generic_op(tensors, program_descriptor)
