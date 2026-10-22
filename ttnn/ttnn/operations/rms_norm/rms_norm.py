# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""rms_norm — RMSNorm(x) = x / sqrt(mean(x^2, dim=-1) + eps) * gamma.

One native dispatch (`ttnn.generic_op`) per call (op_design.md -> Regimes):
  R1 — row-parallel, row-resident: tile-rows are split across the full Tensix-core grid, each core
       keeps a block of `block_rows` tile-rows resident in L1 for both the reduction pass and the
       scaling pass.
  R1-HS — HEIGHT_SHARDED: every Tensix core holds full-W rows of its shard (zero-copy shard CBs), so the
       reduction stays local — the sharded path with one W-slice per core (no combine).
  R3 — W-split + cross-core Σx² combine: WIDTH/BLOCK-sharded inputs (zero-copy shard CBs), and
       interleaved TILE inputs whose rows are too wide for R1 or too few to fill the grid.

Registry model: INPUT_TAGGERS / SUPPORTED / EXCLUSIONS / validate() below drive both the
runtime gate and the golden-test xfail decisions.
"""

from __future__ import annotations

from typing import Optional

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .rms_norm_program_descriptor import create_program_descriptor


# ---------------------------------------------------------------------------
# Default compute config — the ONE place the Phase 0 default is defined.
# ---------------------------------------------------------------------------


def default_compute_kernel_config() -> ttnn.ComputeConfigDescriptor:
    """Fresh Phase 0 default compute config (maxed-out precision corner)."""
    return ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        math_approx_mode=False,
    )


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS
# ---------------------------------------------------------------------------

TILE_DIM = 32


def tag_alignment(inputs, axes):
    """tile_aligned (H%32==0 and W%32==0) / w_non_aligned (W%32!=0) / h_non_aligned (W aligned, H not)."""
    shape = inputs[0]
    H, W = shape[-2], shape[-1]
    if W % TILE_DIM != 0:
        return "w_non_aligned"
    if H % TILE_DIM != 0:
        return "h_non_aligned"
    return "tile_aligned"


def tag_rank(inputs, axes):
    return int(len(inputs[0]))


INPUT_TAGGERS = {
    "alignment": tag_alignment,
    "rank": tag_rank,
}


# gamma_dtype / gamma_layout are tensor-property axes of the (optional) gamma tensor, not
# shape facets: the golden cartesian iterates them from TARGET, and registering them in
# INPUT_TAGGERS would remove them from that iteration (eval/feature_matrix.py:112-113).
# They are classified here from the gamma tensor itself, with the "none" sentinel when absent.
def tag_gamma_dtype(gamma):
    return gamma.dtype if gamma is not None else "none"


def tag_gamma_layout(gamma):
    return gamma.layout if gamma is not None else "none"


# ---------------------------------------------------------------------------
# 2. SUPPORTED
# ---------------------------------------------------------------------------

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
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
    ],
}


# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------

EXCLUSIONS = [
    # Natively rejected precision combination (precision_convention.md).
    {"dtype": ttnn.float32, "fp32_dest_acc_en": False},
    # Structural capability gap: zeroing the W-pad columns of the last tile is per-element work in
    # the reader; inside a Bfp8_b tile it is exponent-section work (16 datums share one exponent),
    # which the reader does not do. Pad *tiles* (all-zero bytes) are a valid Bfp8_b zero and stay.
    {"dtype": ttnn.bfloat8_b, "alignment": "w_non_aligned"},
]


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def _hard_validate(input_tensor, gamma):
    """Structural errors (ValueError), raised before any support check."""
    shape = list(input_tensor.shape)
    if len(shape) < 2:
        raise ValueError(f"rms_norm: input must have rank >= 2, got rank {len(shape)}")
    if gamma is not None:
        g_shape = list(gamma.shape)
        W = shape[-1]
        if g_shape[-1] != W:
            raise ValueError(f"rms_norm: gamma last dim {g_shape[-1]} != input last dim {W}")
        g_volume = 1
        for d in g_shape:
            g_volume *= d
        if g_volume != W:
            raise ValueError(f"rms_norm: gamma must hold exactly W={W} elements, got shape {g_shape}")


def _classify(input_tensor, gamma, compute_kernel_config, memory_config):
    has_gamma = gamma is not None
    axes = {
        "dtype": input_tensor.dtype,
        "layout": input_tensor.layout,
        "fp32_dest_acc_en": bool(compute_kernel_config.fp32_dest_acc_en),
        "gamma_mode": "gamma" if has_gamma else "no_gamma",
        "gamma_dtype": tag_gamma_dtype(gamma),
        "gamma_layout": tag_gamma_layout(gamma),
        "memory_layout": input_tensor.memory_config().memory_layout,
    }
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger((list(input_tensor.shape),), axes)
    return axes


def validate(input_tensor, *, gamma=None, compute_kernel_config=None, memory_config=None):
    cfg = compute_kernel_config if compute_kernel_config is not None else default_compute_kernel_config()
    _hard_validate(input_tensor, gamma)
    axes = _classify(input_tensor, gamma, cfg, memory_config)

    # Output placement is an axis too.
    if memory_config is not None and memory_config.memory_layout not in SUPPORTED["memory_layout"]:
        raise UnsupportedAxisValue(
            f"rms_norm: output memory_layout={memory_config.memory_layout!r} not in SUPPORTED "
            f"{SUPPORTED['memory_layout']}"
        )

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"rms_norm: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")

    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"rms_norm: unsupported combination (refinement candidate): {exc}")

    # The sharded regimes compute in place of the input's shard geometry: the output must carry the same
    # placement (the common norm contract); an interleaved input writes an interleaved output.
    in_mc = input_tensor.memory_config()
    if memory_config is not None and not _same_placement(memory_config, in_mc):
        raise NotImplementedError(f"rms_norm: output placement {memory_config} must match the input placement {in_mc}")
    return cfg


def _same_placement(out_mc, in_mc) -> bool:
    if out_mc.memory_layout != in_mc.memory_layout:
        return False
    if in_mc.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED:
        return True
    a, b = out_mc.shard_spec, in_mc.shard_spec
    return (
        out_mc.buffer_type == in_mc.buffer_type
        and a is not None
        and b is not None
        and list(a.shape) == list(b.shape)
        and a.grid == b.grid
        and a.orientation == b.orientation
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def rms_norm(
    input_tensor: ttnn.Tensor,
    *,
    gamma: Optional[ttnn.Tensor] = None,
    epsilon: float = 1e-6,
    compute_kernel_config: Optional[ttnn.ComputeConfigDescriptor] = None,
    memory_config: Optional[ttnn.MemoryConfig] = None,
) -> ttnn.Tensor:
    cfg = validate(input_tensor, gamma=gamma, compute_kernel_config=compute_kernel_config, memory_config=memory_config)

    device = input_tensor.device()
    output_memory_config = memory_config if memory_config is not None else input_tensor.memory_config()

    # allocate_tensor_on_device takes positional args only.
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(input_tensor.shape)),
        input_tensor.dtype,
        input_tensor.layout,
        device,
        output_memory_config,
    )

    program_descriptor = create_program_descriptor(
        input_tensor, output_tensor, gamma=gamma, epsilon=epsilon, compute_kernel_config=cfg
    )

    io_tensors = [input_tensor] + ([gamma] if gamma is not None else []) + [output_tensor]
    # Output tensor must be last.
    return ttnn.generic_op(io_tensors, program_descriptor)
