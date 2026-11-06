# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""rms_norm — RMSNorm along the last dim as ONE native generic_op dispatch.

    out[..., h, w] = x[..., h, w] * rsqrt(mean_w(x[..., h, :]^2) + eps) * gamma[w]

Registry model: INPUT_TAGGERS / SUPPORTED / EXCLUSIONS / validate() below, then the public entry
point. The blocking (row-groups x optional W split with a cross-Tensix-core combine) is selected in
rms_norm_program_descriptor.select_blocking(); see op_design.md -> Blocking Model.
"""

from __future__ import annotations

from typing import Optional

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .rms_norm_program_descriptor import create_program_descriptor


# ---------------------------------------------------------------------------
# Phase 0 compute-config default — the ONE place the default lives.
# ---------------------------------------------------------------------------


def default_compute_kernel_config() -> ttnn.ComputeConfigDescriptor:
    """Fresh Phase 0 config per call: HiFi4, fp32 DEST accumulation, exact SFPU math."""
    config = ttnn.ComputeConfigDescriptor()
    config.math_fidelity = ttnn.MathFidelity.HiFi4
    config.fp32_dest_acc_en = True
    config.math_approx_mode = False
    return config


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS (shape-derived axes only)
# ---------------------------------------------------------------------------


def tag_alignment(inputs, axes):
    """W (-1) checked first: "w_non_aligned" whenever W % 32 != 0, else H (-2)."""
    shape = inputs[0]
    if shape[-1] % 32 != 0:
        return "w_non_aligned"
    if len(shape) >= 2 and shape[-2] % 32 != 0:
        return "h_non_aligned"
    return "tile_aligned"


def tag_rank(inputs, axes):
    return int(len(inputs[0]))


INPUT_TAGGERS = {
    "alignment": tag_alignment,
    "rank": tag_rank,
}


# gamma_dtype / gamma_layout are tensor-property axes of the OPTIONAL weight. They are NOT registered in
# INPUT_TAGGERS: the harness treats every INPUT_TAGGERS key as shape-derived and stops iterating it from
# TARGET (eval/feature_matrix.py cartesian()), which would collapse the gamma_dtype/gamma_layout axes of the
# golden cartesian. validate() projects them with these two functions; both return the "none" sentinel
# when gamma is absent.
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
    "gamma_layout": [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT, "none"],
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
    # Precision convention: fp32 input requires fp32 DEST accumulation (natively rejected).
    {"dtype": ttnn.float32, "fp32_dest_acc_en": False},
    # bfloat8_b is block-quantized (16-datum shared exponents): no ROW_MAJOR form, and the reader's W-tail
    # zero fill / padded reduce is element-addressed — mirrors feature_spec INVALID so translated tests that
    # do not consult INVALID get a native refusal instead of an unsupported path.
    {"dtype": ttnn.bfloat8_b, "layout": ttnn.ROW_MAJOR_LAYOUT},
    {"dtype": ttnn.bfloat8_b, "alignment": "w_non_aligned"},
    {"dtype": ttnn.bfloat8_b, "alignment": "h_non_aligned"},
    {"gamma_dtype": ttnn.bfloat8_b, "gamma_layout": ttnn.ROW_MAJOR_LAYOUT},
    # ROW_MAJOR input sharded across the W axis with a TILE gamma: an RM shard's W slice starts at any 16 B
    # granule, not a tile edge, so a member's gamma slice is not a run of whole gamma tiles. Mirrors the
    # feature_spec INVALID entries (author-scoped) so translated tests get a native refusal.
    {
        "layout": ttnn.ROW_MAJOR_LAYOUT,
        "memory_layout": ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        "gamma_layout": ttnn.TILE_LAYOUT,
    },
    {
        "layout": ttnn.ROW_MAJOR_LAYOUT,
        "memory_layout": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        "gamma_layout": ttnn.TILE_LAYOUT,
    },
]


PROPERTIES = {
    "multi_core": {"value": True, "source": "declared"},
}


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def _gamma_width(gamma):
    return int(gamma.shape[-1])


def _gamma_volume(gamma):
    v = 1
    for d in gamma.shape:
        v *= int(d)
    return v


def validate(input_tensor, *, gamma=None, epsilon=1e-6, compute_kernel_config=None, memory_config=None):
    shape = list(input_tensor.shape)
    # Structural errors (ValueError) first.
    if len(shape) < 2:
        raise ValueError(f"rms_norm: input rank must be >= 2, got {len(shape)}")
    W = int(shape[-1])
    if gamma is not None:
        if _gamma_width(gamma) != W:
            raise ValueError(f"rms_norm: gamma last dim {_gamma_width(gamma)} != input last dim {W}")
        if _gamma_volume(gamma) != W:
            raise ValueError(f"rms_norm: gamma must hold exactly W={W} elements, shape {list(gamma.shape)}")
    if epsilon is None or not (float(epsilon) >= 0.0) or float(epsilon) == float("inf"):
        raise ValueError(f"rms_norm: epsilon must be finite and >= 0, got {epsilon}")

    config = compute_kernel_config if compute_kernel_config is not None else default_compute_kernel_config()
    axes = {
        "dtype": input_tensor.dtype,
        "fp32_dest_acc_en": bool(config.fp32_dest_acc_en),
        "layout": input_tensor.layout,
        "gamma_mode": "gamma" if gamma is not None else "no_gamma",
        "gamma_dtype": tag_gamma_dtype(gamma),
        "gamma_layout": tag_gamma_layout(gamma),
        "memory_layout": input_tensor.memory_config().memory_layout,
    }
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger((shape,), axes)

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"rms_norm: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")
    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"rms_norm: unsupported combination (refinement candidate): {exc}")

    in_mc = input_tensor.memory_config()
    if in_mc.is_sharded() and in_mc.buffer_type != ttnn.BufferType.L1:
        raise UnsupportedAxisValue(f"rms_norm: sharded input must be L1-resident, got {in_mc.buffer_type!r}")
    # Output placement: interleaved input -> interleaved output; sharded input -> the input's shard spec.
    if memory_config is not None:
        if in_mc.is_sharded():
            if memory_config != in_mc:
                raise UnsupportedAxisValue("rms_norm: a sharded output must use the input's shard spec")
        elif memory_config.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
            raise UnsupportedAxisValue(
                f"rms_norm: output memory_layout={memory_config.memory_layout!r} for an interleaved input "
                f"(the output follows the input's placement)"
            )
    if gamma is not None and gamma.memory_config().memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
        raise UnsupportedAxisValue("rms_norm: gamma must be interleaved")


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
    validate(
        input_tensor,
        gamma=gamma,
        epsilon=epsilon,
        compute_kernel_config=compute_kernel_config,
        memory_config=memory_config,
    )
    if compute_kernel_config is None:
        compute_kernel_config = default_compute_kernel_config()

    device = input_tensor.device()
    out_mem = memory_config if memory_config is not None else input_tensor.memory_config()
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(input_tensor.shape)),
        input_tensor.dtype,
        input_tensor.layout,
        device,
        out_mem,
    )
    descriptor = create_program_descriptor(
        input_tensor, gamma, output_tensor, epsilon=float(epsilon), compute_kernel_config=compute_kernel_config
    )
    io = [input_tensor] + ([gamma] if gamma is not None else []) + [output_tensor]
    return ttnn.generic_op(io, descriptor)
