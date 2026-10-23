# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""rms_norm — RMSNorm over the last dimension, one native generic_op dispatch.

    out[..., w] = x[..., w] * rsqrt(mean(x[..., :]^2) + epsilon) * (gamma[w] if gamma else 1)

Registry model (eval/op_template.py): INPUT_TAGGERS, SUPPORTED, EXCLUSIONS, validate().
The device program (blocking plan, CBs, kernels) lives in rms_norm_program_descriptor.py.
"""

from __future__ import annotations

import math
from typing import Optional

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .rms_norm_program_descriptor import create_program_descriptor


# ---------------------------------------------------------------------------
# Default compute config — the ONE definition of what `compute_kernel_config=None` means.
# ---------------------------------------------------------------------------


def default_compute_kernel_config():
    return ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        math_approx_mode=False,
    )


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS
# ---------------------------------------------------------------------------


def tag_alignment(inputs, axes):
    """W checked first (masked-reduce path), then H (row-padding path)."""
    shape = inputs[0]
    if shape[-1] % 32 != 0:
        return "w_non_aligned"
    if shape[-2] % 32 != 0:
        return "h_non_aligned"
    return "tile_aligned"


def tag_rank(inputs, axes):
    return int(len(inputs[0]))


def tag_gamma_dtype(inputs, axes):
    if axes.get("gamma_mode") == "no_gamma":
        return "none"
    return axes.get("gamma_dtype", "none")


def tag_gamma_layout(inputs, axes):
    if axes.get("gamma_mode") == "no_gamma":
        return "none"
    return axes.get("gamma_layout", "none")


# Only the SHAPE-derived axes are registered. gamma_dtype / gamma_layout are finite TARGET axes: the golden
# cartesian (eval/feature_matrix.cartesian) does NOT iterate an axis that has a registered tagger ("the tagger
# wins"), and a tagger only sees the input shape — so registering tag_gamma_* collapsed every cartesian gamma cell
# to gamma_dtype="none", which INVALID {gamma_mode: gamma, gamma_dtype: none} then skipped (1680 cells, the whole
# gamma path untested by the cartesian suite). The two functions keep the "none"-sentinel contract and are used by
# _axes_for() to classify a real call; the golden runtime classifier (eval/golden_tests/rms_norm/axes.py) pins the
# same sentinel itself.
INPUT_TAGGERS = {
    "alignment": tag_alignment,
    "rank": tag_rank,
}

GAMMA_TAGGERS = {
    "gamma_dtype": tag_gamma_dtype,
    "gamma_layout": tag_gamma_layout,
}


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
    "memory_layout": [ttnn.TensorMemoryLayout.INTERLEAVED],
}


# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------

EXCLUSIONS = [
    # Precision convention: a maxed-precision input with 16-bit DEST accumulation is refused natively.
    {"dtype": ttnn.float32, "fp32_dest_acc_en": False},
    # bfloat8_b input on a non-tile-aligned shape: mirrors the author-scoped INVALID entries of
    # eval/golden_tests/rms_norm/feature_spec.py (block-float quantization of a padded / masked tile), so a caller
    # that bypasses the harness's INVALID skip (e.g. test_translated.py) is refused natively, not run untested.
    {"dtype": ttnn.bfloat8_b, "alignment": "w_non_aligned"},
    {"dtype": ttnn.bfloat8_b, "alignment": "h_non_aligned"},
]


PROPERTIES = {
    "multi_core": {"value": True, "source": "declared"},
}


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def _axes_for(input_tensor, gamma, compute_kernel_config):
    has_gamma = gamma is not None
    axes = {
        "dtype": input_tensor.dtype,
        "layout": input_tensor.layout,
        "fp32_dest_acc_en": bool(getattr(compute_kernel_config, "fp32_dest_acc_en", True)),
        "gamma_mode": "gamma" if has_gamma else "no_gamma",
        "gamma_dtype": gamma.dtype if has_gamma else "none",
        "gamma_layout": gamma.layout if has_gamma else "none",
        "memory_layout": input_tensor.memory_config().memory_layout,
    }
    shape = tuple(input_tensor.shape)
    for axis_name, tagger in {**INPUT_TAGGERS, **GAMMA_TAGGERS}.items():
        axes[axis_name] = tagger((shape,), axes)
    return axes


def validate(input_tensor, *, gamma=None, compute_kernel_config=None, memory_config=None):
    axes = _axes_for(input_tensor, gamma, compute_kernel_config)
    if memory_config is not None:
        axes_out = memory_config.memory_layout
        if axes_out not in SUPPORTED["memory_layout"]:
            raise UnsupportedAxisValue(
                f"rms_norm: output memory_layout={axes_out!r} not in SUPPORTED {SUPPORTED['memory_layout']}"
            )
    if gamma is not None and gamma.memory_config().memory_layout not in SUPPORTED["memory_layout"]:
        raise UnsupportedAxisValue("rms_norm: gamma must be interleaved")

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"rms_norm: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")

    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"rms_norm: unsupported combination (refinement candidate): {exc}")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _structural_checks(input_tensor, gamma, epsilon):
    shape = list(input_tensor.shape)
    if len(shape) < 2:
        raise ValueError(f"rms_norm: input must have rank >= 2, got shape {shape}")
    if not math.isfinite(float(epsilon)) or float(epsilon) < 0.0:
        raise ValueError(f"rms_norm: epsilon must be finite and >= 0, got {epsilon!r}")
    if gamma is not None:
        W = shape[-1]
        gshape = list(gamma.shape)
        if gshape[-1] != W:
            raise ValueError(f"rms_norm: gamma last dim {gshape[-1]} != input last dim {W}")
        volume = 1
        for d in gshape:
            volume *= d
        if volume != W:
            raise ValueError(f"rms_norm: gamma logical volume {volume} != W={W} (expected shape (1,1,1,W))")


def rms_norm(
    input_tensor: ttnn.Tensor,
    *,
    gamma: Optional[ttnn.Tensor] = None,
    epsilon: float = 1e-6,
    compute_kernel_config: Optional[ttnn.ComputeConfigDescriptor] = None,
    memory_config: Optional[ttnn.MemoryConfig] = None,
) -> ttnn.Tensor:
    _structural_checks(input_tensor, gamma, epsilon)
    if compute_kernel_config is None:
        compute_kernel_config = default_compute_kernel_config()
    validate(input_tensor, gamma=gamma, compute_kernel_config=compute_kernel_config, memory_config=memory_config)

    device = input_tensor.device()
    output_memory_config = memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(input_tensor.shape)),
        input_tensor.dtype,
        input_tensor.layout,
        device,
        output_memory_config,
    )

    descriptor = create_program_descriptor(
        input_tensor,
        output_tensor,
        gamma=gamma,
        epsilon=float(epsilon),
        compute_kernel_config=compute_kernel_config,
    )
    io_tensors = [input_tensor] + ([gamma] if gamma is not None else []) + [output_tensor]
    return ttnn.generic_op(io_tensors, descriptor)
