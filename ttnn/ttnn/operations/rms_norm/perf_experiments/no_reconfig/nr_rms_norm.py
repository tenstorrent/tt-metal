# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""rms_norm — row-wise RMS normalization over the last dimension.

    output[..., r, c] = input[..., r, c]
                        * rsqrt( (1/W) * sum_c' input[..., r, c']^2 + epsilon )
                        * gamma[c]

`W` is always the TRUE (unpadded) last-dimension extent — tile padding never
enters the denominator (op_design.md "Regime selection", risk R1).

Registry-model op file: INPUT_TAGGERS / SUPPORTED / EXCLUSIONS / validate().
`INVALID` deliberately does NOT live here — it belongs to
eval/golden_tests/rms_norm/feature_spec.py.
"""

from __future__ import annotations

from typing import Optional

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .nr_descriptor import create_program_descriptor

# ---------------------------------------------------------------------------
# Precision — one exported factory, per references/precision_convention.md.
# The golden axis tagger (eval/golden_tests/rms_norm/axes.py) reads this same
# function, so the Phase 0 default lives in exactly one place.
# ---------------------------------------------------------------------------


def default_compute_kernel_config() -> "ttnn.ComputeConfigDescriptor":
    """Phase 0 maxed-out precision corner. A FACTORY — fresh descriptor per call."""
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi4
    cfg.fp32_dest_acc_en = True
    cfg.math_approx_mode = False
    return cfg


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS — shape facets projected onto categorical axes.
# ---------------------------------------------------------------------------


def tag_alignment(inputs, axes):
    """Three-value alignment split (feature_spec.py contract).

    tile_aligned  — H and W both multiples of 32
    w_non_aligned — W not a multiple of 32 (H may or may not be)
    h_non_aligned — W aligned, H not aligned
    """
    shape = inputs[0]
    if shape[-1] % 32 != 0:
        return "w_non_aligned"
    if shape[-2] % 32 != 0:
        return "h_non_aligned"
    return "tile_aligned"


def tag_rank(inputs, axes):
    return len(inputs[0])


INPUT_TAGGERS = {
    "alignment": tag_alignment,
    "rank": tag_rank,
}


# ---------------------------------------------------------------------------
# 2. SUPPORTED — Phase 0 coverage. One entry per axis the golden
#    feature_spec TARGET enumerates, so out-of-rectangle cells xfail cleanly.
# ---------------------------------------------------------------------------

SUPPORTED = {
    "dtype": [ttnn.float32, ttnn.bfloat16, ttnn.bfloat8_b],
    # Precision is a two-axis model (dtype x fp32_dest_acc_en).  BOTH DEST
    # accumulator widths are native: fp32_dest_acc_en=False halves the DEST datum
    # width and therefore DOUBLES the tile capacity (4 -> 8 at half-sync), which
    # `_dest_limit()` already reports and the block solver re-sweeps against.
    "fp32_dest_acc_en": [True, False],
    "layout": [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
    "alignment": ["tile_aligned", "w_non_aligned", "h_non_aligned"],
    "rank": [2, 3, 4],
    "gamma_mode": ["gamma", "no_gamma"],
    # "none" is the no-weight sentinel and is ALWAYS legal.  gamma may be at a
    # different dtype than the activations (mixed-precision LLMs).
    "gamma_dtype": [ttnn.float32, ttnn.bfloat16, ttnn.bfloat8_b, "none"],
    "gamma_layout": [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT, "none"],
    "memory_layout": [ttnn.TensorMemoryLayout.INTERLEAVED],
}


# ---------------------------------------------------------------------------
# 3. EXCLUSIONS — cells inside cartesian(SUPPORTED) refused for now.
# ---------------------------------------------------------------------------

EXCLUSIONS = [
    # fp32 activations with a bf16 DEST accumulator is a silent precision
    # downgrade; refused natively rather than quietly accepted.
    {"dtype": ttnn.float32, "fp32_dest_acc_en": False},
]


# ---------------------------------------------------------------------------
# 3b. PROPERTIES
# ---------------------------------------------------------------------------

PROPERTIES = {
    "multi_core": {"value": True, "source": "verified"},
    "bounded_cb": {"value": True, "source": "declared"},
    "math_fidelity": {"value": ["LoFi", "HiFi2", "HiFi3", "HiFi4"], "source": "declared"},
}


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def _build_axes(input_tensor, gamma, compute_kernel_config):
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
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger((list(input_tensor.shape),), axes)
    return axes


def validate(input_tensor, *, gamma=None, epsilon=1e-6, compute_kernel_config=None, memory_config=None):
    """Runtime support gate. Called as the entry point's first statement."""
    shape = list(input_tensor.shape)
    if len(shape) < 2:
        raise ValueError(f"rms_norm: input rank must be >= 2, got rank {len(shape)} for shape {shape}")

    if gamma is not None:
        gamma_shape = list(gamma.shape)
        if gamma_shape[-1] != shape[-1]:
            raise ValueError(
                f"rms_norm: gamma last dimension {gamma_shape[-1]} must match " f"input last dimension {shape[-1]}"
            )

    cfg = compute_kernel_config if compute_kernel_config is not None else default_compute_kernel_config()
    axes = _build_axes(input_tensor, gamma, cfg)

    # 1. SUPPORTED — per-axis
    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"rms_norm: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")

    # 1b. The OUTPUT placement is a separate surface: `memory_config` selects
    #     where the result lands, and the writer only knows how to address an
    #     interleaved buffer.  Without this gate an interleaved-in /
    #     sharded-out request would allocate a sharded output and then write it
    #     through an interleaved TensorAccessor — silent corruption instead of
    #     an honest refusal.
    if memory_config is not None and memory_config.memory_layout not in SUPPORTED["memory_layout"]:
        raise UnsupportedAxisValue(
            f"rms_norm: memory_config.memory_layout={memory_config.memory_layout!r} "
            f"not in SUPPORTED {SUPPORTED['memory_layout']}"
        )

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
    memory_config: "ttnn.MemoryConfig" = None,
    _levers: dict = None,
) -> "ttnn.Tensor":
    """Root-mean-square normalization along the last dimension.

    Both TILE_LAYOUT and ROW_MAJOR_LAYOUT inputs are handled natively — no
    host-side to_layout / tilize / untilize / pad / slice anywhere on this path.
    Non-tile-aligned H and W are likewise native; the RMS denominator counts
    only the real elements.

    Precondition (mirrors toy_variance): for a TILE_LAYOUT input with a
    non-tile-aligned W, the implicit tile padding must be FINITE. The masked
    reduce multiplies pad columns by zero, and inf * 0 = NaN. Call
    ttnn.fill_implicit_tile_padding(x, 0.0) first if the padding may hold
    inf/NaN garbage.

    `memory_config` selects the OUTPUT placement (default: the input's).  Phase 0
    supports only INTERLEAVED, so a sharded request is refused by validate()
    through the `memory_layout` axis rather than silently ignored.

    `_levers` is an INTERNAL perf-bench hook (see
    rms_norm_program_descriptor.LEVER_DEFAULTS): a dict overriding individual
    perf knobs so `_bench_rms_norm.py` can measure a lever's counterfactual
    without editing a kernel.  Omitted = every lever at its applied default.
    """
    validate(
        input_tensor,
        gamma=gamma,
        epsilon=epsilon,
        compute_kernel_config=compute_kernel_config,
        memory_config=memory_config,
    )

    cfg = compute_kernel_config if compute_kernel_config is not None else default_compute_kernel_config()

    device = input_tensor.device()

    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(input_tensor.shape)),
        input_tensor.dtype,
        input_tensor.layout,
        device,
        memory_config if memory_config is not None else input_tensor.memory_config(),
    )

    program_descriptor = create_program_descriptor(
        input_tensor,
        gamma,
        output_tensor,
        epsilon=epsilon,
        compute_kernel_config=cfg,
        levers=_levers,
    )

    tensors = [input_tensor] if gamma is None else [input_tensor, gamma]
    tensors.append(output_tensor)  # output MUST be last
    return ttnn.generic_op(tensors, program_descriptor)
