# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""matmul — C = A @ B, fused 2D dual-multicast kernels (registry model).

Phase 0 contract: float32 activation + float32 weight, TILE_LAYOUT,
tile-aligned M/K/N, shared 2D weight (and batched activation against it),
maxed precision (HiFi4, fp32_dest_acc_en=True).

Four registry declarations live inline here (mirroring eval/op_template.py):
INPUT_TAGGERS, SUPPORTED, EXCLUSIONS, validate(). INVALID is NOT declared here
(it lives in eval/golden_tests/matmul/feature_spec.py).
"""

from __future__ import annotations

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .matmul_program_descriptor import create_program_descriptor


# ---------------------------------------------------------------------------
# default compute config — single source of truth (also read by axes.py)
# ---------------------------------------------------------------------------
def default_compute_kernel_config():
    """Phase 0 maxed-precision default: HiFi4, fp32 accumulator, no approx."""
    return ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        math_approx_mode=False,
    )


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS (op-local)
# ---------------------------------------------------------------------------
def tag_alignment(inputs, axes):
    """Most-impactful non-aligned dim, precedence K > N > M (else tile_aligned)."""
    A_shape, B_shape = inputs[0], inputs[1]
    M, K, N = A_shape[-2], A_shape[-1], B_shape[-1]
    if M % 32 == 0 and K % 32 == 0 and N % 32 == 0:
        return "tile_aligned"
    if K % 32 != 0:
        return "k_non_aligned"
    if N % 32 != 0:
        return "n_non_aligned"
    return "m_non_aligned"


def tag_weight_batch(inputs, axes):
    """'single' for a shared 2D weight (or leading dims all 1), else 'batched'."""
    B_shape = inputs[1]
    leading = list(B_shape[:-2])
    return "single" if (len(leading) == 0 or all(d == 1 for d in leading)) else "batched"


INPUT_TAGGERS = {"alignment": tag_alignment, "weight_batch": tag_weight_batch}


# ---------------------------------------------------------------------------
# 2. SUPPORTED (Refinement 1: numerical configurability;
#                Refinement 2: non-tile-aligned M / K / N)
# ---------------------------------------------------------------------------
# dtype (activation) and weight_dtype are INDEPENDENT axes — the cartesian
# covers all 9 (act, weight) pairs, including the bf16-act × fp32-weight mixed
# path. fp32_dest_acc_en spans both the maxed (True) and 16-bit-DEST (False)
# accumulator. The {fp32, acc=False} corner is refused via EXCLUSIONS.
#
# Refinement 2 — alignment spans all four values (M, K, N each non-aligned, plus
# multi-non-aligned which the K>N>M tagger precedence folds into one of the
# three). matmul needs NO host-side pad/slice and NO in-kernel masking for this:
# ttnn's TILE_LAYOUT representation zero-fills the partial-tile padding at
# from_torch time (verified empirically for fp32/bf16/bf8b — see
# changelog.md / probes 015-016), the descriptor already counts tiles with
# ceil_div (so the partial last M/K/N tile IS processed), and the K dot-product
# over the zero pad is 0*0=0. M/N output-padding is sliced off by the output's
# logical shape on to_torch. The invariant is ALSO preserved compositionally:
# this op's own output zero-fills its M/N pad (output pad rows/cols derive from
# zero input pad), so a non-aligned matmul output fed as the next matmul's K is
# still zero-padded. See the kernel-head + descriptor comments.
SUPPORTED = {
    "dtype": [ttnn.float32, ttnn.bfloat16, ttnn.bfloat8_b],
    "weight_dtype": [ttnn.float32, ttnn.bfloat16, ttnn.bfloat8_b],
    "layout": [ttnn.TILE_LAYOUT],
    "fp32_dest_acc_en": [True, False],
    "alignment": ["tile_aligned", "k_non_aligned", "n_non_aligned", "m_non_aligned"],
    "weight_batch": ["single"],
}


# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------
EXCLUSIONS = [
    # Maxed (fp32) activation demands a maxed (fp32) accumulator: fp32 input
    # with a 16-bit DEST accumulator is legal-but-lossy and refused by the
    # precision convention (refinement candidate, not structurally impossible).
    {"dtype": ttnn.float32, "fp32_dest_acc_en": False},
]


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------
def _fp32_acc(compute_kernel_config):
    cfg = compute_kernel_config or default_compute_kernel_config()
    return bool(getattr(cfg, "fp32_dest_acc_en", True))


def validate(input_tensor, weight, *, compute_kernel_config=None):
    """Structural shape-contract checks (ValueError) THEN the registry gate."""
    A_shape = list(input_tensor.shape)
    B_shape = list(weight.shape)

    # --- structural shape-contract checks (raise BEFORE the registry gate) ---
    if len(A_shape) < 2:
        raise ValueError(f"matmul: activation rank {len(A_shape)} < 2")
    if len(B_shape) < 2:
        raise ValueError(f"matmul: weight rank {len(B_shape)} < 2")
    if A_shape[-1] != B_shape[-2]:
        raise ValueError(f"matmul: K mismatch A[-1]={A_shape[-1]} != B[-2]={B_shape[-2]}")
    A_lead = A_shape[:-2]
    B_lead = B_shape[:-2]
    b_has_real_batch = len(B_lead) > 0 and not all(d == 1 for d in B_lead)
    if b_has_real_batch and list(B_lead) != list(A_lead):
        raise ValueError(f"matmul: batched weight leading dims {B_lead} != activation " f"leading dims {A_lead}")

    # --- registry gate (SUPPORTED per-axis, then EXCLUSIONS) ---
    axes = {
        "dtype": input_tensor.dtype,
        "weight_dtype": weight.dtype,
        "layout": input_tensor.layout,
        "fp32_dest_acc_en": _fp32_acc(compute_kernel_config),
    }
    tagger_inputs = (A_shape, B_shape)
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger(tagger_inputs, axes)

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"matmul: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")
    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"matmul: unsupported combination (refinement candidate): {exc}")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def matmul(
    input_tensor: ttnn.Tensor,
    weight: ttnn.Tensor,
    *,
    compute_kernel_config: ttnn.ComputeConfigDescriptor = None,
) -> ttnn.Tensor:
    """Compute C = A @ B with the 2D dual-multicast fused kernels."""
    validate(input_tensor, weight, compute_kernel_config=compute_kernel_config)

    cfg = compute_kernel_config or default_compute_kernel_config()

    device = input_tensor.device()

    # Output: A's leading dims with the last dim swapped to N, activation dtype.
    A_shape = list(input_tensor.shape)
    N = list(weight.shape)[-1]
    output_shape = A_shape[:-1] + [N]

    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(output_shape),
        input_tensor.dtype,
        ttnn.TILE_LAYOUT,
        device,
        ttnn.DRAM_MEMORY_CONFIG,
    )

    program_descriptor = create_program_descriptor(input_tensor, weight, output_tensor, cfg)

    # Output tensor MUST be last in the list.
    return ttnn.generic_op([input_tensor, weight, output_tensor], program_descriptor)
