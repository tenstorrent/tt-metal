# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
groupnorm_sc_N_1_HW_C — single-core GroupNorm for (N, 1, HW, C) tensors.

Supports (C / num_groups) % 32 != 0 from Phase 0 via kernel-internal masking.

Refinement 1: adds bf16/fp32/bfloat8_b for both activation and affine dtypes,
exposes ttnn.WormholeComputeKernelConfig on the entry point, and (when
fp32_dest_acc_en is on, which is the default for this op) drives all
variance / stats intermediate CBs at Float32. The kernel itself is unchanged
— helpers handle data-format reconfig and the reader takes element-size from
CT args.
"""

from __future__ import annotations

from typing import Optional

import ttnn

from .groupnorm_sc_N_1_HW_C_program_descriptor import create_program_descriptor


# ============================================================================
# Registry-model declarations
# ============================================================================


def tag_alignment(inputs, axes):
    """Project (HW, C) onto a categorical alignment axis."""
    shape = inputs[0]
    HW, C = shape[-2], shape[-1]
    if HW % 32 == 0 and C % 32 == 0:
        return "tile_aligned"
    if C % 32 != 0:
        return "c_non_aligned"
    return "hw_non_aligned"


def tag_num_groups(inputs, axes):
    """Per-shape num_groups scalar (used by golden test driver)."""
    return inputs[1] if len(inputs) > 1 else 1


INPUT_TAGGERS = {
    "num_groups": tag_num_groups,
    "alignment": tag_alignment,
}


SUPPORTED = {
    "dtype": [ttnn.bfloat16, ttnn.float32, ttnn.bfloat8_b],
    "layout": [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
    "alignment": ["tile_aligned", "c_non_aligned"],
    "affine": ["gamma_beta", "gamma_only", "no_affine"],
    "affine_dtype": [ttnn.bfloat16, ttnn.float32, ttnn.bfloat8_b],
    "affine_layout": [ttnn.ROW_MAJOR_LAYOUT],
}


# Structural-gap cells (Refinement 1 partial landing).
#
# bf8b is exposed in SUPPORTED so the registry signals the axis is
# recognized, but every cell is EXCLUSIONS-gated for now. Empirically the
# kernel's mask-multiply + multi-iter reduce pipeline accumulates bf8b
# requantization error too quickly to stay within the 0.99 PCC golden
# threshold beyond the trivial Ct=1 + no_affine corner:
#   - single tile (Ct=1) no_affine:   PCC=0.99994 (would pass)
#   - multi tile or with affine:       max_abs blows up (inf elements)
# Fixing bf8b cleanly likely needs the stats path moved off bf8b storage
# (current Refinement 1 already does this — stats CBs at bf16 for bf8b
# input) plus a follow-up to harden the apply-phase bf8b pack. Tracked as
# the bf8b follow-up in op_requirements.md.
#
# Note: bf8b + ROW_MAJOR_LAYOUT is INVALID at the feature_spec level, and
# affine_dtype=bf8b + affine_layout=ROW_MAJOR is also INVALID — both paths
# are unreachable from the test harness. Only TILE-input bf8b + RM affine
# survive the INVALID layer; we exclude all of them here.
EXCLUSIONS = [
    {"dtype": ttnn.bfloat8_b},
]


def validate(
    input_tensor: ttnn.Tensor,
    *,
    gamma: Optional[ttnn.Tensor] = None,
    beta: Optional[ttnn.Tensor] = None,
):
    """Registry runtime gate: raises NotImplementedError on unsupported axis values."""
    axes = {
        "dtype": input_tensor.dtype,
        "layout": input_tensor.layout,
    }
    shape = list(input_tensor.shape)
    axes["alignment"] = tag_alignment((shape,), axes)

    if gamma is not None and beta is not None:
        axes["affine"] = "gamma_beta"
    elif gamma is not None:
        axes["affine"] = "gamma_only"
    else:
        axes["affine"] = "no_affine"

    affine_dtype = ttnn.bfloat16
    affine_layout = ttnn.ROW_MAJOR_LAYOUT
    if gamma is not None:
        affine_dtype = gamma.dtype
        affine_layout = gamma.layout
    elif beta is not None:
        affine_dtype = beta.dtype
        affine_layout = beta.layout
    axes["affine_dtype"] = affine_dtype
    axes["affine_layout"] = affine_layout

    for axis_name, value in axes.items():
        if axis_name in SUPPORTED and value not in SUPPORTED[axis_name]:
            raise NotImplementedError(
                f"groupnorm_sc_N_1_HW_C: axis {axis_name}={value!r} not in SUPPORTED " f"{SUPPORTED[axis_name]!r}"
            )

    for excl in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in excl.items()):
            raise NotImplementedError(f"groupnorm_sc_N_1_HW_C: input falls in EXCLUSIONS cell {excl!r}")


# ============================================================================
# Public entry point
# ============================================================================


def _default_compute_kernel_config() -> ttnn.WormholeComputeKernelConfig:
    """
    Default ComputeKernelConfig for groupnorm_sc_N_1_HW_C.

    Refinement 1 turns on fp32_dest_acc_en by default — the running
    accumulators (sum, sumsq) hold the full reduction in fp32 even when the
    I/O dtype is bf16. This is the main precision win that fixes the 18
    `supported_fail` cells on the large-SDXL shapes. Math fidelity stays at
    HiFi4 (the Phase 0 default of ComputeConfigDescriptor()). math_approx_mode
    stays off — the Rsqrt SFPU we use does NOT benefit from the approx
    variant on the precision-critical rcp_std path.
    """
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
        dst_full_sync_en=False,
    )


def groupnorm_sc_N_1_HW_C(
    input_tensor: ttnn.Tensor,
    num_groups: int,
    *,
    gamma: Optional[ttnn.Tensor] = None,
    beta: Optional[ttnn.Tensor] = None,
    eps: float = 1e-5,
    compute_kernel_config: Optional[ttnn.WormholeComputeKernelConfig] = None,
) -> ttnn.Tensor:
    """
    GroupNorm for channel-last (N, 1, HW, C) tensors.

    Args:
        input_tensor: rank-4, shape[1]==1, dtype ∈ {bf16, fp32, bf8b}, interleaved DRAM.
        num_groups: number of groups, must divide C.
        gamma: optional per-channel scale, shape (1,1,1,C), dtype ∈ {bf16, fp32, bf8b}, RM, DRAM.
        beta: optional per-channel bias, shape (1,1,1,C), dtype ∈ {bf16, fp32, bf8b}, RM, DRAM.
        eps: variance epsilon.
        compute_kernel_config: numerical configuration. Defaults to
            WormholeComputeKernelConfig(math_fidelity=HiFi4, fp32_dest_acc_en=True,
            math_approx_mode=False, dst_full_sync_en=False).

    Returns:
        Output tensor, same shape and dtype as input, always TILE_LAYOUT, DRAM.
    """
    # ----- Argument-shape validation (ValueError, separate from registry gate) -----
    shape = list(input_tensor.shape)
    if len(shape) != 4:
        raise ValueError(f"groupnorm_sc_N_1_HW_C: input must be rank 4 (N, 1, H*W, C); got rank {len(shape)}")
    if shape[1] != 1:
        raise ValueError(f"groupnorm_sc_N_1_HW_C: dim[1] must be 1; got {shape[1]}")
    N, _, HW, C = shape
    if not isinstance(num_groups, int) or num_groups <= 0:
        raise ValueError(f"groupnorm_sc_N_1_HW_C: num_groups must be a positive int; got {num_groups!r}")
    if C % num_groups != 0:
        raise ValueError(f"groupnorm_sc_N_1_HW_C: C ({C}) must be divisible by num_groups ({num_groups})")
    if gamma is not None:
        g_shape = list(gamma.shape)
        if g_shape != [1, 1, 1, C]:
            raise ValueError(f"groupnorm_sc_N_1_HW_C: gamma shape must be (1, 1, 1, C={C}); got {tuple(g_shape)}")
    if beta is not None:
        b_shape = list(beta.shape)
        if b_shape != [1, 1, 1, C]:
            raise ValueError(f"groupnorm_sc_N_1_HW_C: beta shape must be (1, 1, 1, C={C}); got {tuple(b_shape)}")

    # ----- Registry runtime gate -----
    validate(input_tensor, gamma=gamma, beta=beta)

    # ----- Compute kernel config (defaults to the precision-focused config) -----
    if compute_kernel_config is None:
        compute_kernel_config = _default_compute_kernel_config()

    device = input_tensor.device()
    output_memory_config = ttnn.DRAM_MEMORY_CONFIG

    # Output is always TILE_LAYOUT (independent of input layout) per design.
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(shape),
        input_tensor.dtype,
        ttnn.TILE_LAYOUT,
        device,
        output_memory_config,
    )

    program_descriptor = create_program_descriptor(
        input_tensor=input_tensor,
        output_tensor=output_tensor,
        num_groups=num_groups,
        gamma=gamma,
        beta=beta,
        eps=eps,
        compute_kernel_config=compute_kernel_config,
    )

    tensors = [input_tensor]
    if gamma is not None:
        tensors.append(gamma)
    if beta is not None:
        tensors.append(beta)
    tensors.append(output_tensor)  # MUST be last
    return ttnn.generic_op(tensors, program_descriptor)
