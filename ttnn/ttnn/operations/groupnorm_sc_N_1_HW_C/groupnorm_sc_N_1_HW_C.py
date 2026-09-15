# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""groupnorm_sc_N_1_HW_C — entry point + registry declarations.

GroupNorm over a channel-last ``(N, 1, H*W, C)`` tensor:

    y[n, 0, h, c] = (x[n, 0, h, c] - mean[n, g(c)]) * rsqrt(var[n, g(c)] + eps) * gamma[c] + beta[c]

with ``g(c) = c // (C / num_groups)`` and biased statistics over the ``HW * C/G`` elements of
each (image, group).  Groups that straddle 32-channel tile boundaries are handled from Phase 0
through a kernel-generated 0/1 membership matrix (see op_design.md).

Registry model (eval/op_template.py): INPUT_TAGGERS / SUPPORTED / EXCLUSIONS / validate().
INVALID lives in eval/golden_tests/groupnorm_sc_N_1_HW_C/feature_spec.py, never here.
"""

from __future__ import annotations

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .groupnorm_sc_N_1_HW_C_program_descriptor import (
    create_program_descriptor,
    default_compute_kernel_config,
    set_l1_budget_bytes_override,
    set_max_cores_override,
)

__all__ = [
    "groupnorm_sc_N_1_HW_C",
    "default_compute_kernel_config",
    "set_l1_budget_bytes_override",
    "set_max_cores_override",
    "validate",
    "INPUT_TAGGERS",
    "SUPPORTED",
    "EXCLUSIONS",
]


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS
# ---------------------------------------------------------------------------


def tag_alignment(inputs, axes):
    """(N, 1, HW, C): HW is dim -2, C is dim -1. C wins when both are off."""
    shape = inputs[0]
    HW, C = shape[-2], shape[-1]
    if HW % 32 == 0 and C % 32 == 0:
        return "tile_aligned"
    if C % 32 != 0:
        return "c_non_aligned"
    return "hw_non_aligned"


def tag_groups_alignment(inputs, axes):
    """(C / num_groups) % 32 — whole-tile groups vs. groups straddling tiles.

    Reads the sibling ``num_groups`` value from the partial axes dict. Robust to
    degenerate ``num_groups`` so the argument-validation ValueErrors still fire.
    """
    C = inputs[0][-1]
    G = axes.get("num_groups", 1)
    if not isinstance(G, int) or G <= 0:
        return "group_straddling"
    return "group_aligned" if (C // G) % 32 == 0 else "group_straddling"


INPUT_TAGGERS = {
    "alignment": tag_alignment,
    "groups_alignment": tag_groups_alignment,
}


# ---------------------------------------------------------------------------
# 2. SUPPORTED
# ---------------------------------------------------------------------------
#
# Phase 0: every TARGET dtype for activations and weights (the pipeline is dtype-agnostic: page
# formats come from the tensors, all statistics are Float32 pages accumulated in fp32 DEST), both
# input layouts, tile-aligned HW and C, BOTH group alignments (the membership-matrix path is
# alignment-agnostic), all three affine call patterns with weights in either layout. "none" is
# the canonical no-weight sentinel and is always legal. bfloat8_b + ROW_MAJOR (activation or
# weight) is structurally impossible and lives in feature_spec.INVALID, never here.
#
# Alignment (Refinement 2 — padding-independent): `hw_non_aligned` reduces the chunk holding the
# image's last tile-row with a partial REDUCE_COL scaler, so rows >= HW never enter the statistics
# whatever a TILE producer left there; `c_non_aligned` lanes >= C are dropped by the membership
# matrix's zero columns. RM input is tilized in-kernel from sticks: a ragged last tile-row / last
# channel tile is zero-filled over the NoC before only the valid sticks / lanes are read into it.

SUPPORTED = {
    "dtype": [ttnn.bfloat16, ttnn.float32, ttnn.bfloat8_b],
    "layout": [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
    "alignment": ["tile_aligned", "hw_non_aligned", "c_non_aligned"],
    "groups_alignment": ["group_aligned", "group_straddling"],
    "affine": ["gamma_beta", "gamma_only", "no_affine"],
    "affine_dtype": [ttnn.bfloat16, ttnn.float32, ttnn.bfloat8_b, "none"],
    "affine_layout": [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT, "none"],
}


# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------

EXCLUSIONS = []  # Refinement 2 lifted {layout: ROW_MAJOR, alignment: hw_non_aligned} (in-kernel ragged stick reader)


# ---------------------------------------------------------------------------
# 3b. PROPERTIES
# ---------------------------------------------------------------------------

PROPERTIES = {
    "multi_core": {"value": True, "source": "declared"},
    "bounded_cb": {"value": True, "source": "declared"},
    "math_fidelity": {"value": ["HiFi4"], "source": "declared"},
}


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def _affine_axes(gamma, beta):
    """Mirror eval/golden_tests/groupnorm_sc_N_1_HW_C/axes.py exactly."""
    if gamma is not None and beta is not None:
        affine = "gamma_beta"
    elif gamma is not None:
        affine = "gamma_only"
    else:
        affine = "no_affine"
    affine_dtype = gamma.dtype if gamma is not None else "none"
    affine_layout = gamma.layout if gamma is not None else "none"
    return affine, affine_dtype, affine_layout


def validate(input_tensor, num_groups, *, gamma=None, beta=None, compute_kernel_config=None):
    # Precision convention (.claude/references/precision_convention.md): `fp32_dest_acc_en` is read
    # from the caller's config, never silently overridden. `fp32_dest_acc_en` is not a feature_spec
    # axis for this op, so the refusal below is the runtime form of the convention's EXCLUSIONS
    # entry: the statistics path is a Float32-page matmul (`[S;Q] x E^T`, `[mean;rstd] x E_T`) whose
    # correctness contract is HiFi4 + fp32 DEST (matmul_block_helpers.hpp -> Precision), so a 16-bit
    # DEST is refused for EVERY input dtype (fp32 + False is the convention's mandatory refusal;
    # bf16/bf8b + False is a refinement candidate: intermediate CB formats must follow DEST width).
    cfg = compute_kernel_config if compute_kernel_config is not None else default_compute_kernel_config()
    if not bool(getattr(cfg, "fp32_dest_acc_en", True)):
        raise ExcludedCell(
            "groupnorm_sc_N_1_HW_C: compute_kernel_config.fp32_dest_acc_en=False is not supported "
            "(statistics are Float32-page matmuls that require fp32 DEST accumulation; refinement candidate)"
        )
    affine, affine_dtype, affine_layout = _affine_axes(gamma, beta)
    axes = {
        "dtype": input_tensor.dtype,
        "layout": input_tensor.layout,
        "num_groups": num_groups,
        "affine": affine,
        "affine_dtype": affine_dtype,
        "affine_layout": affine_layout,
    }
    shape = tuple(input_tensor.shape)
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger((shape,), axes)

    # 1. SUPPORTED — per axis
    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"groupnorm_sc_N_1_HW_C: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")

    # 2. EXCLUSIONS — cell-level inside SUPPORTED
    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"groupnorm_sc_N_1_HW_C: unsupported combination (refinement candidate): {exc}")


# ---------------------------------------------------------------------------
# Argument validation (ValueError — separate from the registry gate)
# ---------------------------------------------------------------------------


def _validate_arguments(input_tensor, num_groups, gamma, beta, eps):
    shape = tuple(input_tensor.shape)
    if len(shape) != 4:
        raise ValueError(f"groupnorm_sc_N_1_HW_C: input must have rank 4 (N, 1, HW, C); got rank {len(shape)}")
    if shape[1] != 1:
        raise ValueError(f"groupnorm_sc_N_1_HW_C: input dim[1] must be 1; got dim[1]={shape[1]}")
    C = shape[-1]
    if not isinstance(num_groups, int) or num_groups < 1 or num_groups > C or C % num_groups != 0:
        raise ValueError(
            f"groupnorm_sc_N_1_HW_C: num_groups={num_groups} must satisfy 1 <= num_groups <= C and "
            f"C % num_groups == 0 (C={C})"
        )
    for name, t in (("gamma", gamma), ("beta", beta)):
        if t is None:
            continue
        tshape = tuple(t.shape)
        if tshape != (1, 1, 1, C):
            raise ValueError(f"groupnorm_sc_N_1_HW_C: {name} must have shape (1, 1, 1, {C}); got {tshape}")
    if gamma is not None and beta is not None:
        if gamma.dtype != beta.dtype or gamma.layout != beta.layout:
            raise ValueError("groupnorm_sc_N_1_HW_C: gamma and beta must share dtype and layout")
    if not (eps > 0):
        raise ValueError(f"groupnorm_sc_N_1_HW_C: eps must be > 0; got {eps}")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def groupnorm_sc_N_1_HW_C(
    input_tensor: ttnn.Tensor,
    num_groups: int,
    *,
    gamma: ttnn.Tensor = None,
    beta: ttnn.Tensor = None,
    eps: float = 1e-5,
    compute_kernel_config=None,
) -> ttnn.Tensor:
    """GroupNorm on a channel-last (N, 1, H*W, C) tensor. Output: same shape, same dtype, TILE layout."""
    validate(input_tensor, num_groups, gamma=gamma, beta=beta, compute_kernel_config=compute_kernel_config)
    _validate_arguments(input_tensor, num_groups, gamma, beta, eps)

    device = input_tensor.device()
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(input_tensor.shape)),
        input_tensor.dtype,
        ttnn.TILE_LAYOUT,
        device,
        ttnn.DRAM_MEMORY_CONFIG,
    )

    program_descriptor = create_program_descriptor(
        input_tensor,
        output_tensor,
        num_groups=num_groups,
        gamma=gamma,
        beta=beta,
        eps=eps,
        compute_kernel_config=compute_kernel_config,
    )

    io_tensors = [input_tensor]
    if gamma is not None:
        io_tensors.append(gamma)
    if beta is not None:
        io_tensors.append(beta)
    io_tensors.append(output_tensor)  # output MUST be last
    return ttnn.generic_op(io_tensors, program_descriptor)
