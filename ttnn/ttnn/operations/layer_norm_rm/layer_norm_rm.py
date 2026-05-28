# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
layer_norm_rm — main entry point under the registry model.

Row-wise (last-dim) layer normalization on a ROW_MAJOR_LAYOUT float32
tensor. The kernels accept RM input directly (in-kernel tilize) and
produce RM output (in-kernel untilize); no host-side layout conversion.

The op declares the four registry-model artefacts inline:
    1. INPUT_TAGGERS  — categorical projections of input shape (alignment, rank).
    2. SUPPORTED      — per-axis values accepted today.
    3. EXCLUSIONS     — cells inside cartesian(SUPPORTED) refused for now.
    4. validate()     — runtime gate; raises NotImplementedError for anything
                        outside SUPPORTED or matching an EXCLUSIONS entry.

INVALID lives test-side (eval/golden_tests/layer_norm_rm/feature_spec.py);
the op file is intentionally agnostic to it.

Phase-0 envelope:
- precision  = ["fp32_hifi4_fp32acc"]   (input dtype fp32 + HiFi4 + fp32 dest acc)
- layout     = [ROW_MAJOR_LAYOUT]
- alignment  = ["tile_aligned"]         (H % 32 == 0 AND W % 32 == 0)
- rank       = [2, 3, 4]
- affine     = ["gamma_beta", "gamma_only", "no_affine"]
- affine_dtype  = [float32]
- affine_layout = [ROW_MAJOR_LAYOUT]
"""

import ttnn

from .layer_norm_rm_program_descriptor import create_program_descriptor


# ---------------------------------------------------------------------------
# Precision-name resolution
# ---------------------------------------------------------------------------
#
# The golden-test universe treats precision as a single bundled axis
# (eval/golden_tests/layer_norm_rm/feature_spec.py:PRECISION_CONFIG). The
# op's validate() inspects input_tensor.dtype + compute_kernel_config and
# resolves them back to a precision name; SUPPORTED["precision"] is then
# checked against that name. Combinations that don't match any known
# precision-name resolve to None, which will fail the SUPPORTED check.

_PRECISION_NAMES = {
    (ttnn.float32, ttnn.MathFidelity.HiFi4, True): "fp32_hifi4_fp32acc",
    (ttnn.bfloat16, ttnn.MathFidelity.HiFi4, True): "bf16_hifi4_fp32acc",
    (ttnn.bfloat16, ttnn.MathFidelity.HiFi4, False): "bf16_hifi4_bf16acc",
    (ttnn.bfloat8_b, ttnn.MathFidelity.HiFi4, False): "bf8b_hifi4_bf16acc",
}


def _resolve_precision_name(input_dtype, math_fidelity, fp32_dest_acc_en):
    return _PRECISION_NAMES.get((input_dtype, math_fidelity, bool(fp32_dest_acc_en)))


def _default_compute_kernel_config() -> ttnn.ComputeConfigDescriptor:
    """Phase-0 default — installed when caller passes None."""
    return ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        math_approx_mode=False,
    )


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS
# ---------------------------------------------------------------------------
#
# Pure-shape taggers — ignore the `axes` argument. Declared with the
# canonical `(inputs, axes)` signature so they match the test-harness
# contract (feature_matrix.apply_input_taggers).


def tag_alignment(inputs, axes):
    """Three-value alignment bucket — matches feature_spec.py's contract.

    - "tile_aligned"   if both H (-2) and W (-1) are tile-aligned.
    - "w_non_aligned"  if W is not tile-aligned (regardless of H).
    - "h_non_aligned"  if W is aligned but H is not.

    Defensive fall-back: rank-1 (or rank-0) inputs have no -2 axis, so we
    treat a missing H axis as "tile_aligned" (h=1). validate() will still
    raise on rank < 2 because such ranks are not in SUPPORTED["rank"];
    the fall-back just prevents an IndexError before the rank check fires.
    """
    shape = inputs[0]
    h = shape[-2] if len(shape) >= 2 else 1
    w = shape[-1] if len(shape) >= 1 else 1
    if w % 32 != 0:
        return "w_non_aligned"
    if h % 32 != 0:
        return "h_non_aligned"
    return "tile_aligned"


def tag_rank(inputs, axes):
    """Integer rank of the input tensor."""
    return len(inputs[0])


INPUT_TAGGERS = {
    "alignment": tag_alignment,
    "rank": tag_rank,
}


# ---------------------------------------------------------------------------
# 2. SUPPORTED
# ---------------------------------------------------------------------------
#
# Phase-0 envelope. Refinements expand each axis toward TARGET (see
# eval/golden_tests/layer_norm_rm/feature_spec.py and op_requirements.md).

SUPPORTED = {
    # Refinement 1: extended to all four PRECISION_CONFIG modes.
    # The default (compute_kernel_config=None) still resolves to
    # fp32_hifi4_fp32acc — Phase-0 callers see byte-identical behaviour.
    # bf8b_hifi4_bf16acc is only reachable when input layout is TILE
    # (bf8b in ROW_MAJOR is INVALID); for layer_norm_rm Phase 0–R1 the
    # SUPPORTED layout is still ROW_MAJOR only, so this precision name
    # is unreachable through any test path. It is listed here for
    # honesty — if Refinement 2 adds TILE_LAYOUT, bf8b becomes reachable
    # immediately without an op-file change.
    "precision": [
        "fp32_hifi4_fp32acc",
        "bf16_hifi4_fp32acc",
        "bf16_hifi4_bf16acc",
        "bf8b_hifi4_bf16acc",
    ],
    "layout": [ttnn.ROW_MAJOR_LAYOUT],
    "alignment": ["tile_aligned"],
    "rank": [2, 3, 4],
    "affine": ["gamma_beta", "gamma_only", "no_affine"],
    # Refinement 1: extended to all three affine dtypes. bf8b in
    # ROW_MAJOR is INVALID (per feature_spec.py) and bf8b in TILE is
    # currently EXCLUDED below (TILE-layout affine isn't supported
    # until Refinement 2), so bf8b is also unreachable here today; it
    # is listed for the same honesty reason as precision above.
    "affine_dtype": [ttnn.float32, ttnn.bfloat16, ttnn.bfloat8_b],
    # Both TILE and ROW_MAJOR appear: when an affine tensor is supplied
    # the op accepts ROW_MAJOR_LAYOUT only (the kernel tilizes internally).
    # When no affine is supplied, feature_spec.py canonicalises the
    # affine_layout axis to TILE_LAYOUT (so the cartesian's "no_affine"
    # case has a single non-INVALID survivor). We keep both layouts in
    # SUPPORTED and EXCLUDE the (TILE + affine-present) combos below,
    # so the canonical no_affine cell isn't xfailed in the golden suite.
    "affine_layout": [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
}


# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------
#
# When gamma/beta is actually supplied, the op requires ROW_MAJOR for
# those tensors (the in-kernel tilize wraps RM data). TILE-layout affine
# tensors are EXCLUSIONS, not INVALID: a future refinement could accept
# TILE gamma/beta directly (skipping the in-kernel tilize for those CBs).

EXCLUSIONS = [
    {"affine": "gamma_only", "affine_layout": ttnn.TILE_LAYOUT},
    {"affine": "gamma_beta", "affine_layout": ttnn.TILE_LAYOUT},
]


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------
#
# Single runtime gate. Builds the axes dict the same way the test harness
# does (tensor properties + kwargs + every tagger applied in order),
# then enforces SUPPORTED + EXCLUSIONS. INVALID is the test harness's
# concern (cells are skipped before validate() is ever called).

# When no affine tensor is supplied, feature_spec.py's INVALID list
# canonicalises the (affine_dtype, affine_layout) cartesian to a single
# survivor: (float32, TILE_LAYOUT). Mirror that here so validate()'s
# axes-dict matches the test harness's view of the canonical no_affine cell.
_NO_TENSOR_AFFINE_DTYPE = ttnn.float32
_NO_TENSOR_AFFINE_LAYOUT = ttnn.TILE_LAYOUT


def _affine_axes(gamma, beta):
    """Derive (affine, affine_dtype, affine_layout) from gamma/beta inputs.

    The three axes are bundled in feature_spec.py to mirror the way the
    golden suite iterates them. When no affine tensor is supplied
    ("no_affine"), the dtype/layout cells are canonicalised (see INVALID
    in feature_spec.py) — pick the canonical (float32, ROW_MAJOR_LAYOUT)
    pair so validate() succeeds. When only one of gamma/beta is supplied,
    the supplied tensor drives the format axes.
    """
    has_gamma = gamma is not None
    has_beta = beta is not None
    if has_gamma and has_beta:
        affine = "gamma_beta"
    elif has_gamma:
        affine = "gamma_only"
    elif has_beta:
        # Not in the TARGET universe ("beta_only" doesn't exist) — flag
        # it explicitly so the SUPPORTED check rejects with a clear msg.
        affine = "beta_only"
    else:
        affine = "no_affine"

    if has_gamma:
        affine_dtype = gamma.dtype
        affine_layout = gamma.layout
    elif has_beta:
        affine_dtype = beta.dtype
        affine_layout = beta.layout
    else:
        affine_dtype = _NO_TENSOR_AFFINE_DTYPE
        affine_layout = _NO_TENSOR_AFFINE_LAYOUT

    return affine, affine_dtype, affine_layout


def validate(
    input_tensor,
    gamma=None,
    beta=None,
    *,
    epsilon: float = 1e-5,
    compute_kernel_config: ttnn.ComputeConfigDescriptor = None,
) -> None:
    """Runtime gate.

    Raises NotImplementedError for anything outside the SUPPORTED rectangle
    or matching an EXCLUSIONS entry. INVALID is the test harness's concern.
    """
    if compute_kernel_config is None:
        compute_kernel_config = _default_compute_kernel_config()

    # math_approx_mode is not in the bundled precision name; reject it
    # explicitly (the kernel's add_unary + rsqrt path doesn't use SFPU
    # approx mode, so honoring it would silently mis-spec the contract).
    if bool(compute_kernel_config.math_approx_mode):
        raise NotImplementedError(f"layer_norm_rm: math_approx_mode=True not supported (SUPPORTED expects False)")

    precision = _resolve_precision_name(
        input_tensor.dtype,
        compute_kernel_config.math_fidelity,
        compute_kernel_config.fp32_dest_acc_en,
    )

    affine, affine_dtype, affine_layout = _affine_axes(gamma, beta)

    axes = {
        "precision": precision,
        "layout": input_tensor.layout,
        "affine": affine,
        "affine_dtype": affine_dtype,
        "affine_layout": affine_layout,
    }
    inputs = (tuple(input_tensor.shape),)
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger(inputs, axes)

    # 1. SUPPORTED — per-axis values.
    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise NotImplementedError(f"layer_norm_rm: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")

    # 2. EXCLUSIONS — cell-level inside the SUPPORTED rectangle.
    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise NotImplementedError(f"layer_norm_rm: unsupported combination (refinement candidate): {exc}")

    # Shape contracts that are not on a SUPPORTED axis.
    shape = list(input_tensor.shape)
    W = shape[-1]
    if gamma is not None:
        gamma_shape = list(gamma.shape)
        if gamma_shape != [1, 1, 1, W]:
            raise NotImplementedError(
                f"layer_norm_rm: gamma.shape={tuple(gamma_shape)} not in SUPPORTED [(1, 1, 1, {W})]"
            )
    if beta is not None:
        beta_shape = list(beta.shape)
        if beta_shape != [1, 1, 1, W]:
            raise NotImplementedError(
                f"layer_norm_rm: beta.shape={tuple(beta_shape)} not in SUPPORTED [(1, 1, 1, {W})]"
            )

    if not (epsilon > 0):
        raise NotImplementedError(f"layer_norm_rm: epsilon={epsilon} must be finite, positive")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def layer_norm(
    input_tensor: ttnn.Tensor,
    gamma: ttnn.Tensor = None,
    beta: ttnn.Tensor = None,
    *,
    epsilon: float = 1e-5,
    compute_kernel_config: ttnn.ComputeConfigDescriptor = None,
    memory_config: ttnn.MemoryConfig = None,
) -> ttnn.Tensor:
    """
    Per-row layer normalization over the last dim of a ROW_MAJOR_LAYOUT
    float32 tensor.

        y[..., h, w] = ((x[..., h, w] - mean(x[..., h, :])) /
                        sqrt(var(x[..., h, :]) + epsilon))
                       * (gamma[w] if gamma else 1)
                       + (beta[w]  if beta  else 0)

    Args:
        input_tensor: rank ∈ {2, 3, 4}, float32, ROW_MAJOR_LAYOUT, on-device.
            The final two dims must be tile-aligned (H % 32 == 0, W % 32 == 0).
        gamma: optional scale tensor of shape (1, 1, 1, W), float32, RM.
        beta:  optional shift tensor of shape (1, 1, 1, W), float32, RM.
        epsilon: positive float; added to the variance before rsqrt.
        compute_kernel_config: None (entry point installs the Phase-0
            default of math_fidelity=HiFi4, fp32_dest_acc_en=True), or an
            explicit ttnn.ComputeConfigDescriptor. Configs that don't
            resolve to a precision name in SUPPORTED["precision"] are
            rejected.
        memory_config: output memory config (default DRAM interleaved).

    Returns:
        Output tensor with the same shape, dtype, and layout as input_tensor.
    """
    if compute_kernel_config is None:
        compute_kernel_config = _default_compute_kernel_config()

    validate(
        input_tensor,
        gamma,
        beta,
        epsilon=epsilon,
        compute_kernel_config=compute_kernel_config,
    )

    device = input_tensor.device()
    output_memory_config = memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG

    # Output shape = input shape; same dtype and (RM) layout.
    output_shape = list(input_tensor.shape)

    # allocate_tensor_on_device requires POSITIONAL args, not keyword args.
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(output_shape),
        input_tensor.dtype,
        input_tensor.layout,
        device,
        output_memory_config,
    )

    program_descriptor = create_program_descriptor(
        input_tensor,
        output_tensor,
        gamma=gamma,
        beta=beta,
        epsilon=epsilon,
        compute_kernel_config=compute_kernel_config,
    )

    # Build the IO tensor list — output tensor must be LAST.
    io_tensors = [input_tensor]
    if gamma is not None:
        io_tensors.append(gamma)
    if beta is not None:
        io_tensors.append(beta)
    io_tensors.append(output_tensor)

    return ttnn.generic_op(io_tensors, program_descriptor)
