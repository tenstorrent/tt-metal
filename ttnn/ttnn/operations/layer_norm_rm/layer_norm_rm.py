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
    # All four PRECISION_CONFIG modes listed for honesty. bf8b is excluded
    # at runtime (see EXCLUSIONS below) — it's structurally incompatible
    # with the TILE→RM wrap. Phase 0 / R1 listed bf8b "for honesty,
    # unreachable while layout was RM-only"; R2 made it reachable but the
    # wrap exposes a structural gap, so it moves to EXCLUSIONS.
    "precision": [
        "fp32_hifi4_fp32acc",
        "bf16_hifi4_fp32acc",
        "bf16_hifi4_bf16acc",
        "bf8b_hifi4_bf16acc",
    ],
    # Refinement 2: TILE_LAYOUT support for the input tensor. Handled at
    # the entry point via ttnn.to_layout (TILE → RM on the way in, RM →
    # TILE on the way out). The kernel itself is RM-input / RM-output —
    # the layout decision lives at the data-access boundary, not in the
    # math (see /memory-layouts §1 and softmax-R3's mirror-image wrap).
    "layout": [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT],
    # Refinement 3: non-tile-aligned shapes are now supported.
    # - "w_non_aligned" (W % 32 != 0): the LAST reduce-row tile is partially
    #   valid; the reader emits a (full, partial) scaler pair via
    #   `prepare_partial_reduce_scalers<…, partial_w>(1/W)` and compute
    #   routes `ReducePartialScaler::last_tile_at(1)` through Pass A and
    #   Pass B's `accumulate_reduce_block<SUM, REDUCE_ROW>` calls to mask
    #   the padded W positions.
    # - "h_non_aligned" (W aligned, H % 32 != 0): num_strips uses ceil
    #   division; the global last strip has < 32 valid rows. The reader and
    #   writer pass that count to the tilize-dataflow helpers which natively
    #   handle partial-row blocks (the helper still pops BLOCK_SIZE tile-pages
    #   from the CB to balance the producer count; the padded rows compute
    #   junk but never reach DRAM).
    "alignment": ["tile_aligned", "w_non_aligned", "h_non_aligned"],
    "rank": [2, 3, 4],
    "affine": ["gamma_beta", "gamma_only", "no_affine"],
    # Refinement 1: extended to all three affine dtypes. bf8b in
    # ROW_MAJOR is INVALID (per feature_spec.py); Refinement 2 lifts the
    # input-layout restriction, making bf8b in TILE reachable.
    "affine_dtype": [ttnn.float32, ttnn.bfloat16, ttnn.bfloat8_b],
    # Refinement 2: both TILE and ROW_MAJOR are now accepted for the
    # supplied affine tensor. The entry point converts TILE-layout
    # gamma/beta to ROW_MAJOR before invoking the kernel (the kernel's
    # in-kernel tilize expects RM data). For the no_affine cell,
    # feature_spec.py canonicalises (affine_dtype, affine_layout) to
    # (float32, TILE_LAYOUT) — validate() mirrors that canonical pair
    # via _NO_TENSOR_AFFINE_{DTYPE,LAYOUT} below.
    "affine_layout": [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
}


# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------
#
# Refinement 2:
#   * Removed the two `(affine=gamma_*, affine_layout=TILE)` pairs from
#     Phase 0 — TILE-layout gamma/beta is now accepted via the entry-
#     point wrap to ROW_MAJOR_LAYOUT.
#
#   * Added `{"precision": "bf8b_hifi4_bf16acc"}` — bf8b input only
#     exists in TILE layout (bf8b in RM is INVALID per feature_spec.py).
#     The entry-point's TILE→RM wrap silently downcasts bf8b → bf16
#     (ttnn.to_layout has no way to preserve a block format outside its
#     block layout), so the output round-trips back as bf16 instead of
#     bf8b. This is a structural capability gap: supporting bf8b would
#     require either an in-kernel bf8b path (not in scope here) or a
#     dedicated TILE-input/TILE-output kernel variant. A future
#     refinement can revisit this; the cell stays in SUPPORTED for
#     honesty (the listed precision name has a PRECISION_CONFIG entry)
#     but is rejected at runtime by EXCLUSIONS.

EXCLUSIONS = [
    {"precision": "bf8b_hifi4_bf16acc"},
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
    Per-row layer normalization over the last dim of an input tensor.

        y[..., h, w] = ((x[..., h, w] - mean(x[..., h, :])) /
                        sqrt(var(x[..., h, :]) + epsilon))
                       * (gamma[w] if gamma else 1)
                       + (beta[w]  if beta  else 0)

    Args:
        input_tensor: rank ∈ {2, 3, 4}, float32 / bfloat16 / bfloat8_b,
            layout ∈ {ROW_MAJOR_LAYOUT, TILE_LAYOUT}, on-device.
            The final two dims must be tile-aligned (H % 32 == 0, W % 32 == 0).
        gamma: optional scale tensor of shape (1, 1, 1, W); dtype ∈
            {float32, bfloat16, bfloat8_b}; layout ∈ {ROW_MAJOR_LAYOUT,
            TILE_LAYOUT}.
        beta:  optional shift tensor of shape (1, 1, 1, W); same dtype /
            layout surface as gamma.
        epsilon: positive float; added to the variance before rsqrt.
        compute_kernel_config: None (entry point installs the Phase-0
            default of math_fidelity=HiFi4, fp32_dest_acc_en=True), or an
            explicit ttnn.ComputeConfigDescriptor. Configs that don't
            resolve to a precision name in SUPPORTED["precision"] are
            rejected.
        memory_config: output memory config (default DRAM interleaved).

    Returns:
        Output tensor with the same shape, dtype, and layout as input_tensor.
        Layout is preserved end-to-end — the op wraps with ttnn.to_layout
        so the RM-input / RM-output kernel below sees the canonical form.
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

    # ------------------------------------------------------------------
    # Refinement 2 entry-point wrapping.
    #
    # The kernel beneath this entry point is an RM-input / RM-output
    # layer-norm. SUPPORTED["layout"] now allows TILE_LAYOUT for the
    # input tensor and TILE_LAYOUT for gamma/beta — handled here by
    # converting any TILE tensor to ROW_MAJOR before the kernel runs
    # and restoring the user's original layout on the way out.
    #
    # This mirrors softmax-R3 (which accepts RM by wrapping to TILE).
    # ------------------------------------------------------------------
    original_layout = input_tensor.layout
    needs_layout_convert = original_layout == ttnn.TILE_LAYOUT

    work_input = input_tensor
    work_gamma = gamma
    work_beta = beta
    if needs_layout_convert:
        work_input = ttnn.to_layout(work_input, ttnn.ROW_MAJOR_LAYOUT)
    if work_gamma is not None and work_gamma.layout == ttnn.TILE_LAYOUT:
        work_gamma = ttnn.to_layout(work_gamma, ttnn.ROW_MAJOR_LAYOUT)
    if work_beta is not None and work_beta.layout == ttnn.TILE_LAYOUT:
        work_beta = ttnn.to_layout(work_beta, ttnn.ROW_MAJOR_LAYOUT)

    work_shape = list(work_input.shape)

    # allocate_tensor_on_device requires POSITIONAL args, not keyword args.
    # The inner output is always RM (the kernel writes RM sticks).
    work_output = ttnn.allocate_tensor_on_device(
        ttnn.Shape(work_shape),
        work_input.dtype,
        work_input.layout,
        device,
        output_memory_config,
    )

    program_descriptor = create_program_descriptor(
        work_input,
        work_output,
        gamma=work_gamma,
        beta=work_beta,
        epsilon=epsilon,
        compute_kernel_config=compute_kernel_config,
    )

    # Build the IO tensor list — output tensor must be LAST.
    io_tensors = [work_input]
    if work_gamma is not None:
        io_tensors.append(work_gamma)
    if work_beta is not None:
        io_tensors.append(work_beta)
    io_tensors.append(work_output)

    output_tensor = ttnn.generic_op(io_tensors, program_descriptor)

    # Restore the user's original layout (TILE if they supplied TILE).
    if needs_layout_convert:
        output_tensor = ttnn.to_layout(output_tensor, original_layout)

    return output_tensor
