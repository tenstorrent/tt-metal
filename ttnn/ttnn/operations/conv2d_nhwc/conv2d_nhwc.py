# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""conv2d_nhwc — 2D convolution over a channels-last (NHWC) ROW_MAJOR activation.

Implemented as implicit im2col + blocked matmul, with the output-position (M)
dimension distributed across the full Tensix grid:

    out[n, ho, wo, co] = sum_{kh, kw, ci}
        in[n, ho*S + D*kh - P, wo*S + D*kw - P, ci] * W[kh, kw, ci, co] + bias[co]

The activation is never materialized as an im2col matrix. The reader kernel
gathers, per (output-position, tap, channel-slice), a contiguous run of the
NHWC activation directly into a row-major CB; the compute kernel tilizes that
block inside `matmul_block`'s per-K-block hook and accumulates.

Registry-model declarations (INPUT_TAGGERS / SUPPORTED / EXCLUSIONS /
validate) live inline here. INVALID is *not* declared here — it lives in
`eval/golden_tests/conv2d_nhwc/feature_spec.py`.
"""

from __future__ import annotations

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .conv2d_nhwc_program_descriptor import (
    create_program_descriptor,
    elem_size_of,
    group_blocking,
)


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS
# ---------------------------------------------------------------------------
#
# Every tagger receives the bundled 7-tuple
#   inputs = (input_shape, C_out, kernel_size, padding, stride, groups, dilation)
# plus the partial axes dict (unused by these four — they are pure functions of
# the bundled scalars).


def tag_channel_alignment(inputs, axes):
    """C_in / C_out tile alignment — drives K-dim vs N-dim tile masking."""
    input_shape, C_out = inputs[0], inputs[1]
    C_in = input_shape[-1]
    a_in = (C_in % 32) == 0
    a_out = (C_out % 32) == 0
    if a_in and a_out:
        return "tile_aligned"
    if a_out:  # only C_in is bad
        return "c_in_unaligned"
    if a_in:  # only C_out is bad
        return "c_out_unaligned"
    return "both_unaligned"


def tag_stride_mode(inputs, axes):
    """stride == 1 → unit, else strided."""
    return "unit" if inputs[4] == 1 else "strided"


def tag_groups_mode(inputs, axes):
    """dense / depthwise / grouped. `dense` is checked FIRST so that
    (C_in == 1, groups == 1) classifies as dense, not depthwise."""
    C_in = inputs[0][-1]
    groups = inputs[5]
    if groups == 1:
        return "dense"
    if groups == C_in:
        return "depthwise"
    return "grouped"


def tag_dilation_mode(inputs, axes):
    """dilation == 1 → unit, else dilated."""
    return "unit" if inputs[6] == 1 else "dilated"


INPUT_TAGGERS = {
    "channel_alignment": tag_channel_alignment,
    "stride_mode": tag_stride_mode,
    "groups_mode": tag_groups_mode,
    "dilation_mode": tag_dilation_mode,
}


# ---------------------------------------------------------------------------
# 2. SUPPORTED
# ---------------------------------------------------------------------------

SUPPORTED = {
    # Activation dtype (also the output dtype). bfloat8_b is not in TARGET:
    # the activation is ROW_MAJOR by op contract and block-float needs TILE.
    "dtype": [ttnn.float32, ttnn.bfloat16],
    # Prepared-weight dtype. Independent of the activation dtype — the matmul
    # unpackers carry per-operand data formats, and matmul_block reconfigs
    # srcA/srcB separately, so every dtype x weight_dtype cell pipes through.
    "weight_dtype": [ttnn.float32, ttnn.bfloat16, ttnn.bfloat8_b],
    "layout": [ttnn.ROW_MAJOR_LAYOUT],
    "bias_mode": ["with_bias", "no_bias"],
    # Refinement 3: C_in and C_out no longer have to be multiples of 32.
    #   K-dim (C_in)  — `Ct = ceil(C_in_pg / 32)`; the reader reads only the
    #     bytes a stick actually owns and zero-fills the channel tail, and
    #     prepare_conv2d_weights zero-pads the weight's K rows to `Ct*32` per
    #     tap so the padded K-lanes multiply zero by zero.
    #   N-dim (C_out) — `Nt = ceil(C_out / 32)`; the writer truncates the last
    #     N-block's scatter to the columns the output stick actually owns, and
    #     the prepared weight/bias carry ttnn's zero tile padding in the
    #     garbage columns.
    "channel_alignment": ["tile_aligned", "c_in_unaligned", "c_out_unaligned", "both_unaligned"],
    # Stride and dilation are pure reader address arithmetic: the gather already
    # computes `hi = ho*stride + dilation*kh - padding`, and the descriptor
    # derives H_out / W_out from the dilated effective kernel. Both axes went to
    # TARGET in Refinement 2 with no kernel change beyond that.
    "stride_mode": ["unit", "strided"],
    # Refinement 4: grouped and depthwise convolution. The C_out axis is
    # partitioned into tile-aligned *column blocks*, each bundling
    # `G_blk = 32 / gcd(C_out/groups, 32)` consecutive groups; the reader
    # offsets its im2col gather by the column block's channel base and
    # prepare_conv2d_weights emits a block-diagonal weight inside the block.
    # See `GroupBlocking` in conv2d_nhwc_program_descriptor.py.
    "groups_mode": ["dense", "grouped", "depthwise"],
    "dilation_mode": ["unit", "dilated"],
}


# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------
#
# EXCLUSIONS holds cells *inside* cartesian(SUPPORTED) that the op refuses
# for now. Still empty after Refinement 1.
#
# Mixed precision (activation dtype != weight dtype) was the expected EXCLUSIONS
# candidate once R1 widened both dtype axes — the worry being that srcA and srcB
# carrying different L1 formats would need a per-operand reconfig regime the
# kernel does not have. It turned out not to: `matmul_block` reconfigures the
# two unpackers independently from each CB's own declared format, so all six
# (dtype x weight_dtype) cells pipe through unmodified. Measured on the shape
# ladder in tests/.../precision_matrix_results.md — worst cell is
# fp32-act x bf8b-weight at rel_rms 1.72e-02 against a 2.0e-02 band, and the
# full 1927-cell golden sweep is clean. Nothing to exclude.

EXCLUSIONS = []


# ---------------------------------------------------------------------------
# 3b. PROPERTIES
# ---------------------------------------------------------------------------

PROPERTIES = {
    # Refinement 2: the M dimension (N*H_out*W_out output positions, grouped
    # into `Mt`-tile-row blocks) is split across the full compute-with-storage
    # grid via ttnn.split_work_to_cores. No inter-core data dependency.
    "multi_core": {"value": True, "source": "verified"},
    "bounded_cb": {"value": True, "source": "declared"},
    # User-selectable via `compute_kernel_config`. Defaults: HiFi4 when both
    # operands are fp32, HiFi2 otherwise (HiFi4 requests with a non-fp32
    # operand + fp32_dest_acc_en are clamped to HiFi3 — WH-B0 issue #38306).
    "math_fidelity": {"value": ["HiFi4", "HiFi3", "HiFi2", "LoFi"], "source": "declared"},
}


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


TILE = 32


def _round_up_to_tile(n):
    """`n` rounded up to the next multiple of 32 (the tile side)."""
    return ((int(n) + TILE - 1) // TILE) * TILE


def _bundle(input_shape, C_out, kernel_size, padding, stride, groups, dilation):
    return (tuple(input_shape), int(C_out), int(kernel_size), int(padding), int(stride), int(groups), int(dilation))


def _validate_axes(*, inputs, dtype, weight_dtype, layout, bias_mode=None, skip_axes=()):
    """Shared runtime gate. `bias_mode` is omitted by the prepare_* entry
    points, which cannot observe it — every other axis is checked there too.

    `skip_axes` drops axes the caller genuinely cannot observe (the bias path
    has no input-channel dimension, so it cannot decide `groups_mode`). A
    skipped axis is simply not gated there; the forward entry point and
    prepare_conv2d_weights, which both see the real C_in, still gate it.
    """
    axes = {
        "dtype": dtype,
        "weight_dtype": weight_dtype,
        "layout": layout,
    }
    if bias_mode is not None:
        axes["bias_mode"] = bias_mode
    for axis_name, tagger in INPUT_TAGGERS.items():
        if axis_name in skip_axes:
            continue
        axes[axis_name] = tagger(inputs, axes)

    # 1. SUPPORTED — per axis
    for axis, allowed in SUPPORTED.items():
        if axis not in axes:
            continue  # bias_mode when called from prepare_*, or a skipped axis
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"conv2d_nhwc: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")

    # 2. EXCLUSIONS — cell-level inside SUPPORTED
    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"conv2d_nhwc: unsupported combination (refinement candidate): {exc}")

    return axes


def validate(
    input_tensor,
    weight_tensor,
    kernel_size,
    *,
    bias=None,
    padding=0,
    stride=1,
    groups=1,
    dilation=1,
):
    """Runtime gate for the forward entry point.

    Order matters: argument sanity (what the taggers need to run) → SUPPORTED
    / EXCLUSIONS → the remaining shape checks. A cell outside SUPPORTED must
    surface as a *support refusal* (NotImplementedError), not as a ValueError
    about some downstream shape, because the golden harness decorates those
    cells xfail(strict, raises=NotImplementedError).
    """
    _check_args(input_tensor, kernel_size, padding, stride, groups, dilation)
    input_memory_layout = input_tensor.memory_config().memory_layout
    supported_input_memory_layouts = {
        ttnn.TensorMemoryLayout.INTERLEAVED,
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    }
    if input_memory_layout not in supported_input_memory_layouts:
        raise UnsupportedAxisValue(
            "conv2d_nhwc: input_memory_layout="
            f"{input_memory_layout!r} not in SUPPORTED {supported_input_memory_layouts}"
        )
    C_out = int(weight_tensor.shape[-1])
    axes = _validate_axes(
        inputs=_bundle(tuple(input_tensor.shape), C_out, kernel_size, padding, stride, groups, dilation),
        dtype=input_tensor.dtype,
        weight_dtype=weight_tensor.dtype,
        layout=input_tensor.layout,
        bias_mode="with_bias" if bias is not None else "no_bias",
    )
    _check_shapes(input_tensor, weight_tensor, kernel_size, padding, stride, groups, dilation, bias=bias)
    return axes


def _check_args(input_tensor, kernel_size, padding, stride, groups, dilation):
    """Argument sanity the taggers depend on (raises ValueError)."""
    if len(input_tensor.shape) != 4:
        raise ValueError(f"conv2d_nhwc: input must be 4D [N,H,W,C_in], got {tuple(input_tensor.shape)}")
    if kernel_size < 1 or stride < 1 or dilation < 1 or groups < 1 or padding < 0:
        raise ValueError(f"conv2d_nhwc: bad conv params k={kernel_size} s={stride} p={padding} g={groups} d={dilation}")


def _check_shapes(input_tensor, weight_tensor, kernel_size, padding, stride, groups, dilation, *, bias=None):
    """Shape consistency that is not an axis (raises ValueError)."""
    N, H, W, C_in = (int(d) for d in input_tensor.shape)
    C_out = int(weight_tensor.shape[-1])
    if C_in % groups or C_out % groups:
        raise ValueError(f"conv2d_nhwc: groups={groups} must divide C_in={C_in} and C_out={C_out}")
    eff_k = dilation * (kernel_size - 1) + 1
    if H + 2 * padding < eff_k or W + 2 * padding < eff_k:
        raise ValueError(
            f"conv2d_nhwc: effective kernel {eff_k} does not fit in padded input " f"({H}+2*{padding}, {W}+2*{padding})"
        )

    # The kernel indexes the prepared weight as a [K, C_out] matrix with
    # K = kH*kW*ceil(chans_cb/32)*32 rows (see prepare_conv2d_weights), where
    # `chans_cb` is the per-column-block channel window the grouped blocking
    # picked (== C_in for a dense conv). The per-tap channel count is rounded
    # UP to a tile so that an unaligned channel run still lands each tap on a
    # tile boundary — the reader's K-block index arithmetic assumes exactly
    # `Ct*32` weight rows per tap. A weight that was not produced by
    # prepare_conv2d_weights for this exact conv config would silently read
    # past its own rows, so check it here.
    gb = group_blocking(C_in=C_in, C_out=C_out, groups=groups, elem_size=elem_size_of(input_tensor.dtype))
    expected_k_rows = kernel_size * kernel_size * _round_up_to_tile(gb.chans_cb)
    k_rows = int(weight_tensor.shape[-2])
    if k_rows != expected_k_rows:
        raise ValueError(
            f"conv2d_nhwc: weight has {k_rows} K-rows, expected "
            f"kernel_size^2 * roundup32(chans_per_column_block) = {expected_k_rows}. Did you call "
            f"prepare_conv2d_weights with the same kernel_size/groups/input_dtype?"
        )
    if bias is not None and int(bias.shape[-1]) != C_out:
        raise ValueError(f"conv2d_nhwc: bias last dim {int(bias.shape[-1])} != C_out {C_out}")


# ---------------------------------------------------------------------------
# Weight / bias preparation
# ---------------------------------------------------------------------------


def prepare_conv2d_weights(
    torch_weight,
    *,
    kernel_size,
    stride=1,
    padding=0,
    groups=1,
    dilation=1,
    input_dtype,
    weight_dtype,
    input_memory_config,
    device,
):
    """`[C_out, C_in/G, kH, kW]` (PyTorch convention) → device TILE tensor of
    logical shape `[1, 1, kH*kW*roundup32(chans_cb), C_out]`.

    Row index is `k_row = (kh*kW + kw) * roundup32(chans_cb) + local_ci` —
    tap-major, channel-minor, with the per-tap channel run **zero-padded up to
    a tile boundary**. That padding is the K-dim half of Refinement 3: the
    reader gathers `Ct*32` channels per tap out of an activation stick that
    only owns some of them and zero-fills the difference, so the matching
    weight rows must exist and must be zero. Padding here rather than in the
    kernel keeps every tap on a tile boundary, which is what the reader's
    `k_block -> (tap, channel-slice)` decomposition assumes.

    **Grouped / depthwise (Refinement 4).** `chans_cb` is the channel window of
    one *column block* — a bundle of `G_blk` consecutive groups wide enough to
    tile-align its column count (see `GroupBlocking` in the descriptor). For a
    dense conv `chans_cb == C_in` and this is the Phase 0 layout verbatim. For
    `groups > 1`, `local_ci` indexes into the column block's window rather than
    into the whole channel axis:

        column co  ->  group g = co // (C_out/G)
                   ->  column block  cb = g // G_blk
                   ->  the reader hands the matmul channels
                       [cb*chans_cb, cb*chans_cb + Ct*32)
                   ->  so g's weights sit at local rows
                       [(g % G_blk)*(C_in/G), ... + C_in/G)

    Every other row of that column is zero, which is what makes the extra
    channels a column block drags in contribute nothing. The matrix stays
    `[K, C_out]` with **no** wasted columns — different column blocks reuse the
    same K rows with different channel meanings — so this layout is strictly
    smaller than the Phase 0 dense `[kH*kW*(C_in/G), C_out]` one for grouped
    convs and identical for dense ones.

    C_out is *not* padded in the logical shape — the op derives C_out from
    `weight_tensor.shape[-1]`, so widening it here would silently widen the
    output tensor. ttnn's TILE conversion zero-pads the last dim up to the tile
    grid on its own, which is exactly the "garbage columns are zero" property
    the writer's N-dim truncation wants.

    TILE layout is the NoC-alignment seam: tile page sizes (4096 / 2048 / 1088
    bytes) are multiples of every DRAM read alignment in the fleet, so the
    writer reads whole pages with no offset arithmetic.
    """
    import torch  # local import: torch is a caller-side dependency, not an op one

    C_out, C_in_pg, kH, kW = (int(d) for d in torch_weight.shape)
    C_in = C_in_pg * groups
    # The taggers read only the channel count out of input_shape, so a 1x1
    # spatial placeholder keeps every derivable axis exact here.
    _validate_axes(
        inputs=_bundle((1, 1, 1, C_in), C_out, kernel_size, padding, stride, groups, dilation),
        dtype=input_dtype,
        weight_dtype=weight_dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    if (kH, kW) != (kernel_size, kernel_size):
        raise ValueError(f"conv2d_nhwc: weight spatial dims {(kH, kW)} != kernel_size {kernel_size}")

    # [C_out, C_in/G, kH, kW] -> [kH, kW, C_in/G, C_out]
    w_perm = torch_weight.permute(2, 3, 1, 0).contiguous()

    gb = group_blocking(C_in=C_in, C_out=C_out, groups=groups, elem_size=elem_size_of(input_dtype))
    # Rows per tap: the column block's channel window, rounded up to a tile so
    # every tap starts on a tile boundary (the reader's K-block decomposition
    # assumes exactly Ct*32 rows per tap).
    k_rows_per_tap = _round_up_to_tile(gb.chans_cb)
    cols_per_group = C_out // groups

    # Scatter each group's [C_in/G] channel rows to its slot inside the column
    # block's window. Rows outside a column's own group stay zero, which is what
    # makes the extra channels a column block drags in contribute nothing.
    # For groups == 1 this is G_blk == 1, lb == 0 => the Phase 0 layout verbatim.
    packed = torch.zeros(kH, kW, k_rows_per_tap, C_out, dtype=w_perm.dtype)
    for g in range(groups):
        lb = (g % gb.G_blk) * C_in_pg
        c0 = g * cols_per_group
        packed[:, :, lb : lb + C_in_pg, c0 : c0 + cols_per_group] = w_perm[:, :, :, c0 : c0 + cols_per_group]
    w = packed.reshape(1, 1, kH * kW * k_rows_per_tap, C_out)
    return ttnn.from_torch(
        w,
        dtype=weight_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def prepare_conv2d_bias(
    torch_bias,
    *,
    kernel_size,
    stride=1,
    padding=0,
    groups=1,
    dilation=1,
    input_dtype,
    weight_dtype,
    input_memory_config,
    device,
):
    """`[C_out]` → device TILE tensor of logical shape `[1, 1, 32, C_out]`.

    `add_bias_bcast_rows` with `BiasBroadcast::RowBroadcast` reads row 0 of
    each bias tile and broadcasts it down; the other 31 rows are don't-care
    but are filled so the tensor stays self-describing (and a future
    `Elementwise` refinement is a host-side no-op).
    """
    C_out = int(torch_bias.shape[-1])
    # C_in is genuinely not derivable from a bias tensor, and `groups_mode`
    # (dense / grouped / depthwise) is a function of C_in — so the bias path
    # cannot decide that axis and must not guess it. Refinement 4 dropped the
    # old `C_in_hint = 32 * groups` fake, which reported depthwise convs as
    # "grouped": harmless while both were unsupported, actively wrong now that
    # both are. `channel_alignment` reads only C_out here, which the bias does
    # own, so a tile-aligned placeholder makes that axis a pure function of
    # C_out — exact for every axis still gated here. The forward entry point and
    # prepare_conv2d_weights both see the real C_in and gate `groups_mode`.
    _validate_axes(
        inputs=_bundle((1, 1, 1, TILE), C_out, kernel_size, padding, stride, groups, dilation),
        dtype=input_dtype,
        weight_dtype=weight_dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        skip_axes={"groups_mode"},
    )

    b = torch_bias.reshape(1, 1, 1, C_out).expand(1, 1, 32, C_out).contiguous()
    return ttnn.from_torch(
        b,
        dtype=weight_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def conv2d_nhwc(
    input_tensor: ttnn.Tensor,
    weight_tensor: ttnn.Tensor,
    kernel_size: int,
    *,
    bias: ttnn.Tensor = None,
    padding: int = 0,
    stride: int = 1,
    groups: int = 1,
    dilation: int = 1,
    memory_config: ttnn.MemoryConfig = None,
    compute_kernel_config=None,
) -> ttnn.Tensor:
    """2D convolution over an NHWC ROW_MAJOR activation.

    `compute_kernel_config` is an optional `ttnn.WormholeComputeKernelConfig` /
    `ttnn.BlackholeComputeKernelConfig` — it is read by duck-typing, so any
    object exposing `math_fidelity`, `fp32_dest_acc_en`, `math_approx_mode`,
    and `dst_full_sync_en` is accepted. (The arch-agnostic
    `ttnn.DeviceComputeKernelConfig` is the abstract variant base and has no
    Python constructor, which is why this is not type-checked.) Omitting it
    reproduces the op's defaults exactly: `fp32_dest_acc_en=True`,
    `math_approx_mode=False`, `dst_full_sync_en=False`, and `math_fidelity`
    = HiFi4 when both the activation and the weight are fp32, HiFi2 otherwise.
    See `conv2d_nhwc_program_descriptor.resolve_compute_kernel_config` for the
    HiFi4 clamp that protects the narrow-operand K-accumulator.
    """
    validate(
        input_tensor,
        weight_tensor,
        kernel_size,
        bias=bias,
        padding=padding,
        stride=stride,
        groups=groups,
        dilation=dilation,
    )

    N, H, W, _C_in = (int(d) for d in input_tensor.shape)
    C_out = int(weight_tensor.shape[-1])
    eff_k = dilation * (kernel_size - 1) + 1
    H_out = (H + 2 * padding - eff_k) // stride + 1
    W_out = (W + 2 * padding - eff_k) // stride + 1

    device = input_tensor.device()
    out_memory_config = memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG

    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape([N, H_out, W_out, C_out]),
        input_tensor.dtype,
        ttnn.ROW_MAJOR_LAYOUT,
        device,
        out_memory_config,
    )

    program_descriptor = create_program_descriptor(
        input_tensor,
        weight_tensor,
        bias,
        output_tensor,
        kernel_size=kernel_size,
        padding=padding,
        stride=stride,
        groups=groups,
        dilation=dilation,
        compute_kernel_config=compute_kernel_config,
    )

    tensors = [input_tensor, weight_tensor]
    if bias is not None:
        tensors.append(bias)
    tensors.append(output_tensor)  # output MUST be last
    return ttnn.generic_op(tensors, program_descriptor)
