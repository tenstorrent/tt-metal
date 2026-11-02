# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""conv2d_nhwc — 2D convolution on channels-last activations (registry model).

One ttnn.generic_op dispatch per call: im2col gather (reader) -> tilize ->
K-blocked matmul with packer-L1 accumulation -> bias -> untilize (compute) ->
DRAM write (writer), over a 2-D (output-pixel x output-channel) core split.
See op_design.md (Blocking Model) for the regime this implements (R0).
"""

from __future__ import annotations

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .conv2d_nhwc_program_descriptor import (
    conv_geometry,
    create_program_descriptor,
    select_blocking,
)

# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS — over the bundled INPUTS tuple
#    (input_shape, C_out, kernel_size, padding, stride, groups, dilation
#     [, memory_layout [, layout [, activation]]])
# ---------------------------------------------------------------------------


def tag_channel_alignment(inputs, axes):
    C_in = inputs[0][-1]
    C_out = inputs[1]
    in_ok = C_in % 32 == 0
    out_ok = C_out % 32 == 0
    if in_ok and out_ok:
        return "tile_aligned"
    if out_ok:
        return "c_in_unaligned"
    if in_ok:
        return "c_out_unaligned"
    return "both_unaligned"


def tag_stride_mode(inputs, axes):
    return "unit" if inputs[4] == 1 else "strided"


def tag_groups_mode(inputs, axes):
    C_in = inputs[0][-1]
    groups = inputs[5]
    if groups == 1:
        return "dense"
    if groups == C_in:
        return "depthwise"
    return "grouped"


def tag_dilation_mode(inputs, axes):
    return "unit" if inputs[6] == 1 else "dilated"


def tag_memory_layout(inputs, axes):
    return inputs[7] if len(inputs) > 7 else ttnn.TensorMemoryLayout.INTERLEAVED


def tag_layout(inputs, axes):
    return inputs[8] if len(inputs) > 8 else ttnn.ROW_MAJOR_LAYOUT


def tag_activation(inputs, axes):
    return inputs[9] if len(inputs) > 9 else "none"


INPUT_TAGGERS = {
    "channel_alignment": tag_channel_alignment,
    "stride_mode": tag_stride_mode,
    "groups_mode": tag_groups_mode,
    "dilation_mode": tag_dilation_mode,
    "memory_layout": tag_memory_layout,
    "layout": tag_layout,
    "activation": tag_activation,
}

# ---------------------------------------------------------------------------
# 2. SUPPORTED (Phase 0)
# ---------------------------------------------------------------------------

SUPPORTED = {
    "dtype": [ttnn.float32],
    "weight_dtype": [ttnn.float32],
    "layout": [ttnn.ROW_MAJOR_LAYOUT],
    "bias_mode": ["with_bias", "no_bias"],
    "activation": ["none"],
    "channel_alignment": ["tile_aligned"],
    "stride_mode": ["unit"],
    "groups_mode": ["dense"],
    "dilation_mode": ["unit"],
    "memory_layout": [ttnn.TensorMemoryLayout.INTERLEAVED],
}

# ---------------------------------------------------------------------------
# 3. EXCLUSIONS — none in Phase 0 (one-dtype rectangle: no mixed precision)
# ---------------------------------------------------------------------------

EXCLUSIONS = []

# Axes that prepare_* can see (no layout / activation / bias presence).
_PREPARE_AXES = (
    "dtype",
    "weight_dtype",
    "channel_alignment",
    "stride_mode",
    "groups_mode",
    "dilation_mode",
    "memory_layout",
)


def _check_axes(axes, axis_names, who):
    for axis in axis_names:
        allowed = SUPPORTED[axis]
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"{who}: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")
    for exc in EXCLUSIONS:
        if all(k in axes and axes[k] == v for k, v in exc.items()):
            raise ExcludedCell(f"{who}: unsupported combination (refinement candidate): {exc}")


def _canon_activation(activation):
    return "none" if activation is None else activation


def _spatial(input_tensor, input_spatial):
    shape = list(input_tensor.shape)
    C_in = shape[-1]
    if input_spatial is not None:
        N, H, W = (int(v) for v in input_spatial)
        if len(shape) != 4 or shape[0] != 1 or shape[1] != 1 or shape[2] != N * H * W:
            raise ValueError(
                f"conv2d_nhwc: input_spatial={input_spatial} inconsistent with flattened input shape {shape}"
            )
        return N, H, W, C_in
    if len(shape) != 4:
        raise ValueError(f"conv2d_nhwc: expected a 4-D [N,H,W,C] input, got {shape}")
    if input_tensor.layout == ttnn.TILE_LAYOUT:
        raise ValueError("conv2d_nhwc: a TILE activation must be passed flattened with input_spatial")
    return shape[0], shape[1], shape[2], C_in


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


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
    input_spatial=None,
    activation=None,
):
    N, H, W, C_in = _spatial(input_tensor, input_spatial)
    C_out = int(weight_tensor.shape[-1])
    inputs = ((N, H, W, C_in), C_out, kernel_size, padding, stride, groups, dilation)
    axes = {
        "dtype": input_tensor.dtype,
        "weight_dtype": weight_tensor.dtype,
        "bias_mode": "with_bias" if bias is not None else "no_bias",
    }
    for name, tagger in INPUT_TAGGERS.items():
        axes[name] = tagger(inputs, axes)
    # Real tensor / kwarg values override the test-side projections.
    axes["memory_layout"] = input_tensor.memory_config().memory_layout
    axes["layout"] = input_tensor.layout
    axes["activation"] = _canon_activation(activation)
    _check_axes(axes, list(SUPPORTED.keys()), "conv2d_nhwc")
    eff_k = dilation * (kernel_size - 1) + 1
    if H + 2 * padding < eff_k or W + 2 * padding < eff_k:
        raise ValueError("conv2d_nhwc: kernel larger than the padded input")
    if bias is not None and int(bias.shape[-1]) != C_out:
        raise ValueError("conv2d_nhwc: bias C_out does not match the weight")
    return N, H, W, C_in, C_out


def _validate_prepare(
    C_in, C_out, *, kernel_size, stride, padding, groups, dilation, input_dtype, weight_dtype, input_memory_config, who
):
    """Shared prepare-time gate. C_in may be None (bias prepare does not see it): a
    tile-aligned C_in placeholder is used, so only C_out drives channel_alignment."""
    if C_in is None:
        C_in = 32 * groups
    inputs = ((1, 1, 1, C_in), C_out, kernel_size, padding, stride, groups, dilation)
    axes = {"dtype": input_dtype, "weight_dtype": weight_dtype}
    for name, tagger in INPUT_TAGGERS.items():
        axes[name] = tagger(inputs, axes)
    axes["memory_layout"] = input_memory_config.memory_layout
    _check_axes(axes, _PREPARE_AXES, who)


# ---------------------------------------------------------------------------
# prepare_* — the weight / bias layout seam
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
    """[C_out, C_in/g, kH, kW] -> [1, 1, K_t*32, C_out] TILE, K order [kh][kw*tap_width+ci],
    each kernel-row band padded to whole tiles (band_tiles*32 rows)."""
    import torch

    C_out = int(torch_weight.shape[0])
    C_in = int(torch_weight.shape[1]) * groups
    _validate_prepare(
        C_in,
        C_out,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        groups=groups,
        dilation=dilation,
        input_dtype=input_dtype,
        weight_dtype=weight_dtype,
        input_memory_config=input_memory_config,
        who="prepare_conv2d_weights",
    )
    kH, kW = int(torch_weight.shape[2]), int(torch_weight.shape[3])
    geo = conv_geometry(C_in=C_in, kH=kH, kW=kW, groups=groups, elem_bytes=_elem_bytes(input_dtype))
    tap_width, band_tiles = geo["tap_width"], geo["band_tiles"]
    C_in_g = C_in // groups
    w = torch_weight.float().permute(2, 3, 1, 0)  # [kH, kW, C_in_g, C_out]
    if tap_width != C_in_g:
        w = torch.nn.functional.pad(w, (0, 0, 0, tap_width - C_in_g))
    w = w.reshape(kH, kW * tap_width, C_out)
    band_rows = band_tiles * 32
    if band_rows != kW * tap_width:
        w = torch.nn.functional.pad(w, (0, 0, 0, band_rows - kW * tap_width))
    w = w.reshape(1, 1, kH * band_rows, C_out).contiguous()
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
    """[C_out] -> [1, 1, 32, C_out] TILE with row 0 = bias, rows 1..31 zero."""
    import torch

    C_out = int(torch_bias.shape[0])
    _validate_prepare(
        None,
        C_out,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        groups=groups,
        dilation=dilation,
        input_dtype=input_dtype,
        weight_dtype=weight_dtype,
        input_memory_config=input_memory_config,
        who="prepare_conv2d_bias",
    )
    b = torch.zeros(1, 1, 32, C_out, dtype=torch.float32)
    b[0, 0, 0, :] = torch_bias.float()
    return ttnn.from_torch(
        b,
        dtype=weight_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _elem_bytes(dtype):
    return 4 if dtype == ttnn.float32 else 2


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
    input_spatial: tuple = None,
    activation: str = None,
) -> ttnn.Tensor:
    N, H, W, C_in, C_out = validate(
        input_tensor,
        weight_tensor,
        kernel_size,
        bias=bias,
        padding=padding,
        stride=stride,
        groups=groups,
        dilation=dilation,
        input_spatial=input_spatial,
        activation=activation,
    )
    device = input_tensor.device()
    grid = device.compute_with_storage_grid_size()
    plan = select_blocking(
        N=N,
        H=H,
        W=W,
        C_in=C_in,
        C_out=C_out,
        kH=kernel_size,
        kW=kernel_size,
        padding=padding,
        stride=stride,
        dilation=dilation,
        groups=groups,
        elem_bytes=_elem_bytes(input_tensor.dtype),
        num_cores=grid.x * grid.y,
        has_bias=bias is not None,
    )
    H_out, W_out = plan["H_out"], plan["W_out"]
    out_shape = [1, 1, N * H_out * W_out, C_out] if input_spatial is not None else [N, H_out, W_out, C_out]
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(out_shape),
        input_tensor.dtype,
        input_tensor.layout,
        device,
        ttnn.DRAM_MEMORY_CONFIG,
    )
    descriptor = create_program_descriptor(input_tensor, weight_tensor, bias, output_tensor, plan, grid)
    io = [input_tensor, weight_tensor] + ([bias] if bias is not None else []) + [output_tensor]
    return ttnn.generic_op(io, descriptor)
