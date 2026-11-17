# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""conv2d_3x3: dense 3x3 convolution of one image on the flattened channels-last contract.

    input  [1, 1, H*W, C_in]           (row h*W + w = one pixel)
    output [1, 1, H_out*W_out, C_out]  (always TILE, dtype = input dtype)

One device program per call: im2col gather (reader) -> tilize + K-blocked matmul with fp32
L1 accumulation (+ row-broadcast bias, + ReLU on the final value) (compute) -> output tiles
(writer). A TILE input is untilized in-kernel into the reader's halo slab first. The weights come from
prepare_conv2d_weights (rows ordered (kh, kw, c), each kh block padded to a tile multiple).

Placement: a DRAM-interleaved input gives a DRAM-interleaved output computed on the whole grid. An L1
HEIGHT_SHARDED input gives an L1 HEIGHT_SHARDED output on the input's shard grid, shard
[roundup32(ceil(M / n_cores)), roundup32(C_out)], packed in place. An L1 BLOCK_SHARDED input (C_in split
over the grid's columns) gives an L1 BLOCK_SHARDED output on the same grid, shard
[roundup32(ceil(M / n_rows)), roundup32(ceil(C_out / n_cols))]; the column partial sums are combined
across each grid row on device.
"""

from __future__ import annotations

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .conv2d_3x3_program_descriptor import (
    Conv3x3Geometry,
    block_grid_shape,
    create_program_descriptor,
    l1_cb_budget_bytes,
    m_split,
    select_blocking,
)

TILE = 32
_ML = ttnn.TensorMemoryLayout

# The gather copies a pixel's C_in real channels L1->L1 and leaves its padding lanes at zero; an L1
# NoC transfer moves whole 16 B words, so C_in must fill them (8 bf16 channels).
C_IN_GRANULE = 8


# ---------------------------------------------------------------------------
# Geometry shared by prepare_* and validate()
# ---------------------------------------------------------------------------


def _round_up(value, multiple):
    return -(-value // multiple) * multiple


def output_hw(input_height, input_width, *, padding=0, stride=1, dilation=1):
    eff_k = 2 * dilation + 1
    return (
        (input_height + 2 * padding - eff_k) // stride + 1,
        (input_width + 2 * padding - eff_k) // stride + 1,
    )


def _shard_width(memory_config):
    spec = memory_config.shard_spec
    if spec is None:
        raise ValueError(f"conv2d_3x3: {memory_config.memory_layout} memory config carries no shard_spec")
    return int(spec.shape[1])


def per_pixel_channels(c_in, input_layout, input_memory_config):
    """C_in_p: the per-pixel channel width the reader moves (one value for prepare_* and validate)."""
    memory_layout = input_memory_config.memory_layout
    if memory_layout == _ML.BLOCK_SHARDED:
        return _shard_width(input_memory_config)
    if input_layout == ttnn.TILE_LAYOUT:
        return _round_up(c_in, TILE)
    if memory_layout == _ML.HEIGHT_SHARDED:
        return _shard_width(input_memory_config)
    # ROW_MAJOR interleaved: one slab row is one DRAM-aligned page, so every page read lands aligned.
    return _round_up(c_in, ttnn.get_dram_alignment() // 2)


def channel_slices(c_in, input_layout, input_memory_config):
    """Channel slices K is split into: one per BLOCK shard column, else 1."""
    if input_memory_config.memory_layout != _ML.BLOCK_SHARDED:
        return 1
    return -(-c_in // per_pixel_channels(c_in, input_layout, input_memory_config))


def padded_k(c_in, input_layout, input_memory_config):
    """K_p: im2col width. Each kh block (3 taps x C_in_p) is padded to a tile multiple on its own; a
    BLOCK-sharded input repeats that per channel slice."""
    c_in_p = per_pixel_channels(c_in, input_layout, input_memory_config)
    return channel_slices(c_in, input_layout, input_memory_config) * 3 * _round_up(3 * c_in_p, TILE)


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS — inputs = ((H, W, C_in), C_out, padding, stride, dilation)
# ---------------------------------------------------------------------------


def tag_channel_alignment(inputs, axes):
    c_in_aligned = inputs[0][2] % TILE == 0
    c_out_aligned = inputs[1] % TILE == 0
    if c_in_aligned and c_out_aligned:
        return "tile_aligned"
    if c_out_aligned:
        return "c_in_unaligned"
    if c_in_aligned:
        return "c_out_unaligned"
    return "both_unaligned"


def tag_hw_alignment(inputs, axes):
    (H, W, _), _, padding, stride, dilation = inputs
    h_out, w_out = output_hw(H, W, padding=padding, stride=stride, dilation=dilation)
    return "hw_aligned" if (H * W) % TILE == 0 and (h_out * w_out) % TILE == 0 else "hw_non_aligned"


def tag_stride_mode(inputs, axes):
    return "unit" if inputs[3] == 1 else "strided"


def tag_dilation_mode(inputs, axes):
    return "unit" if inputs[4] == 1 else "dilated"


def tag_padding_mode(inputs, axes):
    padding, dilation = inputs[2], inputs[4]
    if padding == 0:
        return "valid"
    if padding == dilation:
        return "same"
    return "other"


INPUT_TAGGERS = {
    "channel_alignment": tag_channel_alignment,
    "hw_alignment": tag_hw_alignment,
    "stride_mode": tag_stride_mode,
    "dilation_mode": tag_dilation_mode,
    "padding_mode": tag_padding_mode,
}


# ---------------------------------------------------------------------------
# 2. SUPPORTED
# ---------------------------------------------------------------------------

SUPPORTED = {
    "dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
    "weight_dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
    "layout": [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT],
    "memory_layout": [_ML.INTERLEAVED, _ML.HEIGHT_SHARDED, _ML.BLOCK_SHARDED],
    "fp32_dest_acc_en": [True],
    "bias_mode": ["with_bias", "no_bias"],
    "activation": ["none", "relu"],
    # C_out off the tile grid needs no kernel work: prepared weight and bias columns past C_out are
    # zero tile padding, so the output's padding columns come out zero.
    "channel_alignment": ["tile_aligned", "c_in_unaligned", "c_out_unaligned", "both_unaligned"],
    "hw_alignment": ["hw_aligned"],
    "stride_mode": ["unit", "strided"],
    "dilation_mode": ["unit", "dilated"],
    "padding_mode": ["valid", "same"],
}


# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------

EXCLUSIONS = []


PROPERTIES = {
    "multi_core": {"value": True, "source": "declared"},
    "math_fidelity": {"value": ["LoFi", "HiFi2", "HiFi3", "HiFi4"], "source": "declared"},
}


def default_compute_kernel_config():
    return ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        math_approx_mode=False,
    )


# ---------------------------------------------------------------------------
# Support gate (shared by validate() and prepare_*)
# ---------------------------------------------------------------------------


def _check_supported(axes, op_name):
    """SUPPORTED per axis, then EXCLUSIONS. `axes` may be partial (prepare_* cannot see every axis);
    an axis given as a set of candidate values is refused only when no candidate is supported."""
    for axis, value in axes.items():
        allowed = SUPPORTED[axis]
        candidates = value if isinstance(value, frozenset) else (value,)
        if not any(c in allowed for c in candidates):
            raise UnsupportedAxisValue(f"{op_name}: {axis}={value!r} not in SUPPORTED {allowed}")
    for exc in EXCLUSIONS:
        if all(k in axes and not isinstance(axes[k], frozenset) and axes[k] == v for k, v in exc.items()):
            raise ExcludedCell(f"{op_name}: unsupported combination (refinement candidate): {exc}")


def _prepare_axes(*, input_dtype, input_layout, weight_dtype, input_memory_config, stride, padding, dilation):
    shape_free_inputs = ((0, 0, 0), 0, padding, stride, dilation)
    return {
        "dtype": input_dtype,
        "layout": input_layout,
        "weight_dtype": weight_dtype,
        "memory_layout": input_memory_config.memory_layout,
        "stride_mode": tag_stride_mode(shape_free_inputs, {}),
        "dilation_mode": tag_dilation_mode(shape_free_inputs, {}),
        "padding_mode": tag_padding_mode(shape_free_inputs, {}),
    }


# ---------------------------------------------------------------------------
# prepare_conv2d_weights / prepare_conv2d_bias (model-init time; the hot path does none of this)
# ---------------------------------------------------------------------------


def prepare_conv2d_weights(
    torch_weight,
    *,
    stride=1,
    padding=0,
    dilation=1,
    input_dtype,
    input_layout,
    weight_dtype,
    input_memory_config,
    device,
):
    """[C_out, C_in, 3, 3] -> TILE [1, 1, K_p, C_out]; rows (slice, kh, kw, c), c fastest, each kh block
    of 3*C_in_p rows zero-padded to a multiple of 32. A BLOCK-sharded input has one slice per shard
    column (C_in_p = shard width); every other placement has one."""
    if torch_weight.dim() != 4 or tuple(torch_weight.shape[2:]) != (3, 3):
        raise ValueError(f"conv2d_3x3: weight must be [C_out, C_in, 3, 3], got {list(torch_weight.shape)}")
    c_out, c_in = int(torch_weight.shape[0]), int(torch_weight.shape[1])
    _check_c_in_granule(c_in)
    axes = _prepare_axes(
        input_dtype=input_dtype,
        input_layout=input_layout,
        weight_dtype=weight_dtype,
        input_memory_config=input_memory_config,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )
    axes["channel_alignment"] = tag_channel_alignment(((0, 0, c_in), c_out, padding, stride, dilation), {})
    _check_supported(axes, "prepare_conv2d_weights")

    import torch

    c_in_p = per_pixel_channels(c_in, input_layout, input_memory_config)
    slices = channel_slices(c_in, input_layout, input_memory_config)
    kh_rows = _round_up(3 * c_in_p, TILE)
    w = torch_weight.permute(2, 3, 1, 0)  # [kh, kw, c, C_out]
    w = torch.nn.functional.pad(w, (0, 0, 0, slices * c_in_p - c_in))
    w = w.reshape(3, 3, slices, c_in_p, c_out).permute(2, 0, 1, 3, 4)  # [slice, kh, kw, c, C_out]
    w = w.reshape(slices, 3, 3 * c_in_p, c_out)
    w = torch.nn.functional.pad(w, (0, 0, 0, kh_rows - 3 * c_in_p))
    w = w.reshape(1, 1, slices * 3 * kh_rows, c_out)
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
    stride=1,
    padding=0,
    dilation=1,
    input_dtype,
    input_layout,
    weight_dtype,
    input_memory_config,
    device,
):
    """[C_out] -> TILE [1, 1, 1, C_out] (tile row 0 holds the bias), dtype = weight_dtype."""
    if torch_bias.dim() != 1:
        raise ValueError(f"conv2d_3x3: bias must be [C_out], got {list(torch_bias.shape)}")
    c_out = int(torch_bias.shape[0])
    axes = _prepare_axes(
        input_dtype=input_dtype,
        input_layout=input_layout,
        weight_dtype=weight_dtype,
        input_memory_config=input_memory_config,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )
    # C_in is not visible here: refuse only when no C_in alignment could make the cell supported.
    if c_out % TILE == 0:
        axes["channel_alignment"] = frozenset({"tile_aligned", "c_in_unaligned"})
    else:
        axes["channel_alignment"] = frozenset({"c_out_unaligned", "both_unaligned"})
    _check_supported(axes, "prepare_conv2d_bias")

    return ttnn.from_torch(
        torch_bias.reshape(1, 1, 1, c_out),
        dtype=weight_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def _check_c_in_granule(c_in):
    if c_in % C_IN_GRANULE != 0:
        raise ValueError(
            f"conv2d_3x3: C_in={c_in} must be a multiple of {C_IN_GRANULE} (pad input channels with zeros on the host)"
        )


def _check_height_shard(input_tensor):
    memory_config = input_tensor.memory_config()
    if memory_config.memory_layout != _ML.HEIGHT_SHARDED:
        return
    if memory_config.buffer_type != ttnn.BufferType.L1:
        raise ValueError("conv2d_3x3: a HEIGHT_SHARDED input must be sharded in L1")
    spec = memory_config.shard_spec
    if spec is None:
        raise ValueError("conv2d_3x3: HEIGHT_SHARDED input carries no shard_spec")
    padded_width = int(input_tensor.padded_shape[-1])
    if int(spec.shape[1]) != padded_width:
        raise ValueError(
            f"conv2d_3x3: HEIGHT shard width {spec.shape[1]} must be the full padded row width {padded_width}"
        )


def _check_block_shard(input_tensor):
    memory_config = input_tensor.memory_config()
    if memory_config.memory_layout != _ML.BLOCK_SHARDED:
        return
    if memory_config.buffer_type != ttnn.BufferType.L1:
        raise ValueError("conv2d_3x3: a BLOCK_SHARDED input must be sharded in L1")
    spec = memory_config.shard_spec
    if spec is None:
        raise ValueError("conv2d_3x3: BLOCK_SHARDED input carries no shard_spec")
    if len(spec.grid.ranges()) != 1:
        raise ValueError("conv2d_3x3: a BLOCK_SHARDED input's shard grid must be one rectangle")
    # Every grid column must hold a channel slice: the K split runs over exactly the grid's columns.
    _, grid_cols = block_grid_shape(spec)
    c_in = int(input_tensor.shape[-1])
    slices = channel_slices(c_in, input_tensor.layout, memory_config)
    if slices != grid_cols:
        raise ValueError(
            f"conv2d_3x3: BLOCK shard width {spec.shape[1]} splits C_in={c_in} into {slices} slices, "
            f"but the shard grid has {grid_cols} columns"
        )


def _check_contract(
    input_tensor, weight_tensor, bias, *, input_height, input_width, padding, stride, dilation, activation
):
    shape = list(input_tensor.shape)
    if len(shape) != 4 or shape[0] != 1 or shape[1] != 1 or shape[2] != input_height * input_width:
        raise ValueError(
            f"conv2d_3x3: input must be [1, 1, input_height*input_width, C_in] = "
            f"[1, 1, {input_height * input_width}, C_in], got {shape}"
        )
    if stride not in (1, 2):
        raise ValueError(f"conv2d_3x3: stride must be 1 or 2, got {stride}")
    if padding < 0:
        raise ValueError(f"conv2d_3x3: padding must be >= 0, got {padding}")
    if dilation < 1:
        raise ValueError(f"conv2d_3x3: dilation must be >= 1, got {dilation}")
    if activation not in (None, "relu"):
        raise ValueError(f"conv2d_3x3: activation must be None or 'relu', got {activation!r}")
    h_out, w_out = output_hw(input_height, input_width, padding=padding, stride=stride, dilation=dilation)
    if h_out < 1 or w_out < 1:
        raise ValueError(f"conv2d_3x3: output extent {h_out}x{w_out} is empty")

    c_in = shape[3]
    _check_c_in_granule(c_in)
    _check_height_shard(input_tensor)
    _check_block_shard(input_tensor)
    w_shape = list(weight_tensor.shape)
    k_p = padded_k(c_in, input_tensor.layout, input_tensor.memory_config())
    if len(w_shape) != 4 or w_shape[0] != 1 or w_shape[1] != 1 or w_shape[2] != k_p:
        raise ValueError(
            f"conv2d_3x3: weight must be [1, 1, K_p={k_p}, C_out] for C_in={c_in} and this input "
            f"layout/memory config, got {w_shape}"
        )
    if weight_tensor.layout != ttnn.TILE_LAYOUT:
        raise ValueError("conv2d_3x3: weight must be TILE (use prepare_conv2d_weights)")
    c_out = w_shape[3]
    if bias is not None:
        if list(bias.shape) != [1, 1, 1, c_out] or bias.layout != ttnn.TILE_LAYOUT:
            raise ValueError(f"conv2d_3x3: bias must be TILE [1, 1, 1, {c_out}], got {list(bias.shape)}")
        # cb_bias pages are sized from the weight tile format (prepare_conv2d_bias emits weight_dtype).
        if bias.dtype != weight_tensor.dtype:
            raise ValueError(
                f"conv2d_3x3: bias dtype {bias.dtype} must match weight dtype {weight_tensor.dtype} "
                f"(use prepare_conv2d_bias with the same weight_dtype)"
            )


def validate(
    input_tensor,
    weight_tensor,
    *,
    input_height,
    input_width,
    bias=None,
    padding=0,
    stride=1,
    dilation=1,
    activation=None,
    compute_kernel_config=None,
):
    _check_contract(
        input_tensor,
        weight_tensor,
        bias,
        input_height=input_height,
        input_width=input_width,
        padding=padding,
        stride=stride,
        dilation=dilation,
        activation=activation,
    )
    config = compute_kernel_config or default_compute_kernel_config()
    axes = {
        "dtype": input_tensor.dtype,
        "weight_dtype": weight_tensor.dtype,
        "layout": input_tensor.layout,
        "memory_layout": input_tensor.memory_config().memory_layout,
        "fp32_dest_acc_en": bool(config.fp32_dest_acc_en),
        "bias_mode": "with_bias" if bias is not None else "no_bias",
        "activation": "none" if activation is None else activation,
    }
    tagger_inputs = (
        (input_height, input_width, int(input_tensor.shape[-1])),
        int(weight_tensor.shape[-1]),
        padding,
        stride,
        dilation,
    )
    for axis, tagger in INPUT_TAGGERS.items():
        axes[axis] = tagger(tagger_inputs, axes)
    _check_supported(axes, "conv2d_3x3")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def conv2d_3x3(
    input_tensor,
    weight_tensor,
    *,
    input_height,
    input_width,
    bias=None,
    padding=0,
    stride=1,
    dilation=1,
    activation=None,
    compute_kernel_config=None,
):
    validate(
        input_tensor,
        weight_tensor,
        input_height=input_height,
        input_width=input_width,
        bias=bias,
        padding=padding,
        stride=stride,
        dilation=dilation,
        activation=activation,
        compute_kernel_config=compute_kernel_config,
    )
    config = compute_kernel_config or default_compute_kernel_config()
    device = input_tensor.device()

    c_in = int(input_tensor.shape[-1])
    c_out = int(weight_tensor.shape[-1])
    c_in_p = per_pixel_channels(c_in, input_tensor.layout, input_tensor.memory_config())
    h_out, w_out = output_hw(input_height, input_width, padding=padding, stride=stride, dilation=dilation)
    input_memory_config = input_tensor.memory_config()
    block_sharded = input_memory_config.memory_layout == _ML.BLOCK_SHARDED

    geometry = Conv3x3Geometry(
        input_height=input_height,
        input_width=input_width,
        c_in_p=c_in_p,
        c_out=c_out,
        padding=padding,
        stride=stride,
        dilation=dilation,
        output_height=h_out,
        output_width=w_out,
        c_in=c_in,
        tile_input=input_tensor.layout == ttnn.TILE_LAYOUT,
        in0_tile_bytes=ttnn.tile_size(input_tensor.dtype),
        weight_tile_bytes=ttnn.tile_size(weight_tensor.dtype),
        out_tile_bytes=ttnn.tile_size(input_tensor.dtype),
        has_bias=bias is not None,
        sharded_output=input_memory_config.is_sharded(),
        k_cols=channel_slices(c_in, input_tensor.layout, input_memory_config),
    )
    if geometry.sharded_output:
        # Same shard grid and scheme as the input. HEIGHT: one M work index per shard, shard i on shard
        # core i. BLOCK: one M work index per grid row, its output columns split over the grid columns.
        in_spec = input_memory_config.shard_spec
        m_work_cores = block_grid_shape(in_spec)[0] if block_sharded else in_spec.grid.num_cores()
        work_grid = (m_work_cores, 1)
        core_m_tiles, _ = m_split(geometry, m_work_cores)
        n_tiles = geometry.out_n_tiles
        output_memory_config = ttnn.MemoryConfig(
            input_memory_config.memory_layout,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(in_spec.grid, [core_m_tiles * TILE, n_tiles * TILE], in_spec.orientation),
        )
        input_shard_pages = (
            (in_spec.shape[0] // TILE) * (in_spec.shape[1] // TILE)
            if input_tensor.layout == ttnn.TILE_LAYOUT
            else in_spec.shape[0]
        )
        resident_shard_bytes = (
            input_shard_pages * input_tensor.buffer_aligned_page_size()
            + core_m_tiles * n_tiles * geometry.out_tile_bytes
        )
    else:
        grid = device.compute_with_storage_grid_size()
        work_grid = (grid.x, grid.y)
        output_memory_config = ttnn.DRAM_MEMORY_CONFIG
        resident_shard_bytes = 0
    blocking = select_blocking(geometry, work_grid, l1_cb_budget_bytes(resident_shard_bytes))

    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, h_out * w_out, c_out]),
        input_tensor.dtype,
        ttnn.TILE_LAYOUT,
        device,
        output_memory_config,
    )
    descriptor = create_program_descriptor(
        input_tensor,
        weight_tensor,
        bias,
        output_tensor,
        geometry=geometry,
        blocking=blocking,
        compute_kernel_config=config,
        apply_relu=activation == "relu",
    )
    io_tensors = [input_tensor, weight_tensor] + ([bias] if bias is not None else []) + [output_tensor]
    return ttnn.generic_op(io_tensors, descriptor)
