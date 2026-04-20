"""
Consteval functions for the VAE decoder TTNN model.

These replicate EXACTLY the consteval patterns from the generated main.py code.
Each function produces the same output as its original main_const_eval_N counterpart.
"""

import ttnn


DRAM = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)


# ---------------------------------------------------------------------------
# prepare_conv_weights: wraps ttnn.prepare_conv_weights with the exact same
# parameters as the generated code.
# ---------------------------------------------------------------------------
def prepare_conv_weights(
    weight_tensor,
    device,
    in_channels,
    out_channels,
    batch_size,
    input_height,
    input_width,
    kernel_size,
    stride,
    padding,
    dilation,
    groups,
    act_block_h_override,
    slice_config=None,
):
    return ttnn.prepare_conv_weights(
        weight_tensor=weight_tensor,
        input_memory_config=DRAM,
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=in_channels,
        out_channels=out_channels,
        batch_size=batch_size,
        input_height=input_height,
        input_width=input_width,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        has_bias=True,
        groups=groups,
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=act_block_h_override,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=slice_config,
    )


# ---------------------------------------------------------------------------
# prepare_conv_bias: wraps the full reshape -> permute -> ROW_MAJOR ->
# from_device -> prepare_conv_bias pipeline.
# ---------------------------------------------------------------------------
def prepare_conv_bias(
    bias_tensor,
    device,
    channels,
    in_channels,
    out_channels,
    batch_size,
    input_height,
    input_width,
    kernel_size,
    stride,
    padding,
    dilation,
    groups,
    act_block_h_override,
):
    t = ttnn.to_device(bias_tensor, device=device, memory_config=DRAM)
    t = ttnn.to_layout(t, ttnn.Layout.TILE, None, memory_config=None)
    t_reshaped = ttnn.reshape(t, [1, channels, 1, 1], memory_config=DRAM)
    ttnn.deallocate(t, False)
    t_perm = ttnn.permute(t_reshaped, [0, 2, 3, 1], memory_config=DRAM, pad_value=0.0)
    ttnn.deallocate(t_reshaped, False)
    t_rm = ttnn.to_layout(t_perm, ttnn.Layout.ROW_MAJOR, None, memory_config=None)
    ttnn.deallocate(t_perm, False)
    t_host = ttnn.from_device(t_rm)
    ttnn.deallocate(t_rm, False)
    result = ttnn.prepare_conv_bias(
        bias_tensor=t_host,
        input_memory_config=DRAM,
        input_layout=ttnn.Layout.TILE,
        in_channels=in_channels,
        out_channels=out_channels,
        batch_size=batch_size,
        input_height=input_height,
        input_width=input_width,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=act_block_h_override,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
    )
    ttnn.deallocate(t_host, False)
    return result


# ---------------------------------------------------------------------------
# reshape_gn_weight: matches consteval_2 pattern
# to_device -> TILE -> reshape [1, C, 1, 1] -> permute [0,2,3,1] ->
# typecast F32 -> permute [0,3,1,2] -> reshape [1, 32, C/32, 1]
# ---------------------------------------------------------------------------
def reshape_gn_weight(tensor, device, channels):
    cpg = channels // 32
    t = ttnn.to_device(tensor, device=device, memory_config=DRAM)
    t = ttnn.to_layout(t, ttnn.Layout.TILE, None, memory_config=None)
    t_r = ttnn.reshape(t, [1, channels, 1, 1], memory_config=DRAM)
    ttnn.deallocate(t, False)
    t_p = ttnn.permute(t_r, [0, 2, 3, 1], memory_config=DRAM, pad_value=0.0)
    ttnn.deallocate(t_r, False)
    t_f32 = ttnn.typecast(t_p, ttnn.DataType.FLOAT32, memory_config=DRAM)
    ttnn.deallocate(t_p, False)
    t_p2 = ttnn.permute(t_f32, [0, 3, 1, 2], memory_config=DRAM, pad_value=0.0)
    ttnn.deallocate(t_f32, False)
    t_out = ttnn.reshape(t_p2, [1, 32, cpg, 1], memory_config=DRAM)
    ttnn.deallocate(t_p2, False)
    return t_out


# ---------------------------------------------------------------------------
# reshape_gn_weight_attn: matches consteval_63 pattern
# to_device -> TILE -> reshape [1, C, 1] -> permute [0, 2, 1] ->
# reshape [1, C] -> typecast F32 -> reshape [1, 1, C] ->
# permute [0, 2, 1] -> reshape [1, 32, C/32, 1]
# ---------------------------------------------------------------------------
def reshape_gn_weight_attn(tensor, device, channels):
    cpg = channels // 32
    t = ttnn.to_device(tensor, device=device, memory_config=DRAM)
    t = ttnn.to_layout(t, ttnn.Layout.TILE, None, memory_config=None)
    t_r1 = ttnn.reshape(t, [1, channels, 1], memory_config=DRAM)
    ttnn.deallocate(t, False)
    t_p1 = ttnn.permute(t_r1, [0, 2, 1], memory_config=DRAM, pad_value=0.0)
    ttnn.deallocate(t_r1, False)
    t_r2 = ttnn.reshape(t_p1, [1, channels], memory_config=DRAM)
    ttnn.deallocate(t_p1, False)
    t_f32 = ttnn.typecast(t_r2, ttnn.DataType.FLOAT32, memory_config=DRAM)
    ttnn.deallocate(t_r2, False)
    t_r3 = ttnn.reshape(t_f32, [1, 1, channels], memory_config=DRAM)
    ttnn.deallocate(t_f32, False)
    t_p2 = ttnn.permute(t_r3, [0, 2, 1], memory_config=DRAM, pad_value=0.0)
    ttnn.deallocate(t_r3, False)
    t_out = ttnn.reshape(t_p2, [1, 32, cpg, 1], memory_config=DRAM)
    ttnn.deallocate(t_p2, False)
    return t_out


# ---------------------------------------------------------------------------
# make_scalar: creates a scalar constant on device in the given dtype,
# reshaped to [1, 1, 1, 1]. Matches consteval_13, 91, 94, 103, 106, 124.
# ---------------------------------------------------------------------------
def make_scalar(device, value, dtype=ttnn.DataType.FLOAT32):
    t = ttnn.full(
        shape=ttnn.Shape([]),
        fill_value=value,
        dtype=dtype,
        layout=ttnn.Layout.TILE,
        device=device,
        memory_config=DRAM,
    )
    t_out = ttnn.reshape(t, [1, 1, 1, 1], memory_config=DRAM)
    ttnn.deallocate(t, False)
    return t_out


# ---------------------------------------------------------------------------
# make_ones_scalars: matches consteval_81 - produces var_3 [1,1,1,1] and
# var_4 [1,1], both BF16 1.0
# ---------------------------------------------------------------------------
def make_ones_scalars(device):
    t = ttnn.full(
        shape=ttnn.Shape([]),
        fill_value=1.0,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=device,
        memory_config=DRAM,
    )
    t_4d = ttnn.reshape(t, [1, 1, 1, 1], memory_config=DRAM)
    t_2d = ttnn.reshape(t, [1, 1], memory_config=DRAM)
    ttnn.deallocate(t, False)
    return t_4d, t_2d


# ---------------------------------------------------------------------------
# make_upsample_matrix: matches consteval_104, 119, 126
# Creates a [out_size, in_size] BF16 eq-matrix on device for nearest-neighbor
# upsampling via matmul.
# ---------------------------------------------------------------------------
def make_upsample_matrix(device, in_size, out_size):
    # Create integer indices [0 .. in_size-1]
    idx_int = ttnn.Tensor(
        list(range(in_size)),
        [in_size],
        ttnn.DataType.INT32,
        ttnn.Layout.TILE,
        device,
        memory_config=DRAM,
    )
    # Create half-step float indices [0.0, 0.5, 1.0, 1.5, ...]
    coords = [i * 0.5 for i in range(out_size)]
    coord_f = ttnn.Tensor(
        coords,
        [out_size],
        ttnn.DataType.FLOAT32,
        ttnn.Layout.TILE,
        device,
        memory_config=DRAM,
    )
    # floor -> int
    coord_floor = ttnn.floor(coord_f, memory_config=DRAM)
    ttnn.deallocate(coord_f, False)
    coord_int = ttnn.typecast(coord_floor, ttnn.DataType.INT32, memory_config=DRAM)
    ttnn.deallocate(coord_floor, False)
    # reshape to column [out_size, 1]
    coord_col = ttnn.reshape(coord_int, [out_size, 1], memory_config=DRAM)
    ttnn.deallocate(coord_int, False)
    # transpose to row [1, out_size]
    coord_row = ttnn.permute(coord_col, [1, 0], memory_config=DRAM, pad_value=0.0)
    ttnn.deallocate(coord_col, False)
    # reshape idx to column-ish [1, in_size] then transpose
    idx_2d = ttnn.reshape(idx_int, [1, in_size], memory_config=DRAM)
    ttnn.deallocate(idx_int, False)
    idx_col = ttnn.permute(idx_2d, [1, 0], memory_config=DRAM, pad_value=0.0)
    ttnn.deallocate(idx_2d, False)
    # eq broadcast: [1, out_size] vs [in_size, 1] -> [in_size, out_size]
    # but original does it as coord_row vs idx_col
    eq = ttnn.eq(coord_row, idx_col, dtype=ttnn.DataType.BFLOAT16, memory_config=DRAM)
    ttnn.deallocate(idx_col, False)
    ttnn.deallocate(coord_row, False)
    return eq
