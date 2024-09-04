# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import List, Union
from tt_lib import tensor
from tt_lib.utils import _nearest_32, _nearest_y
import torch
import ttnn
from loguru import logger


def conv(weight: List[Union[int, float]], conv_params, device, bias=None):
    """
    Returns a function that performs a Convolution.
    For bias, it calls bcast op with autoformatting
    """
    assert len(conv_params) == 10
    K, C, R, S, U, V, P_H, P_W, dilation, groups = [conv_params[i] for i in range(10)]
    # Hardcode block sizes
    act_block_h = 4
    act_block_w = 4
    weight_block_h = act_block_w
    weight_block_w = 4
    out_subblock_h = 4
    out_subblock_w = 2
    if dilation != 1 or groups != 1:
        return None
    weights_shape = [K, C, R, S]
    weights_channels_padded_shape = [_nearest_32(K), _nearest_y(C, 16), R, S]
    weight_untiled = ttnn.Tensor(weight, weights_shape, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT).pad(
        weights_channels_padded_shape, (0, 0, 0, 0), 0
    )
    weight_tiled_ = tensor.convert_conv_weight_tensor_to_tiled_layout(weight_untiled, weight_block_h, weight_block_w)
    weight_on_device = weight_tiled_.to(device)
    if bias is None:
        bias_on_device = None
    else:
        bias_shape = [1, 1, 1, K]
        bias_channels_padded_shape = [1, 1, 1, _nearest_32(K)]
        bias_ = ttnn.Tensor(bias, bias_shape, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT).pad(
            bias_channels_padded_shape, (0, 0, 0, 0), 0
        )
        bias_on_device = bias_.to(device)

    def conv_(activation):
        output = ttnn.operations.conv2d.conv_legacy(
            activation,
            weight_on_device,
            None,
            [R, S, U, V, P_H, P_W],
            act_block_h,
            act_block_w,
            weight_block_w,
            out_subblock_h,
            out_subblock_w,
            K,
            False,
        )

        assert output.storage_type() == ttnn.StorageType.DEVICE

        if bias_on_device is not None:
            output_plus_bias = ttnn.add(output, bias_on_device)
            if output_plus_bias.get_layout() != ttnn.ROW_MAJOR_LAYOUT:
                assert output_plus_bias.get_layout() == ttnn.TILE_LAYOUT
                assert output_plus_bias.storage_type() == ttnn.StorageType.DEVICE
                output_plus_bias = ttnn.untilize(output_plus_bias, memory_config=output_plus_bias.memory_config())
                assert output_plus_bias.get_layout() == ttnn.ROW_MAJOR_LAYOUT
            return output_plus_bias

        return output

    return conv_


def resnet50_1x1_conv_as_matmul(
    weight: List[Union[int, float]],
    conv_params,
    device,
    bias,
    matmul_config,
    fuse_relu=False,
    output_mem_config=None,
    weights_dtype=None,
    output_dtype=None,
    compute_kernel_config=None,
    untilize_out=False,
):
    """
    Returns a function that performs a Convolution. Bias is fused with matmul.
    """
    assert len(conv_params) == 10
    K, C, R, S, U, V, P_H, P_W, dilation, groups = [conv_params[i] for i in range(10)]

    assert R == S and R == 1 and P_H == P_W and P_H == 0 and U == V and U == 1
    assert dilation == 1 and groups == 1

    weights_shape = [K, C, R, S]

    assert C % 16 == 0
    assert K % 32 == 0

    weights_untiled_dtype = weights_dtype if weights_dtype != ttnn.bfloat8_b else ttnn.float32
    weight_untiled = ttnn.Tensor(weight, weights_shape, weights_untiled_dtype, ttnn.ROW_MAJOR_LAYOUT)
    # weight for matmul op
    weight_tiled_ = tensor.convert_conv_weight_tensor_to_tiled_layout(weight_untiled, 1, 1, output_dtype=weights_dtype)

    weight_on_device = weight_tiled_.to(device)
    bias_shape = [1, 1, 1, K]
    bias_channels_padded_shape = [1, 1, 32, K]
    bias = torch.nn.functional.pad(torch.Tensor(bias).reshape(bias_shape), (0, 0, 0, 31)).flatten().tolist()
    bias_ = ttnn.Tensor(bias, bias_channels_padded_shape, weights_dtype, ttnn.ROW_MAJOR_LAYOUT).to(ttnn.TILE_LAYOUT)
    bias_on_device = bias_.to(device)
    if isinstance(matmul_config, dict):
        matmul_program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=matmul_config["compute_with_storage_grid_size"],
            in0_block_w=matmul_config["in0_block_w"],
            out_subblock_h=matmul_config["out_subblock_h"],
            out_subblock_w=matmul_config["out_subblock_w"],
            per_core_M=matmul_config["per_core_M"],
            per_core_N=matmul_config["per_core_N"],
            transpose_mcast=False,
            fused_activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU) if fuse_relu else None,
        )
    else:
        matmul_program_config = matmul_config
        if fuse_relu:
            matmul_program_config.fused_activation = ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)

    def conv_(activation):
        # conv1x1 stride 1 padding 0, use matmul op
        output = ttnn.linear(
            activation,
            weight_on_device,
            bias=bias_on_device,
            program_config=matmul_program_config,
            memory_config=activation.memory_config() if output_mem_config is None else output_mem_config,
            dtype=output_dtype,
            compute_kernel_config=compute_kernel_config,
        )

        return output

    return conv_


def resnet50_optimized_conv(
    weight: List[Union[int, float]],
    conv_params,
    device,
    act_block_shape_hw,
    weight_block_shape_hw,
    outsubblock_shape_hw,
    out_block_shape_h,
    grid_size,
    per_core_out_matrix_h_ntiles,
    per_core_weight_matrix_w_ntiles,
    bias,
    fuse_relu=False,
    output_mem_config=None,
    input_tensor_shape=None,
    weights_dtype=None,
    output_dtype=None,
    math_fidelity=None,
    act_c_num_blocks=1,
    compute_kernel_config=None,
):
    """
    Returns a function that performs a Convolution. Bias is fused with conv.
    """
    assert len(conv_params) == 10
    K, C, R, S, U, V, P_H, P_W, dilation, groups = [conv_params[i] for i in range(10)]

    assert len(act_block_shape_hw) == 2
    assert len(weight_block_shape_hw) == 2
    assert len(outsubblock_shape_hw) == 2
    assert act_block_shape_hw[1] == weight_block_shape_hw[0]
    assert act_block_shape_hw[0] % 32 == 0
    assert act_block_shape_hw[1] % 32 == 0

    out_block_h = (int)(out_block_shape_h / 32)
    act_block_h = (int)(act_block_shape_hw[0] / 32)
    act_block_w = (int)(act_block_shape_hw[1] / 32)
    weight_block_h = act_block_w
    weight_block_w = (int)(weight_block_shape_hw[1] / 32)
    out_subblock_h = (int)(outsubblock_shape_hw[0] / 32)
    out_subblock_w = (int)(outsubblock_shape_hw[1] / 32)
    assert out_subblock_h * out_subblock_w <= 8

    assert dilation == 1 and groups == 1
    assert (
        weight_block_w == per_core_weight_matrix_w_ntiles
    ), "weight_block_w should be equal to per_core_weight_matrix_w_ntiles"
    assert out_block_h == per_core_out_matrix_h_ntiles, "out_block_h should be equal to per_core_out_matrix_h_ntiles"

    weights_shape = [K, C, R, S]
    weights_channels_padded_shape = [_nearest_32(K), _nearest_y(C, 16), R, S]
    weights_untiled_dtype = weights_dtype if weights_dtype != ttnn.bfloat8_b else ttnn.float32
    weight_untiled = ttnn.Tensor(weight, weights_shape, weights_untiled_dtype, ttnn.ROW_MAJOR_LAYOUT).pad(
        weights_channels_padded_shape, (0, 0, 0, 0), 0
    )
    act_block_w_equals_input_channels_x_filter_width = act_block_shape_hw[1] == (C * S)
    # for conv op, pad the weights to block shape
    if act_block_w_equals_input_channels_x_filter_width:
        weight_tiled_ = tensor.convert_conv_weight_tensor_to_tiled_layout(
            weight_untiled,
            weight_block_h,
            weight_block_w,
            output_dtype=weights_dtype,
        )
    else:
        if R == 1 and S == 1:
            assert C % act_block_shape_hw[1] == 0
        else:
            assert act_block_shape_hw[1] == C
        weight_tiled_ = tensor.convert_conv_weight_tensor_to_tiled_layout(
            weight_untiled,
            weight_block_h,
            weight_block_w,
            output_dtype=weights_dtype,
        )
    weight_on_device = weight_tiled_.to(device)
    bias_shape = [1, 1, 1, K]
    assert K % (weight_block_w * 32) == 0
    bias_channels_padded_shape = [1, 1, 32, _nearest_32(K)]
    bias = (
        torch.nn.functional.pad(torch.Tensor(bias).reshape(bias_shape), (0, _nearest_32(K) - K, 0, 31))
        .flatten()
        .tolist()
    )
    bias_ = ttnn.Tensor(bias, bias_channels_padded_shape, weights_dtype, ttnn.ROW_MAJOR_LAYOUT).to(ttnn.TILE_LAYOUT)
    bias_on_device = bias_.to(device)

    opt_conv_parall_conf = ttnn.operations.conv2d.OptimizedConvParallelizationConfig(
        grid_size=grid_size,
        num_cores_nhw=grid_size[0],
        per_core_out_matrix_height_ntiles=per_core_out_matrix_h_ntiles,
        per_core_out_matrix_width_ntiles=per_core_weight_matrix_w_ntiles,
    )
    opt_conv_block_conf = ttnn.operations.conv2d.OptimizedConvBlockConfig(
        act_block_h_ntiles=act_block_h,
        act_block_w_ntiles=act_block_w,
        out_subblock_h_ntiles=out_subblock_h,
        out_subblock_w_ntiles=out_subblock_w,
    )

    def conv_(activation):
        output = ttnn.operations.conv2d.optimized_conv(
            activation,
            weight_on_device,
            bias=bias_on_device,
            conv_reader_indices=None,
            conv_params=[R, S, U, V, P_H, P_W],
            output_channels=K,
            untilize_out=False,
            has_bias=True,
            fuse_relu=fuse_relu,
            math_fidelity=math_fidelity,
            parallelization_config=opt_conv_parall_conf,
            block_config=opt_conv_block_conf,
            extra_padding_for_32_B_alignment=0,
            memory_config=activation.memory_config() if output_mem_config is None else output_mem_config,
            dtype=output_dtype,
            input_tensor_shape=input_tensor_shape,
            compute_kernel_config=compute_kernel_config,
        )
        return output

    return conv_


def resnet50_first_conv(
    weight: List[Union[int, float]],
    conv_params,
    device,
    act_block_shape_hw,
    weight_block_shape_hw,
    outsubblock_shape_hw,
    out_block_shape_h,
    grid_size,
    per_core_out_matrix_h_ntiles,
    bias,
    padded_filter_window_width,
    fuse_relu=False,
    out_mem_config=None,
    weights_dtype=None,
    output_dtype=None,
    math_fidelity=None,
):
    """
    Returns a function that performs a Convolution. Bias is fused with conv.
    """
    assert len(conv_params) == 10
    K, C, R, S, U, V, P_H, P_W, dilation, groups = [conv_params[i] for i in range(10)]
    extra_padding_for_32B_alignment = 25

    assert len(act_block_shape_hw) == 2
    assert len(weight_block_shape_hw) == 2
    assert len(outsubblock_shape_hw) == 2
    assert act_block_shape_hw[1] == weight_block_shape_hw[0]
    assert act_block_shape_hw[0] % 32 == 0
    assert act_block_shape_hw[1] % 32 == 0

    out_block_h = (int)(out_block_shape_h / 32)
    act_block_h = (int)(act_block_shape_hw[0] / 32)
    act_block_w = (int)(act_block_shape_hw[1] / 32)
    weight_block_h = act_block_w
    weight_block_w = (int)(weight_block_shape_hw[1] / 32)
    out_subblock_h = (int)(outsubblock_shape_hw[0] / 32)
    out_subblock_w = (int)(outsubblock_shape_hw[1] / 32)
    assert out_subblock_h * out_subblock_w <= 8

    assert dilation == 1 and groups == 1

    weights_shape = [K, C, R, S]
    assert padded_filter_window_width >= S
    weights_channels_padded_shape = [
        _nearest_32(K),
        _nearest_y(C, 4),
        R,
        padded_filter_window_width,
    ]  # first conv channel padded to 4 only
    weights_untiled_dtype = weights_dtype if weights_dtype != ttnn.bfloat8_b else ttnn.float32
    weight_untiled = ttnn.Tensor(weight, weights_shape, weights_untiled_dtype, ttnn.ROW_MAJOR_LAYOUT).pad(
        weights_channels_padded_shape, (0, 0, 0, 0), 0
    )
    per_core_weight_matrix_w_ntiles = (int)(weights_channels_padded_shape[0] / 32)
    # for conv op, pad the weights to block shape
    weight_tiled_ = tensor.convert_conv_weight_tensor_to_special_padding_tiled_layout(
        weight_untiled, weight_block_h, weight_block_w, output_dtype=weights_dtype
    )
    weight_on_device = weight_tiled_.to(device)

    bias_shape = [1, 1, 1, K]
    assert K % (weight_block_w * 32) == 0
    bias_channels_padded_shape = [1, 1, 32, _nearest_32(K)]
    bias = (
        torch.nn.functional.pad(torch.Tensor(bias).reshape(bias_shape), (0, _nearest_32(K) - K, 0, 31))
        .flatten()
        .tolist()
    )
    bias_ = ttnn.Tensor(bias, bias_channels_padded_shape, weights_dtype, ttnn.ROW_MAJOR_LAYOUT).to(ttnn.TILE_LAYOUT)
    bias_on_device = bias_.to(device)

    # Resnet50 first conv is pre-padded on host
    P_H = 0
    P_W = 0

    opt_conv_parall_conf = ttnn.operations.conv2d.OptimizedConvParallelizationConfig(
        grid_size=grid_size,
        num_cores_nhw=grid_size[0],
        per_core_out_matrix_height_ntiles=per_core_out_matrix_h_ntiles,
        per_core_out_matrix_width_ntiles=per_core_weight_matrix_w_ntiles,
    )
    opt_conv_block_conf = ttnn.operations.conv2d.OptimizedConvBlockConfig(
        act_block_h_ntiles=act_block_h,
        act_block_w_ntiles=act_block_w,
        out_subblock_h_ntiles=out_subblock_h,
        out_subblock_w_ntiles=out_subblock_w,
    )

    def conv_(activation):
        output_plus_bias = ttnn.operations.conv2d.optimized_conv(
            activation,
            weight_on_device,
            bias_on_device,
            None,
            [R, padded_filter_window_width, U, V, P_H, P_W],
            K,
            False,
            True,
            fuse_relu,
            math_fidelity,
            opt_conv_parall_conf,
            opt_conv_block_conf,
            extra_padding_for_32B_alignment,
            out_mem_config,
            output_dtype,
        )
        return output_plus_bias

    return conv_


def resnet50_1x1_conv_s2_as_downsample_and_matmul(
    weight: List[Union[int, float]],
    conv_params,
    downsample_params,
    device,
    bias,
    matmul_config,
    out_sharded_mem_config,
    weights_dtype,
    output_dtype,
    compute_kernel_config,
):
    """
    Returns a function that performs a Convolution. Bias is fused with matmul.
    """
    assert len(conv_params) == 10
    K, C, R, S, U, V, P_H, P_W, dilation, groups = [conv_params[i] for i in range(10)]

    assert R == S and R == 1 and P_H == P_W and P_H == 0 and U == V and U == 2
    assert dilation == 1 and groups == 1

    weights_shape = [K, C, R, S]

    assert C % 16 == 0
    assert K % 32 == 0

    weights_untiled_dtype = weights_dtype if weights_dtype != ttnn.bfloat8_b else ttnn.float32
    weight_untiled = ttnn.Tensor(weight, weights_shape, weights_untiled_dtype, ttnn.ROW_MAJOR_LAYOUT)
    # weight for matmul op
    weight_tiled_ = tensor.convert_conv_weight_tensor_to_tiled_layout(weight_untiled, 1, 1, output_dtype=weights_dtype)

    weight_on_device = weight_tiled_.to(device)
    bias_shape = [1, 1, 1, K]
    bias_channels_padded_shape = [1, 1, 32, K]
    bias = torch.nn.functional.pad(torch.Tensor(bias).reshape(bias_shape), (0, 0, 0, 31)).flatten().tolist()
    bias_ = ttnn.Tensor(bias, bias_channels_padded_shape, weights_dtype, ttnn.ROW_MAJOR_LAYOUT).to(ttnn.TILE_LAYOUT)
    bias_on_device = bias_.to(device)
    if isinstance(matmul_config, dict):
        matmul_program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=matmul_config["compute_with_storage_grid_size"],
            in0_block_w=matmul_config["in0_block_w"],
            out_subblock_h=matmul_config["out_subblock_h"],
            out_subblock_w=matmul_config["out_subblock_w"],
            per_core_M=matmul_config["per_core_M"],
            per_core_N=matmul_config["per_core_N"],
            transpose_mcast=False,
            fused_activation=None,
        )
    else:
        matmul_program_config = matmul_config

    def conv_(activation):
        # downsample op
        output = ttnn.downsample(activation, downsample_params, dtype=output_dtype)
        output = ttnn.linear(
            output,
            weight_on_device,
            bias=bias_on_device,
            program_config=matmul_program_config,
            memory_config=out_sharded_mem_config,
            dtype=output_dtype,
            compute_kernel_config=compute_kernel_config,
        )

        return output

    return conv_
