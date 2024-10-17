// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "conv_transpose2d.hpp"
#include <sys/types.h>
#include <cstdint>
#include "common/bfloat16.hpp"

using namespace tt;
namespace ttnn {
namespace operations::conv {
using sliding_window::SlidingWindowConfig;
using sliding_window::ParallelConfig;

namespace conv_transpose2d {

template <typename T>
Tensor _transform_weights_for_conv_transpose2d(
    const Tensor& conv_weight_tensor) {
    auto in_w_shape = conv_weight_tensor.get_legacy_shape();
    auto dtype = conv_weight_tensor.dtype();
    // in_w_shape = {in_channels, out_channels, kernel_height, kernel_width}
    // out_w_shape = {out_channels, in_channels, kernel_height, kernel_width}
    //Flip kernel_height and kernel_width
    auto compute = [&in_w_shape, &dtype](const auto& input_buffer) {
        auto in_channels   = in_w_shape[0];
        auto out_channels  = in_w_shape[1];
        auto kernel_height = in_w_shape[2];
        auto kernel_width  = in_w_shape[3];
        ttnn::SimpleShape output_shape{out_channels, in_channels, kernel_height, kernel_width};
        auto output_buffer = owned_buffer::create<T>(output_shape.volume());


        for(auto in_kernel_height_index = 0; in_kernel_height_index < kernel_height; in_kernel_height_index++)
        {
            auto out_buffer_kh_index = kernel_height - in_kernel_height_index - 1;
            for(auto in_kernel_width_index = 0; in_kernel_width_index < kernel_width; in_kernel_width_index++)
            {
                auto out_buffer_kw_index = kernel_width - in_kernel_width_index - 1;
                for(auto in_channels_index = 0; in_channels_index < in_channels; in_channels_index++)
                {
                    for(auto out_channels_index = 0; out_channels_index < out_channels; out_channels_index++)
                    {
                        auto in_idx = in_channels_index * kernel_height * kernel_width * out_channels + out_channels_index * kernel_height * kernel_width + in_kernel_height_index * kernel_width + in_kernel_width_index;
                        auto out_idx = out_channels_index * in_channels * kernel_height * kernel_width + in_channels_index * kernel_height * kernel_width + out_buffer_kh_index * kernel_width + out_buffer_kw_index;
                        output_buffer[out_idx] = input_buffer[in_idx];
                    }
                }
            }
        }
        return Tensor(std::move(OwnedStorage{std::move(output_buffer)}), output_shape, dtype, Layout::ROW_MAJOR);
    };
    auto convert_tensor = [&compute](const auto& conv_weight_tensor) {
        return std::visit(
            [&compute](auto&& storage) -> Tensor {
                using StorageType = std::decay_t<decltype(storage)>;
                if constexpr (std::is_same_v<StorageType, OwnedStorage>) {
                    return compute(owned_buffer::get_as<T>(storage.buffer));
                } else if constexpr (std::is_same_v<StorageType, BorrowedStorage>) {
                    return compute(borrowed_buffer::get_as<T>(storage.buffer));
                } else {
                    TT_THROW("Unsupported storage type");
                }
            },
            conv_weight_tensor.get_storage());
    };

    return is_multi_device_tensor(conv_weight_tensor) ? transform(conv_weight_tensor, convert_tensor) : convert_tensor(conv_weight_tensor);
}


Tensor transform_weights_for_conv_transpose2d(const Tensor& conv_weight_tensor) {
            switch (conv_weight_tensor.get_dtype()) {
                case DataType::BFLOAT16:
                    return _transform_weights_for_conv_transpose2d<::bfloat16>(conv_weight_tensor);
                case DataType::FLOAT32:
                    return _transform_weights_for_conv_transpose2d<float>(conv_weight_tensor);
                case DataType::UINT32:
                    return _transform_weights_for_conv_transpose2d<uint32_t>(conv_weight_tensor);
                default: TT_THROW("Unsupported data type for transform_weights_for_conv_transpose2d",conv_weight_tensor.get_dtype());
            }
};

template<typename T>
std::tuple<ttnn::Tensor, uint32_t, uint32_t, ttnn::Tensor, std::optional<ttnn::Tensor>> conv_transpose2d(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& weight_tensor,
    T * device,
    uint32_t in_channels,
    uint32_t out_channels,
    uint32_t batch_size,
    uint32_t input_height,
    uint32_t input_width,
    std::array<uint32_t, 2> kernel_size,
    std::array<uint32_t, 2> stride,
    std::array<uint32_t, 2> padding,
    std::array<uint32_t, 2> output_padding,
    std::array<uint32_t, 2> dilation,
    uint32_t groups,
    std::optional<const ttnn::Tensor> bias_tensor,
    std::optional<const conv2d::Conv2dConfig> conv_config_)
    {
        conv2d::Conv2dConfig conv_config = conv_config_.value_or(conv2d::Conv2dConfig());

        //Inverse of sliding_window.get_output_shape()
        SlidingWindowConfig sliding_window_config = SlidingWindowConfig{
            .batch_size = batch_size,
            .input_hw = {input_height, input_width},
            .window_hw = kernel_size,
            .stride_hw = stride,
            .pad_hw = padding,
            .output_pad_hw = output_padding,
            .dilation_hw = {dilation[0], dilation[1]},
            .is_transpose = true
        };

        uint32_t output_height = (input_height - 1) * stride[0] - 2 * padding[0] + dilation[0] * (kernel_size[0] - 1) + output_padding[0] + 1;
        uint32_t output_width  = (input_width  - 1) * stride[1] - 2 * padding[1] + dilation[1] * (kernel_size[1] - 1) + output_padding[1] + 1;

        //Dimensions of Input to Conv u_op
        uint32_t full_input_height = output_height + dilation[0] * (kernel_size[0] - 1);
        uint32_t full_input_width  =  output_width + dilation[1] * (kernel_size[1] - 1);

        //Size of input after adding interleaved 0s.
        uint32_t strided_input_height = (input_height - 1) * stride[0] + 1;
        uint32_t strided_input_width  = (input_width -  1) * stride[1] + 1;

        uint32_t input_pad_top  = (full_input_height - strided_input_height)/2;
        uint32_t input_pad_bottom = full_input_height - strided_input_height - input_pad_top;

        uint32_t input_pad_left = (full_input_width - strided_input_width)/2;
        uint32_t input_pad_right = full_input_width - strided_input_width - input_pad_left;

        log_debug(LogOp, "Input : {}x{}", input_height, input_width);
        log_debug(LogOp, "Output : {}x{}", output_height, output_width);

        log_debug(LogOp, "Conv Op Input : {}x{}", full_input_height, full_input_width);
        log_debug(LogOp, "Strided Input : {}x{}", strided_input_height, strided_input_width);

        log_debug(LogOp, "Padding : ({},{}) ({},{})", input_pad_top, input_pad_bottom, input_pad_left, input_pad_right);


        DeviceComputeKernelConfig compute_kernel_config;
        switch (device->arch()) {
            case tt::ARCH::WORMHOLE_B0:
                compute_kernel_config = WormholeComputeKernelConfig(
                    {.math_fidelity = conv_config.math_fidelity,
                    .math_approx_mode = conv_config.math_approx_mode_enabled,
                    .fp32_dest_acc_en = conv_config.fp32_dest_acc_enabled,
                    .packer_l1_acc = conv_config.packer_l1_accum_enabled});
                break;

            case tt::ARCH::GRAYSKULL:
                compute_kernel_config = GrayskullComputeKernelConfig(
                    {.math_fidelity = conv_config.math_fidelity, .math_approx_mode = conv_config.math_approx_mode_enabled});
                break;

            case tt::ARCH::BLACKHOLE:
                compute_kernel_config = BlackholeComputeKernelConfig(
                    {.math_fidelity = conv_config.math_fidelity,
                    .math_approx_mode = conv_config.math_approx_mode_enabled,
                    .fp32_dest_acc_en = conv_config.fp32_dest_acc_enabled,
                    .packer_l1_acc = conv_config.packer_l1_accum_enabled});
                break;

            default:
                TT_ASSERT(false);
        }


        //Call Halo Transpose
        auto [input_tensor_post_tm, parallel_config, tensor_manipulated] = conv2d::shard_or_reshard_tensor_if_required(
            device, input_tensor, conv_config, batch_size, output_height, output_width, in_channels, out_channels, kernel_size, stride);

        sliding_window_config.num_cores_nhw = conv2d::get_num_cores_nhw_from_parallel_config(parallel_config);
        sliding_window_config.core_range_set = input_tensor_post_tm.memory_config().shard_spec.value().grid;

        if (tensor_manipulated) {
            if (conv_config.deallocate_activation) {
                ttnn::Tensor input_tensor_ = input_tensor;  // TODO: allow in place modification of inputs to the op
                input_tensor_.deallocate();
                // ttnn::operations::core::deallocate(input_tensor_);
            }
            conv_config.deallocate_activation = true;
        }

        auto halo_output = ttnn::halo(
            DefaultQueueId,
            input_tensor_post_tm,
            sliding_window_config,
            0,
            false,
            parallel_config.shard_orientation == ShardOrientation::COL_MAJOR,
            0,
            input_tensor_post_tm.memory_config());

        //Call Conv2d u_op with Stride = 1, Padding = 0.
        auto conv_out_memory_config = conv2d::create_sharded_memory_config_from_parallel_config(
        ttnn::Shape(std::array<uint32_t, 4>{1, 1, batch_size * output_height * output_width, tt::round_up(out_channels, 32)}),
        parallel_config,
        32);
        auto opt_conv_op_parallel_config = conv2d::determine_conv_op_parallel_config_from_conv_output_mem_config(
            conv_out_memory_config,
            conv2d::get_num_cores_nhw_from_parallel_config(parallel_config),
            conv2d::get_num_cores_channels_from_parallel_config(parallel_config)
        );
        auto opt_conv_op_block_config = conv2d::determine_per_core_conv_block_config(
            parallel_config,
            opt_conv_op_parallel_config,
            tt::round_up(in_channels, conv_config.input_channels_alignment),
            conv_config.act_block_h_override,
            conv_config.act_block_w_div,
            kernel_size[0],
            kernel_size[1],
            conv_config.fp32_dest_acc_enabled,
            conv_config.input_channels_alignment == 16
        );

        //TODO: Flip the Weights
        bool weight_is_on_device = ttnn::is_tensor_on_device_or_multidevice(weight_tensor);
        ttnn::Tensor weight_tensor_on_device = weight_tensor;
        std::optional<ttnn::Tensor> bias_tensor_on_device = bias_tensor;
        if (!weight_is_on_device) {
            // prepare weights in desired layout and move to device


            tie(weight_tensor_on_device, bias_tensor_on_device) = conv2d::prepare_conv_weights_biases_and_move_to_device(
                transform_weights_for_conv_transpose2d(weight_tensor),
                bias_tensor,
                conv_config.input_channels_alignment,
                conv_config.weights_dtype,
                opt_conv_op_block_config.act_block_w_ntiles,
                opt_conv_op_block_config.out_subblock_w_ntiles,
                parallel_config,
                device,
                groups,
                opt_conv_op_block_config.act_block_h_ntiles,
                input_width);
        }
        // call conv micro op
        auto conv_output = optimized_conv_new(
            halo_output,
            weight_tensor_on_device,
            bias_tensor_on_device,
            sliding_window_config,
            out_channels,
            groups,
            conv_config.output_layout == Layout::ROW_MAJOR,
            conv_config.activation == "relu",
            conv_config.math_fidelity,
            opt_conv_op_parallel_config,
            opt_conv_op_block_config,
            conv_out_memory_config,
            conv_config.dtype,
            {batch_size, input_height, input_width, in_channels},
            conv_config.input_channels_alignment == 16,
            compute_kernel_config,
            conv_config.enable_act_double_buffer,
            conv_config.enable_split_reader,
            conv_config.enable_subblock_padding);
        // ttnn::operations::core::deallocate(halo_output);
        return {conv_output, output_height, output_width, halo_output, bias_tensor_on_device};


    }

template std::tuple<ttnn::Tensor, uint32_t, uint32_t, ttnn::Tensor, std::optional<ttnn::Tensor>> conv_transpose2d<Device>(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& weight_tensor,
    Device * device,
    uint32_t in_channels,
    uint32_t out_channels,
    uint32_t batch_size,
    uint32_t input_height,
    uint32_t input_width,
    std::array<uint32_t, 2> kernel_size,
    std::array<uint32_t, 2> stride,
    std::array<uint32_t, 2> padding,
    std::array<uint32_t, 2> output_padding,
    std::array<uint32_t, 2> dilation,
    uint32_t groups,
    std::optional<const ttnn::Tensor> bias_tensor,
    std::optional<const conv2d::Conv2dConfig> conv_config_);

template std::tuple<ttnn::Tensor, uint32_t, uint32_t, ttnn::Tensor, std::optional<ttnn::Tensor>> conv_transpose2d<MeshDevice>(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& weight_tensor,
    MeshDevice * device,
    uint32_t in_channels,
    uint32_t out_channels,
    uint32_t batch_size,
    uint32_t input_height,
    uint32_t input_width,
    std::array<uint32_t, 2> kernel_size,
    std::array<uint32_t, 2> stride,
    std::array<uint32_t, 2> padding,
    std::array<uint32_t, 2> output_padding,
    std::array<uint32_t, 2> dilation,
    uint32_t groups,
    std::optional<const ttnn::Tensor> bias_tensor,
    std::optional<const conv2d::Conv2dConfig> conv_config_);

}
}
}
