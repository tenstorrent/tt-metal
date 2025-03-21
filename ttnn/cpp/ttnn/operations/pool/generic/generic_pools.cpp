// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "generic_pools.hpp"

#include <tt-metalium/buffer_constants.hpp>
#include "ttnn/operations/conv/conv2d/conv2d_utils.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/sliding_window/halo/halo.hpp"
#include "ttnn/operations/sliding_window/sliding_window.hpp"
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/math.hpp>

#include <limits>

namespace ttnn {
namespace operations::pool {

namespace {

// Return a single bf16 init value for the pool type in u32 (packed in the least 16 bits)
uint32_t get_bf16_pool_init_value(Pool2DType pool_type) {
    float value;
    switch (pool_type) {
        case Pool2DType::MAX_POOL2D: value = -std::numeric_limits<float>::infinity(); break;
    }
    return bfloat16(value).to_packed();
}

}  // namespace

template <Pool2DType pool_type>
Tensor Pool2DOp<pool_type>::invoke(
    QueueId queue_id,
    const Tensor& input_tensor,
    uint32_t batch_size,
    uint32_t input_h,
    uint32_t input_w,
    uint32_t channels,
    std::array<uint32_t, 2> kernel_size,
    std::array<uint32_t, 2> stride,
    std::array<uint32_t, 2> padding,
    std::array<uint32_t, 2> dilation,
    const std::optional<const MemoryConfig>& memory_config,
    const std::optional<const TensorMemoryLayout> applied_shard_scheme,
    bool ceil_mode) {

    Conv2dConfig conv_config = Conv2dConfig();
    const uint32_t output_height =
        ((input_h - kernel_size[0] - ((kernel_size[0] - 1) * (dilation[0] - 1)) + 2 * padding[0]) / stride[0]) + 1;
    const uint32_t output_width =
        ((input_w - kernel_size[1] - ((kernel_size[0] - 1) * (dilation[0] - 1)) + 2 * padding[1]) / stride[1]) + 1;

    DeviceComputeKernelConfig compute_config = conv::get_conv_default_compute_kernel_config(input_tensor.device());

    const auto compute_grid_size = input_tensor.device()->compute_with_storage_grid_size();

    bool auto_shard = false;
    if (!input_tensor.is_sharded() && !conv_config.shard_layout.has_value()) {
        // In this case we deduce the shard layout.
        // Note: function calculate_L1_usage called by function below
        //       takes weight tensor estimation into consideration,
        //       which should be omitted in pool op.
        conv_config = conv::determine_conv_config_for_auto_shard(
            conv_config,
            false,  // 1x1 mm
            batch_size,
            channels,
            channels,
            output_height,
            output_width,
            0,  //weight width
            input_h,
            input_w,
            compute_grid_size,
            input_tensor.layout(),
            ttnn::is_tensor_on_device_or_multidevice(input_tensor) ? std::make_optional(input_tensor.memory_config())
                                                                   : std::nullopt,
            kernel_size,
            1,  //groups
            false, //bias enable
            compute_config);
        auto_shard = true;
    }


    ShardOrientation shard_orientation =
        conv_config.transpose_shards ? ShardOrientation::COL_MAJOR : ShardOrientation::ROW_MAJOR;

    auto [input_tensor_post_tm, parallel_config, output_parallel_config] = conv::shard_or_reshard_tensor_if_required(
        input_tensor.device(),
        input_tensor,
        conv_config,
        batch_size,
        output_height,
        output_width,
        channels,
        channels,
        false,  // 1x1 mm,
        auto_shard);

    auto [opt_conv_op_parallel_config, opt_conv_op_block_config, conv_out_memory_config] = conv::get_conv_configs(
        conv_config,
        compute_config,
        parallel_config,
        output_parallel_config,
        channels,
        channels,
        batch_size,
        output_height,
        output_width,
        kernel_size,
        compute_grid_size);

    bool input_is_on_device = ttnn::is_tensor_on_device_or_multidevice(input_tensor_post_tm);
    TT_ASSERT(input_is_on_device);

    // call halo op
    SlidingWindowConfig sliding_window_config = SlidingWindowConfig{
        .batch_size = batch_size,
        .input_hw = {input_h, input_w},
        .window_hw = {kernel_size[0], kernel_size[1]},
        .stride_hw = {stride[0], stride[1]},
        .pad_hw = {padding[0], padding[1]},
        .dilation_hw = {dilation[0], dilation[1]},
        .num_cores_nhw = opt_conv_op_parallel_config.num_cores_nhw,
        .core_range_set = input_tensor_post_tm.memory_config().shard_spec.value().grid,
        .snap_to_tile = true,
    };

    bool bypass_halo =
        (parallel_config.shard_scheme == TensorMemoryLayout::WIDTH_SHARDED &&
            sliding_window_config.pad_hw.first == 0 && sliding_window_config.pad_hw.second == 0);

    if (bypass_halo) {
        if (input_tensor_post_tm.layout() == Layout::TILE) {
            // Reshape is used as a workaround to an issue in to_layout mentioned here :
            // https://github.com/tenstorrent/tt-metal/issues/16330
            input_tensor_post_tm = ttnn::reshape(input_tensor_post_tm, input_tensor_post_tm.get_padded_shape());
            input_tensor_post_tm =
                ttnn::to_layout(input_tensor_post_tm, Layout::ROW_MAJOR, std::nullopt, std::nullopt, input_tensor_post_tm.device());
        }
        parallel_config = conv::determine_parallel_config(
                                            shard_layout,
                                            batch_size,
                                            channels,
                                            output_shape[1],
                                            output_shape[2],
                                            channels,
                                            input_tensor.device()->compute_with_storage_grid_size(),
                                            ShardOrientation::ROW_MAJOR,
                                            false,
                                            false,
                                            false);
        num_cores_nhw = conv::get_num_cores_nhw_from_parallel_config(parallel_config);
        num_cores_c = conv::get_num_cores_channels_from_parallel_config(parallel_config);
        auto sharded_mem_config = conv::create_sharded_memory_config_from_parallel_config(input_tensor_sharded.get_padded_shape(), parallel_config, is_in_tiled ? tt::constants::TILE_HEIGHT : 1);
        input_tensor_sharded = ttnn::to_memory_config(input_tensor_sharded, sharded_mem_config, std::nullopt); // this converts interleaved to sharded
        out_memory_config = input_tensor_sharded.memory_config();
    } else {
        // Call the halo uop
        Tensor halo_output = ttnn::halo(
            queue_id,
            input_tensor_post_tm,
            sliding_window_config,
            0,
            false,
            parallel_config.shard_orientation == ShardOrientation::COL_MAJOR,
            0,
            input_tensor_post_tm.memory_config(),
            false);

        if (conv_config.deallocate_activation) {
            input_tensor_post_tm.deallocate(/*force*/ true);
        }

        input_tensor_post_tm = std::move(halo_output);

        if (conv_config.reallocate_halo_output) {
            input_tensor_post_tm = ttnn::move(input_tensor_post_tm);
        }
    }


    auto output_tensor = ttnn::prim::pool2d(
        queue_id,
        input_tensor_post_tm,
        sliding_window_config,
        pool_type,
        DataType::BFLOAT16,      // input_tensor.dtype(), // currently only bfp16 output is supported
        input_tensor_post_tm.memory_config());

    if (memory_config.has_value() && memory_config.value() != out_memory_config) {
        output_tensor = ttnn::to_memory_config(output_tensor, memory_config.value(), std::nullopt);
    }

    return output_tensor;
}

template class Pool2DOp<Pool2DType::MAX_POOL2D>;

}  // namespace operations::pool
}  // namespace ttnn
