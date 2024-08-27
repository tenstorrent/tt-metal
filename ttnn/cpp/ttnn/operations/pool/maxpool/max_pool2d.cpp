// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "max_pool2d.hpp"

#include "ttnn/operations/conv/conv2d/conv2d.hpp"
#include "ttnn/operations/sliding_window/sliding_window.hpp"
#include "tt_metal/common/math.hpp"
#include "ttnn/common/constants.hpp"

namespace ttnn {
namespace operations::pool {

template<typename T>
Tensor MaxPoolNewOp::invoke(uint8_t queue_id, const Tensor& input_tensor, uint32_t batch_size, uint32_t input_h, uint32_t input_w, uint32_t channels, std::array<uint32_t, 2> kernel_size, std::array<uint32_t, 2> stride, std::array<uint32_t, 2> padding, std::array<uint32_t, 2> dilation, T* device) {

    sliding_window::SlidingWindowConfig sliding_window_config = sliding_window::SlidingWindowConfig(
                                                                    batch_size,
                                                                    input_h, input_w,
                                                                    kernel_size.at(0), kernel_size.at(1),
                                                                    stride.at(0), stride.at(1),
                                                                    padding.at(0), padding.at(1),
                                                                    dilation.at(0), dilation.at(1));
    auto output_shape = sliding_window_config.get_output_shape();
    auto input_tensor_sharded = input_tensor;

    // maxpool output is row major
    bool is_out_tiled = false;
    bool is_in_tiled = input_tensor.dtype() == DataType::BFLOAT8_B; // input tiled for bfp8_b

    sliding_window::ParallelConfig parallel_config;
    MemoryConfig memory_config = input_tensor_sharded.memory_config();
    uint32_t num_cores_nhw = 0;

    if (!memory_config.shard_spec.has_value()) {
        // Input is not sharded. Perform sharding.
        parallel_config = conv::conv2d::determine_parallel_config(
                                            TensorMemoryLayout::HEIGHT_SHARDED,
                                            batch_size,
                                            0,          // in_channels -- not used
                                            output_shape[1],
                                            output_shape[2],
                                            0,          // out_channels -- not used
                                            device,
                                            ShardOrientation::ROW_MAJOR,
                                            false);
        num_cores_nhw = conv::conv2d::get_num_cores_nhw_from_parallel_config(parallel_config);
        auto sharded_mem_config = conv::conv2d::create_sharded_memory_config_from_parallel_config(input_tensor_sharded.shape(), parallel_config, is_in_tiled ? tt::constants::TILE_HEIGHT : 1);
        input_tensor_sharded = ttnn::to_memory_config(input_tensor_sharded, sharded_mem_config, std::nullopt);
        memory_config = input_tensor_sharded.memory_config();
    } else {
        // input is already sharded, use it as is
        const auto shard_grid = memory_config.shard_spec.value().grid;
        const auto shard_scheme = memory_config.memory_layout;
        const auto shard_orientation = memory_config.shard_spec.value().orientation;
        TT_FATAL(shard_scheme == TensorMemoryLayout::HEIGHT_SHARDED, "Only height sharded tensors are supported.");
        TT_FATAL(shard_orientation == ShardOrientation::ROW_MAJOR, "Only row major orientation is supported.");
        parallel_config.grid = shard_grid;
        parallel_config.shard_scheme = shard_scheme;
        parallel_config.shard_orientation = shard_orientation;
        num_cores_nhw = conv::conv2d::get_num_cores_nhw_from_parallel_config(parallel_config);
    }
    // update the shard spec to match the output shape
    auto shard_spec = memory_config.shard_spec.value();
    uint32_t output_shard_width_padded = input_tensor.dtype() == DataType::BFLOAT8_B ? tt::round_up(output_shape[3], tt::constants::TILE_WIDTH) : tt::round_up(output_shape[3] * tt::datum_size(tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype())), tt::constants::TILE_WIDTH);
    uint32_t output_nhw = output_shape[0] * output_shape[1] * output_shape[2];
    uint32_t output_nhw_padded = tt::round_up(output_nhw, num_cores_nhw * (is_out_tiled ? tt::constants::TILE_HEIGHT : 1));
    uint32_t output_shard_height_padded = output_nhw_padded / num_cores_nhw;
    log_debug(tt::LogOp, "output_nhw: {}, output_nhw_padded: {}, output_shard_height_padded: {}, output_shard_width_padded: {}", output_nhw, output_nhw_padded, output_shard_height_padded, output_shard_width_padded);
    memory_config.shard_spec = ShardSpec{shard_spec.grid, {output_shard_height_padded, output_shard_width_padded}, ShardOrientation::ROW_MAJOR, false};

    sliding_window_config = sliding_window::SlidingWindowConfig(
                                            batch_size,
                                            input_h,
                                            input_w,
                                            kernel_size.at(0),
                                            kernel_size.at(1),
                                            stride.at(0),
                                            stride.at(1),
                                            padding.at(0),
                                            padding.at(1),
                                            dilation.at(0),
                                            dilation.at(1),
                                            num_cores_nhw,
                                            parallel_config.grid,
                                            false);
    // call the halo uop
    uint32_t neg_inf_pad_val = 0xf7ff;
    auto haloed_tensor = ttnn::halo(
        queue_id,
        input_tensor_sharded,
        sliding_window_config,
        neg_inf_pad_val,
        false,
        parallel_config.shard_orientation == ShardOrientation::COL_MAJOR,
        0,
        input_tensor_sharded.memory_config(),
        is_out_tiled);

    return ttnn::prim::max_pool_new(
        queue_id,
        {haloed_tensor},
        {
            sliding_window_config,
            DataType::BFLOAT16,      // input_tensor.dtype(), // currently only bfp16 output is supported
            memory_config});
}

// device template specializations
template Tensor MaxPoolNewOp::invoke<Device>(uint8_t queue_id, const Tensor& input_tensor, uint32_t batch_size, uint32_t input_h, uint32_t input_w, uint32_t channels, std::array<uint32_t, 2> kernel_size, std::array<uint32_t, 2> stride, std::array<uint32_t, 2> padding, std::array<uint32_t, 2> dilation, Device* device);
template Tensor MaxPoolNewOp::invoke<DeviceMesh>(uint8_t queue_id, const Tensor& input_tensor, uint32_t batch_size, uint32_t input_h, uint32_t input_w, uint32_t channels, std::array<uint32_t, 2> kernel_size, std::array<uint32_t, 2> stride, std::array<uint32_t, 2> padding, std::array<uint32_t, 2> dilation, DeviceMesh* device);

}  // namespace operations::pool
}  // namespace ttnn
