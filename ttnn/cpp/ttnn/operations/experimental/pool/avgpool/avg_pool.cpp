// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/pool/avgpool/avg_pool.hpp"

#include "ttnn/operations/conv/conv2d/conv2d.hpp"
#include "ttnn/operations/experimental/pool/avgpool/device/avg_pool2d_device_op.hpp"
#include "ttnn/operations/sliding_window/sliding_window.hpp"

namespace ttnn::operations::experimental::pool {

Tensor AveragePool2DOp::invoke(
    const Tensor& input_tensor,
    uint32_t batch_size,
    uint32_t input_h,
    uint32_t input_w,
    uint32_t channels,
    const std::array<uint32_t, 2>& kernel_size,
    const std::array<uint32_t, 2>& stride,
    const std::array<uint32_t, 2>& padding,
    bool ceil_mode,
    bool count_include_pad,
    const std::optional<MemoryConfig>& memory_config_arg,
    const std::optional<DataType>& output_dtype,
    uint8_t queue_id) {
    const std::array<uint32_t, 2>& dilation = {1, 1};

    sliding_window::SlidingWindowConfig sliding_window_config{
        .batch_size = batch_size,
        .input_hw = {input_h, input_w},
        .window_hw = {kernel_size.at(0), kernel_size.at(1)},
        .stride_hw = {stride.at(0), stride.at(1)},
        .pad_hw = {padding.at(0), padding.at(1)},
        .dilation_hw = {dilation.at(0), dilation.at(1)}};
    const auto output_shape = sliding_window_config.get_output_shape();
    tt::log_info("{}", output_shape);
    const bool is_out_tiled = false;

    sliding_window::ParallelConfig parallel_config;
    uint32_t num_cores_nhw = 0;

    // TODO
    assert(!input_tensor.is_sharded());

    // Input is not sharded. Perform sharding.
    parallel_config = conv::conv2d::determine_parallel_config(
        TensorMemoryLayout::HEIGHT_SHARDED,
        batch_size,
        0,  // in_channels -- not used
        output_shape[1],
        output_shape[2],
        0,  // out_channels -- not used
        input_tensor.device(),
        ShardOrientation::ROW_MAJOR,
        is_out_tiled);
    num_cores_nhw = conv::conv2d::get_num_cores_nhw_from_parallel_config(parallel_config);
    auto sharded_mem_config = conv::conv2d::create_sharded_memory_config_from_parallel_config(
        input_tensor.shape(), parallel_config, is_out_tiled ? input_tensor.get_tile().get_tile_shape().at(0) : 1);
    auto input_tensor_sharded = ttnn::to_memory_config(input_tensor, sharded_mem_config, std::nullopt);

    const sliding_window::SlidingWindowConfig halo_sliding_window_config = sliding_window::SlidingWindowConfig{
        .batch_size = batch_size,
        .input_hw = {input_h, input_w},
        .window_hw = {kernel_size.at(0), kernel_size.at(1)},
        .stride_hw = {stride.at(0), stride.at(1)},
        .pad_hw = {padding.at(0), padding.at(1)},
        .dilation_hw = {dilation.at(0), dilation.at(1)},
        .num_cores_nhw = num_cores_nhw,
        .core_range_set = parallel_config.grid,
        .snap_to_tile = false};

    // TODO: Debug
    auto op_trace_metadata = sliding_window::generate_op_trace_metadata(halo_sliding_window_config);
    for (const auto index : op_trace_metadata) {
        tt::log_info("index {}", index);
    }
    for (const auto& boundary :
         sliding_window::generate_shard_boundaries(halo_sliding_window_config, op_trace_metadata)) {
        const auto& [start, end] = boundary;
        tt::log_info("boundary: out {}:{} in {}:{}", start.first, start.second, end.first, end.second);
    }

    // call the halo uop
    const uint32_t pad_val = 0;
    auto haloed_tensor = ttnn::halo(
        queue_id,
        input_tensor_sharded,
        halo_sliding_window_config,
        pad_val,
        /* remote_read=*/false,
        /* transpose_mcast=*/parallel_config.shard_orientation == ShardOrientation::COL_MAJOR,
        /* reshard_num_cores_nhw=*/0,
        /* output_memory_config=*/input_tensor_sharded.memory_config(),
        // TODO: this is nothing to do with output tensor
        /* is_out_tiled=*/false);

    tt::log_info("haloed {}", haloed_tensor);

    MemoryConfig output_memory_config = input_tensor_sharded.memory_config();

    // update the shard spec to match the output shape
    auto shard_spec = output_memory_config.shard_spec.value();
    const uint32_t output_shard_width_padded = tt::round_up(
        output_shape[3] * tt::datum_size(tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype())),
        tt::constants::TILE_WIDTH);
    const uint32_t output_nhw = output_shape[0] * output_shape[1] * output_shape[2];
    const uint32_t output_nhw_padded =
        tt::round_up(output_nhw, num_cores_nhw * (is_out_tiled ? tt::constants::TILE_HEIGHT : 1));
    uint32_t output_shard_height_padded = output_nhw_padded / num_cores_nhw;
    tt::log_info(
        "output_nhw: {}, output_nhw_padded: {}, output_shard_height_padded: {}, output_shard_width_padded: {}",
        output_nhw,
        output_nhw_padded,
        output_shard_height_padded,
        output_shard_width_padded);
    output_memory_config.shard_spec = ShardSpec{
        shard_spec.grid, {output_shard_height_padded, output_shard_width_padded}, ShardOrientation::ROW_MAJOR, false};

    return ttnn::prim::avg_pool2d(
        queue_id, haloed_tensor, halo_sliding_window_config, DataType::BFLOAT16, output_memory_config);
}

}  // namespace ttnn::operations::experimental::pool
