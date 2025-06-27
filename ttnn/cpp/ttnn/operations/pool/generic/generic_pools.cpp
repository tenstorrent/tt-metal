// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "generic_pools.hpp"

#include <cstdint>
#include <optional>
#include "tt-metalium/constants.hpp"
#include <tt-metalium/buffer_types.hpp>
#include "ttnn/operations/conv/conv2d/conv2d_utils.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/pool/pool_utils.hpp"
#include "ttnn/operations/sliding_window/halo/halo.hpp"
#include "ttnn/operations/sliding_window/sliding_window.hpp"
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/math.hpp>
#include <limits>

namespace ttnn {
namespace operations::pool {

// Generic invoke function for both max and avg pool operations. Most of the arguments are shared excpet for the
// dilation which is set to (1,1) for avg pool and count_include_pad and divisor_override which have no effect on
// maxpool.
static Tensor pool2d_invoke(
    QueueId queue_id,
    const Tensor& input_tensor,
    Pool2DType pool_type,
    uint32_t batch_size,
    uint32_t input_h,
    uint32_t input_w,
    uint32_t channels,
    std::array<uint32_t, 2> kernel_size,
    std::array<uint32_t, 2> stride,
    std::array<uint32_t, 2> padding,
    std::optional<std::array<uint32_t, 2>> dilation = std::nullopt,
    bool ceil_mode = false,
    bool count_include_pad = true,
    std::optional<int32_t> divisor_override = std::nullopt,
    const std::optional<const MemoryConfig>& memory_config = std::nullopt,
    const std::optional<const TensorMemoryLayout> applied_shard_scheme = std::nullopt,
    bool in_place_halo = false) {
    uint32_t dilation_h = dilation.has_value() ? dilation.value().at(0) : 1;
    uint32_t dilation_w = dilation.has_value() ? dilation.value().at(1) : 1;
    sliding_window::SlidingWindowConfig sliding_window_config{
        .batch_size = batch_size,
        .channels = channels,
        .input_hw = {input_h, input_w},
        .window_hw = {kernel_size.at(0), kernel_size.at(1)},
        .stride_hw = {stride.at(0), stride.at(1)},
        .padding = {padding.at(0), padding.at(0), padding.at(1), padding.at(1)},
        .dilation_hw = {dilation_h, dilation_w},
        .ceil_mode = ceil_mode,
        .is_avg_pool = pool_type == Pool2DType::AVG_POOL2D,
    };
    auto output_shape = sliding_window_config.get_output_shape();
    auto input_shape = sliding_window_config.get_input_shape();
    auto input_tensor_shape = input_tensor.logical_shape();
    auto input_padded_shape = ttnn::Shape(
        {input_tensor_shape[0], input_tensor_shape[1], input_tensor_shape[2], tt::round_up(input_tensor_shape[3], 8)});
    auto input_padded_tensor = input_tensor;
    input_padded_tensor = input_padded_tensor.reshape(input_tensor_shape, input_padded_shape);
    log_info(tt::LogOp, "input_padded_tensor shape: {}", input_padded_tensor.logical_shape());
    auto input_tensor_sharded = input_padded_tensor;
    // pool output is row major
    bool is_out_tiled = false;
    bool is_in_tiled = input_tensor.layout() == ttnn::TILE_LAYOUT;

    sliding_window::ParallelConfig parallel_config;
    MemoryConfig out_memory_config = input_tensor_sharded.memory_config();
    uint32_t num_cores_nhw = 0;
    uint32_t num_cores_c = 0;
    uint32_t channels_padded = input_padded_shape[3];

    TensorMemoryLayout shard_layout = TensorMemoryLayout::HEIGHT_SHARDED;  // default to height sharding
    if (!out_memory_config.shard_spec().has_value()) {
        // Input is not sharded. Perform sharding.
        if (applied_shard_scheme.has_value()) {
            TT_FATAL(
                (applied_shard_scheme.value() == TensorMemoryLayout::HEIGHT_SHARDED) ||
                    (applied_shard_scheme.value() == TensorMemoryLayout::WIDTH_SHARDED) ||
                    (applied_shard_scheme.value() == TensorMemoryLayout::BLOCK_SHARDED),
                "Only height, width, or block sharding strategies are supported.");
            shard_layout = applied_shard_scheme.value();
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
                is_in_tiled,  // if input is tiled we need to choose num_cores_c to make the shard width to be a tile
                              // multiple, it cannot be 16
                0);
        } else {  // auto-sharding
            std::optional<sliding_window::ParallelConfig> sw_parallel_config =
                pool::determine_pool_config_for_auto_shard(
                    input_tensor, sliding_window_config, channels, pool_type, count_include_pad, divisor_override);
            TT_FATAL(
                sw_parallel_config.has_value(),
                "autosharding could not determine valid shard scheme, please check tensor dimensions");
            parallel_config = sw_parallel_config.value();
        }

        // tt::log_debug(tt::LogOp, "auto sharding spec: {}", parallel_config.shard_scheme);
        num_cores_nhw = conv::get_num_cores_nhw_from_parallel_config(parallel_config);
        num_cores_c = conv::get_num_cores_channels_from_parallel_config(parallel_config);
        auto sharded_mem_config = conv::create_sharded_memory_config_from_parallel_config(
            input_padded_shape, parallel_config, is_in_tiled ? tt::constants::TILE_HEIGHT : 1);
        input_tensor_sharded = ttnn::to_memory_config(
            input_tensor_sharded, sharded_mem_config, std::nullopt);  // this converts interleaved to sharded
        out_memory_config = input_tensor_sharded.memory_config();
    } else {
        // input is already sharded, use it as is
        const auto shard_grid = out_memory_config.shard_spec().value().grid;
        const auto shard_scheme = out_memory_config.memory_layout();
        const auto shard_orientation = out_memory_config.shard_spec().value().orientation;
        TT_FATAL(
            !applied_shard_scheme.has_value(), "A sharding scheme should not be specified for a sharded input tensor.");
        TT_FATAL(shard_orientation == ShardOrientation::ROW_MAJOR, "Only row major orientation is supported.");
        parallel_config.grid = shard_grid;
        parallel_config.shard_scheme = shard_scheme;
        parallel_config.shard_orientation = shard_orientation;
        num_cores_nhw = conv::get_num_cores_nhw_from_parallel_config(parallel_config);
        num_cores_c = conv::get_num_cores_channels_from_parallel_config(parallel_config);
    }

    // update the shard spec to match the output shape
    auto shard_spec = out_memory_config.shard_spec().value();
    uint32_t output_shard_width_padded =
        input_tensor.dtype() == DataType::BFLOAT8_B
            ? tt::round_up(channels / num_cores_c, tt::constants::TILE_WIDTH)
            : tt::round_up(
                  channels_padded / num_cores_c *
                      tt::datum_size(tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype())),
                  tt::constants::TILE_WIDTH);
    uint32_t output_nhw = output_shape[0] * output_shape[1] * output_shape[2];
    uint32_t output_nhw_padded =
        tt::round_up(output_nhw, num_cores_nhw * (is_out_tiled ? tt::constants::TILE_HEIGHT : 1));
    uint32_t output_shard_height_padded = output_nhw_padded / num_cores_nhw;
    uint32_t output_c = channels;
    uint32_t output_c_padded = tt::round_up(output_c, num_cores_c * (is_out_tiled ? tt::constants::TILE_WIDTH : 1));
    uint32_t output_shard_width_padded = output_c_padded / num_cores_c;
    log_debug(
        tt::LogOp,
        "output_nhw: {}, output_nhw_padded: {}, output_shard_height_padded: {}, output_shard_width_padded: {}",
        output_nhw,
        output_nhw_padded,
        output_shard_height_padded,
        output_shard_width_padded);
    out_memory_config = out_memory_config.with_shard_spec(tt::tt_metal::ShardSpec{
        shard_spec.grid, {output_shard_height_padded, output_shard_width_padded}, ShardOrientation::ROW_MAJOR});
    sliding_window_config = sliding_window::SlidingWindowConfig{
        .batch_size = batch_size,
        .channels = channels,
        .input_hw = {input_h, input_w},
        .window_hw = {kernel_size.at(0), kernel_size.at(1)},
        .stride_hw = {stride.at(0), stride.at(1)},
        .padding = {padding.at(0), padding.at(0), padding.at(1), padding.at(1)},
        .dilation_hw = {dilation_h, dilation_w},
        .num_cores_nhw = num_cores_nhw,
        .num_cores_c = num_cores_c,
        .core_range_set = parallel_config.grid,
        .snap_to_tile = false,
        .ceil_mode = ceil_mode,
        .is_avg_pool = pool_type == Pool2DType::AVG_POOL2D,
    };

    // Call the halo uop
    auto haloed_tensor = ttnn::halo(
        queue_id,
        input_tensor_sharded,
        sliding_window_config,
        get_bf16_pool_init_value(pool_type),  // pad_val
        false,
        parallel_config.shard_orientation == ShardOrientation::COL_MAJOR,
        input_tensor_sharded.memory_config(),
        is_out_tiled,
        in_place_halo);

    const uint32_t pre_allocate_size =
        haloed_tensor.device()->allocator()->get_statistics(tt::tt_metal::BufferType::L1).total_allocated_bytes;

    const uint32_t pre_allocate_size =
        haloed_tensor.device()->allocator()->get_statistics(tt::tt_metal::BufferType::L1).total_allocated_bytes;

    auto output_tensor = ttnn::prim::pool2d(
        queue_id,
        haloed_tensor,
        sliding_window_config,
        pool_type,
        DataType::BFLOAT16,  // input_tensor.dtype(), // currently only bfp16 output is supported
        out_memory_config,
        count_include_pad,
        divisor_override,
        pre_allocate_size);

    if (memory_config.has_value() && memory_config.value() != out_memory_config) {
        output_tensor = ttnn::to_memory_config(output_tensor, memory_config.value(), std::nullopt);
    }

    return output_tensor;
}

Tensor MaxPool2DOp::invoke(
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
    bool ceil_mode,
    const std::optional<const MemoryConfig>& memory_config,
    const std::optional<const TensorMemoryLayout> applied_shard_scheme,
    bool in_place_halo) {
    return pool2d_invoke(
        queue_id,
        input_tensor,
        Pool2DType::MAX_POOL2D,
        batch_size,
        input_h,
        input_w,
        channels,
        kernel_size,
        stride,
        padding,
        dilation,
        ceil_mode,
        true,          // count_include_pad
        std::nullopt,  // divisor_override
        memory_config,
        applied_shard_scheme,
        in_place_halo);
}

Tensor AvgPool2DOp::invoke(
    QueueId queue_id,
    const Tensor& input_tensor,
    uint32_t batch_size,
    uint32_t input_h,
    uint32_t input_w,
    uint32_t channels,
    std::array<uint32_t, 2> kernel_size,
    std::array<uint32_t, 2> stride,
    std::array<uint32_t, 2> padding,
    bool ceil_mode,
    bool count_include_pad,
    std::optional<int32_t> divisor_override,
    const std::optional<const MemoryConfig>& memory_config,
    const std::optional<const TensorMemoryLayout> applied_shard_scheme,
    bool in_place_halo) {
    return pool2d_invoke(
        queue_id,
        input_tensor,
        Pool2DType::AVG_POOL2D,
        batch_size,
        input_h,
        input_w,
        channels,
        kernel_size,
        stride,
        padding,
        std::nullopt,  // dilation
        ceil_mode,
        count_include_pad,
        divisor_override,
        memory_config,
        applied_shard_scheme,
        in_place_halo);
}

}  // namespace operations::pool
}  // namespace ttnn
