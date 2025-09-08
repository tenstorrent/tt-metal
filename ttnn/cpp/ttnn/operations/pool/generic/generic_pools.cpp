// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "generic_pools.hpp"

#include "tt-metalium/constants.hpp"
#include <cmath>
#include <tt-metalium/buffer_types.hpp>
#include "ttnn/operations/conv/conv2d/conv2d_utils.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/pool/pool_utils.hpp"
#include "ttnn/operations/sliding_window/halo/halo.hpp"
#include "ttnn/operations/sliding_window/sliding_window.hpp"
#include "ttnn/operations/data_movement/move/move.hpp"
#include "ttnn/operations/functions.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/copy/typecast/typecast.hpp"
#include "ttnn/operations/data_movement/repeat/repeat.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/math.hpp>

namespace ttnn {
namespace operations::pool {

// Generic invoke function for both max and avg pool operations. Most of the arguments are shared excpet for the
// dilation which is set to (1,1) for avg pool and count_include_pad and divisor_override which have no effect on
// maxpool.
static std::variant<Tensor, MaxPoolWithIndicesResult> pool2d_invoke(
    QueueId queue_id,
    const Tensor& input_tensor,
    Pool2DType pool_type,
    uint32_t batch_size,
    uint32_t input_h,
    uint32_t input_w,
    uint32_t channels,
    std::array<uint32_t, 2> kernel_size,
    std::array<uint32_t, 2> stride,
    std::variant<std::array<uint32_t, 2>, std::array<uint32_t, 4>> padding,
    std::optional<std::array<uint32_t, 2>> dilation = std::nullopt,
    bool ceil_mode = false,
    bool count_include_pad = true,
    std::optional<int32_t> divisor_override = std::nullopt,
    const std::optional<const MemoryConfig>& memory_config = std::nullopt,
    const std::optional<const TensorMemoryLayout> applied_shard_scheme = std::nullopt,
    bool in_place_halo = false,
    bool deallocate_input = false,
    bool reallocate_halo_output = true,
    bool return_indices = false) {
    std::array<uint32_t, 4> padding_4d = sliding_window::get_pair_n4_padding(padding);
    bool is_out_tiled = false;  // pool output is row major
    bool is_in_tiled = input_tensor.layout() == ttnn::TILE_LAYOUT;
    validate_input_params(
        input_tensor,
        batch_size,
        input_h,
        input_w,
        channels,
        kernel_size,
        stride,
        padding_4d[0],
        padding_4d[1],
        padding_4d[2],
        padding_4d[3],
        dilation.has_value() ? dilation.value()[0] : 1,
        dilation.has_value() ? dilation.value()[1] : 1,
        is_in_tiled);
    uint32_t dilation_h = dilation.has_value() ? dilation.value().at(0) : 1;
    uint32_t dilation_w = dilation.has_value() ? dilation.value().at(1) : 1;
    sliding_window::SlidingWindowConfig sliding_window_config{
        .batch_size = batch_size,
        .channels = channels,
        .input_hw = {input_h, input_w},
        .window_hw = {kernel_size.at(0), kernel_size.at(1)},
        .stride_hw = {stride.at(0), stride.at(1)},
        .padding = {padding_4d.at(0), padding_4d.at(1), padding_4d.at(2), padding_4d.at(3)},
        .dilation_hw = {dilation_h, dilation_w},
        .ceil_mode = ceil_mode,
        .is_avg_pool = pool_type == Pool2DType::AVG_POOL2D,
    };
    auto output_shape = sliding_window_config.get_output_shape();
    const bool is_input_tensor_in_dram = input_tensor.memory_config().is_dram();
    sliding_window::ParallelConfig parallel_config;
    MemoryConfig out_memory_config = input_tensor.memory_config();
    uint32_t num_cores_nhw = 0;
    uint32_t num_cores_c = 0;
    Tensor input_tensor_sharded = input_tensor;
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
                tt::constants::TILE_WIDTH,
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
                    input_tensor,
                    sliding_window_config,
                    channels,
                    pool_type,
                    count_include_pad,
                    divisor_override,
                    return_indices);
            TT_FATAL(
                sw_parallel_config.has_value(),
                "autosharding could not determine valid shard scheme, please check tensor dimensions");
            parallel_config = sw_parallel_config.value();
        }

        num_cores_nhw = conv::get_num_cores_nhw_from_parallel_config(parallel_config);
        num_cores_c = conv::get_num_cores_channels_from_parallel_config(parallel_config);

        // This is the code path of the non sharded input tensor, this means that input channels
        // can be whatever number here so we need to have the shard_width aligned to the l1 memory alignment
        // which is 8, in case shard_width is multiple of 16 or 32 we will take largest number possible. We are aligning
        // it by changing the padded shape of the tensor.
        uint32_t input_channels_alignment = is_in_tiled ? tt::constants::TILE_WIDTH : 8U;
        if (input_tensor.memory_config().is_sharded() && input_tensor.layout() == Layout::ROW_MAJOR) {
            const uint32_t shard_width = input_tensor.memory_config().shard_spec()->shape[1];
            input_channels_alignment = (shard_width % tt::constants::TILE_WIDTH == 0) ? tt::constants::TILE_WIDTH
                                       : (shard_width % 16 == 0)                      ? 16U
                                                                                      : 8U;
        }

        ttnn::Shape input_tensor_shape = input_tensor.padded_shape();
        uint32_t input_tensor_width_snapped_to_channels_alignment =
            tt::round_up(input_tensor_shape[3], num_cores_c * input_channels_alignment);

        ttnn::Shape input_padded_shape = ttnn::Shape(
            {input_tensor_shape[0],
             input_tensor_shape[1],
             input_tensor_shape[2],
             input_tensor_width_snapped_to_channels_alignment});

        input_tensor_sharded = input_tensor.reshape(input_tensor_shape, input_padded_shape);

        auto sharded_mem_config = conv::create_sharded_memory_config_from_parallel_config(
            input_padded_shape, parallel_config, is_in_tiled ? tt::constants::TILE_HEIGHT : 1);
        input_tensor_sharded = ttnn::to_memory_config(input_tensor_sharded, sharded_mem_config, std::nullopt);
        out_memory_config = input_tensor_sharded.memory_config();
    } else {
        TT_FATAL(
            !applied_shard_scheme.has_value(), "A sharding scheme should not be specified for a sharded input tensor.");
        // input is already sharded, use it as is
        TT_FATAL(
            out_memory_config.shard_spec().value().orientation == ShardOrientation::ROW_MAJOR,
            "Only row major orientation is supported.");

        parallel_config.grid = out_memory_config.shard_spec().value().grid;
        parallel_config.shard_scheme = out_memory_config.memory_layout();
        parallel_config.shard_orientation = out_memory_config.shard_spec().value().orientation;

        num_cores_nhw = conv::get_num_cores_nhw_from_parallel_config(parallel_config);
        num_cores_c = conv::get_num_cores_channels_from_parallel_config(parallel_config);
    }

    // update the shard spec to match the output shape
    auto shard_spec = out_memory_config.shard_spec().value();
    uint32_t output_nhw = output_shape[0] * output_shape[1] * output_shape[2];
    uint32_t output_nhw_padded =
        tt::round_up(output_nhw, num_cores_nhw * (is_out_tiled ? tt::constants::TILE_HEIGHT : 1));
    uint32_t output_shard_height_padded = output_nhw_padded / num_cores_nhw;
    uint32_t output_c = channels;
    uint32_t output_c_padded = tt::round_up(output_c, tt::constants::TILE_WIDTH / 2);
    if (shard_layout == TensorMemoryLayout::WIDTH_SHARDED || shard_layout == TensorMemoryLayout::BLOCK_SHARDED) {
        output_c_padded = tt::round_up(output_c, num_cores_c * tt::constants::TILE_WIDTH / 2);
    }
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
        .padding = {padding_4d.at(0), padding_4d.at(1), padding_4d.at(2), padding_4d.at(3)},
        .dilation_hw = {dilation_h, dilation_w},
        .num_cores_nhw = num_cores_nhw,
        .num_cores_c = num_cores_c,
        .core_range_set = parallel_config.grid,
        .snap_to_tile = false,
        .ceil_mode = ceil_mode,
        .is_avg_pool = pool_type == Pool2DType::AVG_POOL2D,
    };

    // create the index tensor if needed
    Tensor index_tensor_sharded;
    if (return_indices) {
        Shape spatial_shape({1, input_h, input_w, 1});

        // Create indices tensor with UINT32 since repeat operation requires it
        Tensor indices_hw = ttnn::index_all<uint32_t>(
            spatial_shape,
            spatial_shape,  // No padding needed for spatial-only shape
            DataType::UINT32);
        Shape repeat_shape({batch_size, 1, 1, channels});
        Tensor index_full = ttnn::repeat(indices_hw.to_device(input_tensor.device()), repeat_shape);

        // Reshape from [batch_size, input_h, input_w, channels] to [1, 1, batch_size * input_h * input_w, channels]
        uint32_t nhw = batch_size * input_h * input_w;
        Shape flattened_shape({1, 1, nhw, channels});
        Tensor index_full_reshaped = ttnn::reshape(index_full, flattened_shape);

        // Convert to TILE layout for typecast operation
        Tensor index_full_tiled = ttnn::to_layout(index_full_reshaped, ttnn::TILE_LAYOUT);

        // Convert to UINT16
        Tensor index_full_uint16 = ttnn::typecast(index_full_tiled, DataType::UINT16);

        // Convert back to ROW_MAJOR layout
        if (!is_in_tiled) {
            index_full_uint16 = ttnn::to_layout(index_full_uint16, ttnn::ROW_MAJOR_LAYOUT);
        }

        TT_FATAL(
            input_tensor_sharded.memory_config().is_sharded(), "Input tensor must be sharded to shard indices tensor.");
        index_tensor_sharded =
            ttnn::to_memory_config(index_full_uint16, input_tensor_sharded.memory_config(), std::nullopt);
    }

    std::vector<Tensor> haloed_tensors;

    // call the halo uop
    Tensor haloed_tensor = ttnn::halo(
        queue_id,
        input_tensor_sharded,
        sliding_window_config,
        get_bf16_pool_init_value(pool_type),  // pad_val
        false,
        parallel_config.shard_orientation == ShardOrientation::COL_MAJOR,
        input_tensor_sharded.memory_config(),
        is_out_tiled,
        in_place_halo);

    if (deallocate_input || is_input_tensor_in_dram) {
        input_tensor_sharded.deallocate(/*force*/ true);
    }

    if (reallocate_halo_output) {
        haloed_tensor = ttnn::move(haloed_tensor);
    }
    haloed_tensors.push_back(std::move(haloed_tensor));

    if (return_indices) {
        Tensor haloed_index = ttnn::halo(
            queue_id,
            index_tensor_sharded,
            sliding_window_config,
            0,  // pad_val - should never be used as padding should never be the max index
            false,
            parallel_config.shard_orientation == ShardOrientation::COL_MAJOR,
            index_tensor_sharded.memory_config(),
            is_out_tiled,
            in_place_halo);

        if (deallocate_input || is_input_tensor_in_dram) {
            index_tensor_sharded.deallocate(/*force*/ true);
        }

        if (reallocate_halo_output) {
            haloed_index = ttnn::move(haloed_index);
        }
        haloed_tensors.push_back(std::move(haloed_index));
    }

    const uint32_t pre_allocate_size =
        haloed_tensor.device()->allocator()->get_statistics(tt::tt_metal::BufferType::L1).total_allocated_bytes;

    // call the pool2d uop
    std::vector<Tensor> output_tensors = ttnn::prim::pool2d(
        queue_id,
        haloed_tensors,
        sliding_window_config,
        pool_type,
        DataType::BFLOAT16,  // input_tensor.dtype(), // currently only bfp16 output is supported
        out_memory_config,
        count_include_pad,
        divisor_override,
        return_indices,
        pre_allocate_size);

    // format and return the result
    if (memory_config.has_value() && memory_config.value() != out_memory_config) {
        for (int i = 0; i < output_tensors.size(); i++) {
            output_tensors[i] = ttnn::to_memory_config(output_tensors[i], memory_config.value(), std::nullopt);
        }
    }

    if (return_indices) {
        TT_FATAL(
            output_tensors.size() == 2,
            "Expected two output tensors when return_indices is true, but got {}.",
            output_tensors.size());
        return MaxPoolWithIndicesResult{std::move(output_tensors[0]), std::move(output_tensors[1])};
    } else {
        TT_FATAL(output_tensors.size() == 1, "Expected a single output tensor when return_indices is false.");
        return std::move(output_tensors[0]);
    }
}

std::variant<Tensor, MaxPoolWithIndicesResult> MaxPool2DOp::invoke(
    QueueId queue_id,
    const Tensor& input_tensor,
    uint32_t batch_size,
    uint32_t input_h,
    uint32_t input_w,
    uint32_t channels,
    std::array<uint32_t, 2> kernel_size,
    std::array<uint32_t, 2> stride,
    std::variant<std::array<uint32_t, 2>, std::array<uint32_t, 4>> padding,
    std::array<uint32_t, 2> dilation,
    bool ceil_mode,
    const std::optional<const MemoryConfig>& memory_config,
    const std::optional<const TensorMemoryLayout> applied_shard_scheme,
    bool in_place_halo,
    bool deallocate_input,
    bool reallocate_halo_output,
    bool return_indices) {
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
        in_place_halo,
        deallocate_input,
        reallocate_halo_output,
        return_indices);
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
    std::variant<std::array<uint32_t, 2>, std::array<uint32_t, 4>> padding,
    bool ceil_mode,
    bool count_include_pad,
    std::optional<int32_t> divisor_override,
    const std::optional<const MemoryConfig>& memory_config,
    const std::optional<const TensorMemoryLayout> applied_shard_scheme,
    bool in_place_halo,
    bool deallocate_input,
    bool reallocate_halo_output) {
    auto result = pool2d_invoke(
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
        in_place_halo,
        deallocate_input,
        reallocate_halo_output,
        false  // return_indices
    );

    // Average pool always returns just the tensor, never indices
    return std::get<Tensor>(result);
}

}  // namespace operations::pool
}  // namespace ttnn
