// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "generic_pools.hpp"
#include "pool2d_utils.hpp"

#include <tt-metalium/buffer_constants.hpp>
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/sliding_window/halo/halo.hpp"
#include "ttnn/operations/sliding_window/sliding_window.hpp"
#include "ttnn/operations/data_movement/move/move.hpp"
#include "ttnn/operations/conv/conv2d/device/conv2d_op.hpp"
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/math.hpp>
#include "ttnn/operations/ccl/common/uops/ccl_command.hpp"
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
    const Tensor& input_tensor, //1, 1, 36, 6144
    uint32_t batch_size,    //1
    uint32_t input_h,   //6
    uint32_t input_w,   //6
    uint32_t channels,  //6144
    std::array<uint32_t, 2> kernel_size,    //5,5
    std::array<uint32_t, 2> stride, //1,1
    std::array<uint32_t, 2> padding, //2, 2
    std::array<uint32_t, 2> dilation, //1,1
    const std::optional<const MemoryConfig>& memory_config, //std::nullopt
    const std::optional<const TensorMemoryLayout> applied_shard_scheme, //HEIGHT_SHARDED
    bool ceil_mode) {//false
    sliding_window::SlidingWindowConfig sliding_window_config{
            .batch_size = batch_size,
            .input_hw = {input_h, input_w},
            .window_hw = {kernel_size.at(0), kernel_size.at(1)},
            .stride_hw = {stride.at(0), stride.at(1)},
            .pad_hw = {padding.at(0), padding.at(1)},
            .dilation_hw = {dilation.at(0), dilation.at(1)},
            .ceil_mode = ceil_mode,
    };
    auto output_shape = sliding_window_config.get_output_shape();   //1,6,6,0
    auto input_tensor_sharded = input_tensor;

    // pool output is row major
    bool is_out_tiled = false;
    bool is_in_tiled = input_tensor.dtype() == DataType::BFLOAT8_B; // false

    sliding_window::ParallelConfig parallel_config;
    MemoryConfig out_memory_config = input_tensor_sharded.memory_config();
    uint32_t num_cores_nhw = 0;
    uint32_t num_cores_c = 0;

    TensorMemoryLayout shard_layout = TensorMemoryLayout::HEIGHT_SHARDED; // default to height sharding
    if (!out_memory_config.shard_spec.has_value()) { //true
        // Input is not sharded. Perform sharding.
        if (applied_shard_scheme.has_value()) {
            TT_FATAL((applied_shard_scheme.value() == TensorMemoryLayout::HEIGHT_SHARDED) ||
                     (applied_shard_scheme.value() == TensorMemoryLayout::WIDTH_SHARDED) ||
                     (applied_shard_scheme.value() == TensorMemoryLayout::BLOCK_SHARDED),
                     "Only height, width, or block sharding strategies are supported.");
            shard_layout = applied_shard_scheme.value();
        }
        else { //auto-sharding
            const auto compute_grid_size = input_tensor.device()->compute_with_storage_grid_size();//{8,8}
            DeviceComputeKernelConfig compute_config = conv::get_conv_default_compute_kernel_config(input_tensor.device());
            const uint32_t output_height = ((input_h - kernel_size[0] - ((kernel_size[0] - 1) * (dilation[0] - 1)) + 2 * padding[0]) / stride[0]) + 1;
            const uint32_t output_width = ((input_w - kernel_size[1] - ((kernel_size[0] - 1) * (dilation[0] - 1)) + 2 * padding[1]) / stride[1]) + 1;


            shard_layout = pool::determine_pool_config_for_auto_shard(
                batch_size,
                channels,
                output_height,
                output_width,
                input_h,
                input_w,
                compute_grid_size,
                input_tensor.layout(),
                kernel_size,
                compute_config,
                input_tensor.dtype());
        }

        parallel_config = conv::determine_parallel_config(
                                            shard_layout,    //HEIGHT_SHARDED
                                            batch_size,     //1
                                            channels,       //6144
                                            output_shape[1],    //6
                                            output_shape[2],    //6
                                            channels,       //6144
                                            input_tensor.device()->compute_with_storage_grid_size(),    //8,8
                                            ShardOrientation::ROW_MAJOR,
                                            false,
                                            false);
        //parallel_config.grid = {(0,0;7,3),(0,4;3,4)}, shard_scheme = HEIGHT_SHARDED, shard_orientation = ROW_MAJOR,
        num_cores_nhw = conv::get_num_cores_nhw_from_parallel_config(parallel_config);  //36
        num_cores_c = conv::get_num_cores_channels_from_parallel_config(parallel_config);   //1
        auto sharded_mem_config = conv::create_sharded_memory_config_from_parallel_config(input_tensor_sharded.get_padded_shape(), parallel_config, is_in_tiled ? tt::constants::TILE_HEIGHT : 1);
        //sharded_mem_config:{HEIGHT_SHARDED, L1, {{(0,0;7,3),(0,4;3,4)},{1,6144},ROW_MAJOR,PHYSICAL}}
        input_tensor_sharded = ttnn::to_memory_config(input_tensor_sharded, sharded_mem_config, std::nullopt); // 1,1,36,6144
        out_memory_config = input_tensor_sharded.memory_config();   // == sharded_mem_config
    } else {
        // input is already sharded, use it as is
        const auto shard_grid = out_memory_config.shard_spec.value().grid;
        const auto shard_scheme = out_memory_config.memory_layout;
        const auto shard_orientation = out_memory_config.shard_spec.value().orientation;
        TT_FATAL(!applied_shard_scheme.has_value(), "A sharding scheme should not be specified for a sharded input tensor.");
        TT_FATAL(shard_orientation == ShardOrientation::ROW_MAJOR, "Only row major orientation is supported.");
        parallel_config.grid = shard_grid;
        parallel_config.shard_scheme = shard_scheme;
        parallel_config.shard_orientation = shard_orientation;
        num_cores_nhw = conv::get_num_cores_nhw_from_parallel_config(parallel_config);
        num_cores_c = conv::get_num_cores_channels_from_parallel_config(parallel_config);
    }

    // update the shard spec to match the output shape
    auto shard_spec = out_memory_config.shard_spec.value();
    uint32_t output_shard_width_padded = input_tensor.dtype() == DataType::BFLOAT8_B ? ccl::cmd::round_up(channels / num_cores_c, tt::constants::TILE_WIDTH) : ccl::cmd::round_up(channels / num_cores_c * tt::datum_size(tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype())), tt::constants::TILE_WIDTH);    //12288
    uint32_t output_nhw = output_shape[0] * output_shape[1] * output_shape[2];  //36
    uint32_t output_nhw_padded = ccl::cmd::round_up(output_nhw, num_cores_nhw * (is_out_tiled ? tt::constants::TILE_HEIGHT : 1)); //36
    uint32_t output_shard_height_padded = output_nhw_padded / num_cores_nhw;    //1
    log_debug(tt::LogOp, "output_nhw: {}, output_nhw_padded: {}, output_shard_height_padded: {}, output_shard_width_padded: {}", output_nhw, output_nhw_padded, output_shard_height_padded, output_shard_width_padded);
    out_memory_config.shard_spec = tt::tt_metal::ShardSpec{shard_spec.grid, {output_shard_height_padded, output_shard_width_padded}, ShardOrientation::ROW_MAJOR};
    //out_memory_config:{HEIGHT_SHARDED, L1, {{(0,0;7,3),(0,4;3,4)},{1,6144},ROW_MAJOR,PHYSICAL}}

    sliding_window_config = sliding_window::SlidingWindowConfig{
            .batch_size = batch_size,
            .input_hw = {input_h, input_w},
            .window_hw = {kernel_size.at(0), kernel_size.at(1)},
            .stride_hw = {stride.at(0), stride.at(1)},
            .pad_hw = {padding.at(0), padding.at(1)},
            .dilation_hw = {dilation.at(0), dilation.at(1)},
            .num_cores_nhw = num_cores_nhw,
            .num_cores_c = num_cores_c,
            .core_range_set = parallel_config.grid,
            .snap_to_tile = false,
            .ceil_mode = ceil_mode,
    };

    // Call the halo uop
    auto haloed_tensor = ttnn::halo(
        queue_id,
        input_tensor_sharded,
        sliding_window_config,
        get_bf16_pool_init_value(pool_type), // pad_val, 65408
        false,
        parallel_config.shard_orientation == ShardOrientation::COL_MAJOR,
        0,
        input_tensor_sharded.memory_config(),
        is_out_tiled);  //halo:1,1,1620,6144

    auto output_tensor = ttnn::prim::pool2d(
        queue_id,
        haloed_tensor,
        sliding_window_config,
        pool_type,
        DataType::BFLOAT16,      // input_tensor.dtype(), // currently only bfp16 output is supported
        out_memory_config); //output_tensor:1,1,36,6144

    if (memory_config.has_value() && memory_config.value() != out_memory_config) {
        output_tensor = ttnn::to_memory_config(output_tensor, memory_config.value(), std::nullopt);
    }

    return output_tensor;
}

template class Pool2DOp<Pool2DType::MAX_POOL2D>;

}  // namespace operations::pool
}  // namespace ttnn
