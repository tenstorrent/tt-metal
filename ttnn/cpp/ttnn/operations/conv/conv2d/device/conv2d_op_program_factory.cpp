// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "device/device.hpp"
#include "ttnn/operations/conv/conv2d/device/conv2d_op.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/sliding_window/sliding_window.hpp"
#include "tt_metal/common/work_split.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/host_api.hpp"

using namespace tt::constants;

namespace ttnn::operations::conv {
namespace conv2d {



operation::ProgramWithCallbacks multi_core_width_sharded_conv2d(
    tt::tt_metal::Program& program,
    const Tensor& a,
    const Tensor& b,
    const ttnn::Shape& ashape,
    std::optional<const Tensor> bias,
    const std::optional<const Tensor> conv_reader_indices,
    sliding_window::SlidingWindowConfig sliding_window_config,
    uint32_t output_channels,
    uint32_t groups,
    bool untilize_out,
    bool has_bias,
    bool fuse_relu,
    const OptimizedConvParallelizationConfig& parallelization_config,
    const OptimizedConvBlockConfig& block_config,
    bool use_shallow_conv_variant,
    bool transpose_mcast,
    Tensor& output,
    DeviceComputeKernelConfig compute_kernel_config,
    bool enable_act_double_buffer,
    bool enable_split_reader,
    bool enable_subblock_padding);

operation::ProgramWithCallbacks multi_core_block_sharded_conv2d(
    tt::tt_metal::Program& program,
    const Tensor& a,
    const Tensor& b,
    const ttnn::Shape& ashape,
    std::optional<const Tensor> bias,
    const std::optional<const Tensor> conv_reader_indices,
    sliding_window::SlidingWindowConfig sliding_window_config,
    uint32_t output_channels,
    uint32_t groups,
    bool untilize_out,
    bool has_bias,
    bool fuse_relu,
    const OptimizedConvParallelizationConfig& parallelization_config,
    const OptimizedConvBlockConfig& block_config,
    bool use_shallow_conv_variant,
    bool transpose_mcast,
    Tensor& output,
    DeviceComputeKernelConfig compute_kernel_config,
    bool enable_act_double_buffer,
    bool enable_weights_double_buffer,
    bool enable_split_reader,
    bool enable_subblock_padding,
    bool use_non_tile_height);

operation::ProgramWithCallbacks multi_core_height_sharded_conv2d(
    tt::tt_metal::Program& program,
    const Tensor& a,
    const Tensor& b,
    const ttnn::Shape& ashape,
    std::optional<const Tensor> bias,
    const std::optional<const Tensor> conv_reader_indices,
    sliding_window::SlidingWindowConfig sliding_window_config,
    uint32_t output_channels,
    uint32_t groups,
    bool untilize_out,
    bool has_bias,
    bool fuse_relu,
    const OptimizedConvParallelizationConfig& parallelization_config,
    const OptimizedConvBlockConfig& block_config,
    bool use_shallow_conv_variant,
    bool transpose_mcast,
    Tensor& output,
    DeviceComputeKernelConfig compute_kernel_config,
    bool enable_act_double_buffer,
    bool enable_weights_double_buffer,
    bool enable_split_reader,
    bool enable_subblock_padding,
    bool use_non_tile_height);


operation::ProgramWithCallbacks multi_core_conv2d_impl(
    const Tensor& a,
    const Tensor &b,
    std::optional<const Tensor> bias,
    sliding_window::SlidingWindowConfig sliding_window_config,
    uint32_t output_channels,
    uint32_t groups,
    bool untilize_out, bool fuse_relu,
    const OptimizedConvParallelizationConfig& parallelization_config,
    const OptimizedConvBlockConfig& block_config,
    DataType dtype,
    std::array<std::uint32_t, 4> input_tensor_shape,
    bool use_shallow_conv_variant,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config,
    Tensor& output,
    bool enable_act_double_buffer,
    bool enable_weights_double_buffer,
    bool enable_split_reader,
    bool enable_subblock_padding,
    bool use_non_tile_height) {
    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();

    ttnn::operations::sliding_window::ParallelConfig parallel_config;
    parallel_config.grid = a.shard_spec().value().grid;
    parallel_config.shard_scheme = a.memory_config().memory_layout;
    parallel_config.shard_orientation = a.shard_spec().value().orientation;

    // create conv config tensors
    auto pad_metadata = ttnn::operations::sliding_window::generate_pad_metadata(sliding_window_config);
    auto op_trace_metadata = ttnn::operations::sliding_window::generate_op_trace_metadata(sliding_window_config);
    auto shard_boundaries = ttnn::operations::sliding_window::generate_shard_boundaries(sliding_window_config, op_trace_metadata);
    auto conv_sharded_input_top_left_indices =
        ttnn::operations::sliding_window::generate_sliding_window_op_config(op_trace_metadata, shard_boundaries, true, true);
    // create sharded ttnn config tensors
    DataType indices_tt_dtype = DataType::UINT16;
    // For 2d convs, each core in a column or row share the same specs
    CoreCoord grid_size = parallel_config.grid.bounding_box().grid_size();

    bool is_block_sharded = a.memory_config().memory_layout == TensorMemoryLayout::BLOCK_SHARDED;
    auto conv_reader_indices_tensor = ttnn::operations::sliding_window::construct_on_host_config_tensor(
        conv_sharded_input_top_left_indices, sliding_window_config, parallel_config);
    conv_reader_indices_tensor = ttnn::operations::sliding_window::move_config_tensor_to_device(
        conv_reader_indices_tensor, parallel_config, is_block_sharded, a.device());

    // add config tensor to program
    tt::tt_metal::detail::AddConfigBuffer(program, conv_reader_indices_tensor.device_buffer());
    if(parallel_config.shard_scheme == TensorMemoryLayout::WIDTH_SHARDED) {
        return multi_core_width_sharded_conv2d(
            program,
            a,
            b,
            ttnn::Shape(input_tensor_shape),
            bias,
            conv_reader_indices_tensor,
            sliding_window_config,
            output_channels,
            groups,
            untilize_out,
            bias.has_value(),
            fuse_relu,
            parallelization_config,
            block_config,
            use_shallow_conv_variant,
            parallel_config.shard_orientation == ShardOrientation::COL_MAJOR,
            output,
            compute_kernel_config.value(),
            enable_act_double_buffer,
            enable_split_reader,
            enable_subblock_padding);
    } else if(parallel_config.shard_scheme == TensorMemoryLayout::HEIGHT_SHARDED) {
        return multi_core_height_sharded_conv2d(
            program,
            a,
            b,
            ttnn::Shape(input_tensor_shape),
            bias,
            conv_reader_indices_tensor,
            sliding_window_config,
            output_channels,
            groups,
            untilize_out,
            bias.has_value(),
            fuse_relu,
            parallelization_config,
            block_config,
            use_shallow_conv_variant,
            parallel_config.shard_orientation == ShardOrientation::COL_MAJOR,
            output,
            compute_kernel_config.value(),
            enable_act_double_buffer,
            enable_weights_double_buffer,
            enable_split_reader,
            enable_subblock_padding,
            use_non_tile_height);
    } else if(parallel_config.shard_scheme == TensorMemoryLayout::BLOCK_SHARDED) {
        return multi_core_block_sharded_conv2d(
            program,
            a,
            b,
            ttnn::Shape(input_tensor_shape),
            bias,
            conv_reader_indices_tensor,
            sliding_window_config,
            output_channels,
            groups,
            untilize_out,
            bias.has_value(),
            fuse_relu,
            parallelization_config,
            block_config,
            use_shallow_conv_variant,
            parallel_config.shard_orientation == ShardOrientation::COL_MAJOR,
            output,
            compute_kernel_config.value(),
            enable_act_double_buffer,
            enable_weights_double_buffer,
            enable_split_reader,
            enable_subblock_padding,
            use_non_tile_height);
    } else {
        TT_THROW("Unsupported shard scheme {}",parallel_config.shard_scheme);
    }
}  // namespace tt_metal



// TODO: Add namespace for utilities?
std::tuple<CBHandle, CBHandle> create_CBs_for_depthwise_sharded_input(
    tt::tt_metal::Program& program,
    const Tensor& input,
    CoreRange core,
    uint32_t num_cb0_tiles,
    uint32_t num_cb1_tiles,
    uint32_t num_cb0_tilized_tiles,
    uint32_t num_output_tiles,
    uint32_t num_reblock_cb_tiles,
    uint32_t num_writer_output_tiles,
    bool untilize_out,
    tt::DataFormat act_df,
    tt::DataFormat weight_df,
    tt::DataFormat tilized_act_df,
    tt::DataFormat out_df,
    tt::DataFormat bias_df,
    bool weight_width_sliced,
    const Tensor& output,
    uint32_t bias_ntiles,
    bool with_bias,
    bool split_reader,
    bool fp32_dest_acc_en,
    bool packer_l1_acc_en) {
    using namespace CMAKE_UNIQUE_NAMESPACE;
    using namespace tt;
    tt::DataFormat interm0_df =
        packer_l1_acc_en ? (fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b) : out_df;

    uint32_t act_tile_size = tt_metal::detail::TileSize(act_df);
    uint32_t weight_tile_size = tt_metal::detail::TileSize(weight_df);
    uint32_t tilized_act_tile_size = tt_metal::detail::TileSize(tilized_act_df);
    uint32_t out_tile_size = tt_metal::detail::TileSize(out_df);
    uint32_t interm0_single_tile_size = tt_metal::detail::TileSize(interm0_df);

    CBHandle cb_sharded_act = 0;
    if (input.memory_config().is_sharded()) {
        uint32_t num_bytes_for_df = datum_size(act_df);
        auto shard_shape = input.shard_spec().value().shape;
        // 2D-sys-conv already has uint16_t indicies, TODO: do the same for 1D-sys-conv
        TT_FATAL(
            shard_shape[0] <= (1 << 16), "Shard height must be less than 2^16, read pattern indicies are uint16_t");
        CircularBufferConfig cb_sharded_act_config =
            CircularBufferConfig(shard_shape[0] * shard_shape[1] * num_bytes_for_df, {{sharded_act_cb, act_df}})
                .set_page_size(sharded_act_cb, shard_shape[1] * num_bytes_for_df);
        // incoming data is the input cb instead of raw l1/dram addr
        cb_sharded_act_config.set_globally_allocated_address(*input.buffer());
        cb_sharded_act = tt_metal::CreateCircularBuffer(program, core, cb_sharded_act_config);

        // For 1D convs, locally create act matrix in act_cb, which is always ROW_MAJOR BFLOAT16
        // Then, tilize input in compute

        CircularBufferConfig cb_act_config = CircularBufferConfig(num_cb0_tiles * act_tile_size, {{act_cb, act_df}})
                                                    .set_page_size(act_cb, act_tile_size);
        auto cb_act = tt_metal::CreateCircularBuffer(program, core, cb_act_config);
    } else {
        TT_THROW("Input must be sharded!");
    }

    CircularBufferConfig cb_weight_config =
        CircularBufferConfig(num_cb1_tiles * weight_tile_size, {{weight_cb, weight_df}})
            .set_page_size(weight_cb, weight_tile_size);
    auto cb_weight = tt_metal::CreateCircularBuffer(program, core, cb_weight_config);

    // Used for placing tilized activations
    CircularBufferConfig cb_src0_tilized_config =
        CircularBufferConfig(
            num_cb0_tilized_tiles * tilized_act_tile_size, {{tilize_mode_tilized_act_cb, tilized_act_df}})
            .set_page_size(tilize_mode_tilized_act_cb, tilized_act_tile_size);
    auto cb_src0_tilized = tt_metal::CreateCircularBuffer(program, core, cb_src0_tilized_config);

    CBHandle cb_output = 0;
    // Share buffer if same data format
    CoreRangeSet cores(std::set<CoreRange>({core}));

    // breakdown above as separate CBs
    CircularBufferConfig cb_matmul_partials_config = CircularBufferConfig(1 * out_tile_size, {{matmul_partials_cb, out_df}})
        .set_page_size(matmul_partials_cb, out_tile_size);
    auto cb_matmul_partials = tt_metal::CreateCircularBuffer(program, core, cb_matmul_partials_config);

    CircularBufferConfig cb_temp_sum_config = CircularBufferConfig(1 * out_tile_size, {{temp_sum_cb, out_df}})
        .set_page_size(temp_sum_cb, out_tile_size);
    auto cb_temp_sum = tt_metal::CreateCircularBuffer(program, core, cb_temp_sum_config);

    std::map<uint8_t, tt::DataFormat> cb_output_data_format_spec = {
        {out0_cb, out_df}
    };
    CircularBufferConfig cb_output_config = CircularBufferConfig(num_output_tiles * out_tile_size, cb_output_data_format_spec)
        .set_page_size(out0_cb, out_tile_size);

    if (output.is_sharded()) {
        cb_output_config = cb_output_config.set_globally_allocated_address(*output.buffer());
    }
    cb_output = tt_metal::CreateCircularBuffer(program, cores, cb_output_config);

    return {cb_sharded_act, cb_output};
}


}  // namespace conv2d
}  //namespace ttnn::operations::conv
