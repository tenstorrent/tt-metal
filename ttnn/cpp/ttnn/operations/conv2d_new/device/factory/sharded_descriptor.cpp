// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "sharded_descriptor.hpp"

#include <cstdint>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <umd/device/types/xy_pair.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt_stl/assert.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/mesh_workload.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/work_split.hpp>

#include "ttnn/operations/compute_throttle_utils.hpp"
#include "ttnn/operations/conv/conv2d/conv2d_op_program_factory_common.hpp"
#include "ttnn/operations/conv/conv2d/conv2d_utils.hpp"
#include "ttnn/operations/conv/conv2d/device/conv2d_device_operation_types.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"
#include "ttnn/operations/sliding_window/sliding_window.hpp"
#include "ttnn/tensor/types.hpp"

namespace ttnn::prim::conv2d_new_detail {

namespace unary = ttnn::operations::unary;
using ttnn::operations::conv::conv_skip_mcast;
using ttnn::operations::conv::get_num_cores_channels_from_parallel_config;
using ttnn::operations::conv::is_1d_depthwise_conv;
using ttnn::operations::conv::SkipMcast;

// Compute kernel addressing mode divides addresses with 16
constexpr uint32_t COMPUTE_KERNEL_ADDRESS_DIVISOR = 16;

// ---------------------------------------------------------------------------
// Helper struct/function copied from the original factory
// ---------------------------------------------------------------------------

struct ActivationReuseConfig {
    uint32_t image_width_tiles = 0;
    uint32_t image_width_mod_tile = 0;
    uint32_t act_cb_num_tiles_split = 0;
    uint32_t act_cb_num_tiles_split_last = 0;
    uint32_t reuse_window_offset = 0;
    bool readers_process_full_image_widths = false;
    uint32_t tilized_cb_row_offset = 0;
    uint32_t tilized_cb_second_reader_offset = 0;
    // Configuration needed to handle cores with non-meaningful work
    uint32_t num_cores_with_non_meaningful_work = 0;
    std::set<CoreCoord> cores_with_non_meaningful_work;
    bool has_partial_core = false;
    CoreCoord partial_work_core{0, 0};
    uint32_t partial_core_reader_tiles_to_push = 0;
    uint32_t partial_core_writer_remaining_tiles_to_push_to_push = 0;
    bool single_core_processes_multiple_batches = false;
};

ActivationReuseConfig calculate_activation_reuse_params(
    uint32_t output_image_height,
    uint32_t output_image_width,
    uint32_t filter_w,
    uint32_t filter_h,
    uint32_t conv_act_c_read_bytes,
    uint32_t act_block_w_extra_align_bytes,
    uint32_t act_block_h_nsubblocks_split,
    uint32_t act_block_h_nsubblocks_split_last,
    uint32_t tilized_act_tile_size,
    uint32_t act_block_w_ntiles,
    uint32_t act_block_h_ntiles,
    uint32_t single_core_height_ntiles,
    uint32_t total_output_height_ntiles,
    uint32_t padded_total_output_height_ntiles,
    const std::vector<CBInfo>& cb_info,
    bool enable_split_reader,
    const CoreRangeSet& input_cores,
    uint32_t batch) {
    ActivationReuseConfig config;

    // Calculate compile time args needed for activation reuse feature
    config.image_width_tiles = tt::div_up(output_image_width, tt::constants::TILE_HEIGHT);
    config.image_width_mod_tile = output_image_width % tt::constants::TILE_HEIGHT;
    const uint32_t image_width_tile_leftover =
        config.image_width_mod_tile == 0 ? 0 : tt::constants::TILE_HEIGHT - config.image_width_mod_tile;

    // We rely that double buffering is turned off here
    config.act_cb_num_tiles_split = get_cb_info_by_name(cb_info, Conv2dCb::ACT).num_pages;
    if (enable_split_reader) {
        config.act_cb_num_tiles_split_last = get_cb_info_by_name(cb_info, Conv2dCb::ACT_SECOND_READER).num_pages;
    }

    config.reuse_window_offset = filter_w * conv_act_c_read_bytes;
    if (image_width_tile_leftover) {
        config.reuse_window_offset +=
            (filter_w * filter_h * conv_act_c_read_bytes + act_block_w_extra_align_bytes) * image_width_tile_leftover;
    }

    config.readers_process_full_image_widths = act_block_h_nsubblocks_split % config.image_width_tiles == 0 &&
                                               act_block_h_nsubblocks_split_last % config.image_width_tiles == 0 &&
                                               image_width_tile_leftover == 0;

    config.tilized_cb_row_offset = tilized_act_tile_size * act_block_w_ntiles;
    config.tilized_cb_second_reader_offset = tilized_act_tile_size * act_block_h_nsubblocks_split * act_block_w_ntiles;

    uint32_t total_remaining_tiles_to_push = padded_total_output_height_ntiles - total_output_height_ntiles;

    config.num_cores_with_non_meaningful_work = tt::div_up(total_remaining_tiles_to_push, single_core_height_ntiles);

    std::vector<CoreCoord> all_input_cores;
    for (const CoreRange& range : input_cores.ranges()) {
        for (const CoreCoord& core : range) {
            all_input_cores.push_back(core);
        }
    }

    uint32_t partial_core_remaining_tiles = total_remaining_tiles_to_push % single_core_height_ntiles;
    config.has_partial_core = partial_core_remaining_tiles > 0;
    config.partial_core_reader_tiles_to_push = partial_core_remaining_tiles;

    if (partial_core_remaining_tiles > 0) {
        if (enable_split_reader) {
            uint32_t partial_core_act_blocks_to_push = partial_core_remaining_tiles / act_block_h_ntiles;
            config.partial_core_reader_tiles_to_push = partial_core_act_blocks_to_push * act_block_h_nsubblocks_split;
            config.partial_core_writer_remaining_tiles_to_push_to_push =
                partial_core_act_blocks_to_push * act_block_h_nsubblocks_split_last;

            uint32_t partial_core_leftover_tiles = partial_core_remaining_tiles % act_block_h_ntiles;
            if (partial_core_leftover_tiles > act_block_h_nsubblocks_split_last) {
                config.partial_core_writer_remaining_tiles_to_push_to_push += act_block_h_nsubblocks_split_last;
                config.partial_core_reader_tiles_to_push +=
                    partial_core_leftover_tiles - act_block_h_nsubblocks_split_last;
            } else {
                config.partial_core_writer_remaining_tiles_to_push_to_push += partial_core_leftover_tiles;
            }
        }
        uint32_t partial_core_idx = all_input_cores.size() - config.num_cores_with_non_meaningful_work;
        config.partial_work_core = all_input_cores[partial_core_idx];
    }

    uint32_t start_idx = all_input_cores.size() - config.num_cores_with_non_meaningful_work;
    for (uint32_t i = start_idx; i < all_input_cores.size(); i++) {
        config.cores_with_non_meaningful_work.insert(all_input_cores[i]);
    }

    const uint32_t per_core_out_hw = single_core_height_ntiles * tt::constants::TILE_HEIGHT;
    const uint32_t total_batch_out_hw = output_image_height * output_image_width;
    config.single_core_processes_multiple_batches = (batch > 1) && (total_batch_out_hw % per_core_out_hw != 0);

    return config;
}

// ---------------------------------------------------------------------------
// create_descriptor -- builds a ProgramDescriptor declaratively
// ---------------------------------------------------------------------------

tt::tt_metal::ProgramDescriptor Conv2dShardedDescriptorFactory::create_descriptor(
    const Conv2dParams& operation_attributes,
    const Conv2dInputs& tensor_args,
    Tensor& output_tensor,
    tt::tt_metal::Buffer* config_tensor_buffer) {
    using namespace tt::tt_metal;

    ProgramDescriptor desc;

    const auto& a = tensor_args.a;
    const auto& b = tensor_args.b;

    const auto& ashape = ttnn::Shape(operation_attributes.input_tensor_shape);
    const auto& bias = tensor_args.bias;
    const auto& sliding_window_config = operation_attributes.sliding_window_config;

    ttnn::operations::sliding_window::ParallelConfig parallel_config{
        .grid = a.shard_spec().value().grid,
        .shard_scheme = a.memory_config().memory_layout(),
        .shard_orientation = a.shard_spec().value().orientation};

    const auto output_channels = operation_attributes.output_channels;
    const auto groups = operation_attributes.groups;
    const auto untilize_out = operation_attributes.untilize_out;
    const auto has_bias = operation_attributes.has_bias;
    const auto& fused_activation = operation_attributes.activation;
    const auto& parallelization_config = operation_attributes.parallelization_config;
    const auto& block_config = operation_attributes.block_config;
    const auto transpose_mcast = a.shard_spec().value().orientation == ShardOrientation::COL_MAJOR;
    auto& output = output_tensor;
    const auto& compute_kernel_config = operation_attributes.compute_kernel_config;
    const auto enable_act_double_buffer = operation_attributes.enable_act_double_buffer;
    const auto enable_weights_double_buffer = operation_attributes.enable_weights_double_buffer;
    const auto full_inner_dim = operation_attributes.full_inner_dim;
    const auto enable_activation_reuse = operation_attributes.enable_activation_reuse;
    const auto config_tensors_in_dram = operation_attributes.config_tensors_in_dram;
    const auto& force_split_reader = operation_attributes.force_split_reader;

    distributed::MeshDevice* device = a.device();
    TT_FATAL(a.layout() == Layout::ROW_MAJOR, "Conv activation should be in row major layout");
    TT_FATAL(a.memory_config().is_sharded(), "Conv activation must be sharded.");
    TT_FATAL(output_channels <= b.padded_shape()[3], "Invalid weight shape. Incorrect weight tensor.");
    const uint32_t act_block_h_ntiles = block_config.act_block_h_ntiles;
    const uint32_t act_block_w_ntiles = block_config.act_block_w_ntiles;
    const uint32_t weight_block_w_ntiles = parallelization_config.per_core_out_matrix_width_ntile;
    const uint32_t out_block_h_ntiles = parallelization_config.per_core_out_matrix_height_ntile;
    const uint32_t out_subblock_h_ntiles = block_config.out_subblock_h_ntiles;
    const uint32_t out_subblock_w_ntiles = block_config.out_subblock_w_ntiles;

    const SkipMcast skip_mcast = conv_skip_mcast(parallelization_config, a.memory_config().memory_layout());
    const bool skip_activation_mcast = skip_mcast.skip_activation_mcast;
    const bool skip_weights_mcast = skip_mcast.skip_weights_mcast;

    const tt::DataFormat tilized_act_df = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), compute_kernel_config);

    TT_FATAL(
        out_block_h_ntiles >= act_block_h_ntiles,
        "Output block height (in # of tiles) ({}) should be greater than or equal to activation block height (in # of "
        "tiles) ({})",
        out_block_h_ntiles,
        act_block_h_ntiles);

    TT_FATAL(b.layout() == Layout::TILE, "Conv weights should be in tiled layout");
    TT_FATAL(b.padded_shape()[0] == 1, "Conv weight matrix shape is invalid");
    TT_FATAL(b.padded_shape()[1] == 1, "Conv weight matrix shape is invalid");
    uint32_t weight_matrix_height = b.padded_shape()[2];
    uint32_t weight_matrix_width = b.padded_shape()[3];
    uint32_t weight_matrix_width_ntiles = weight_matrix_width / tt::constants::TILE_WIDTH;

    const std::array<uint32_t, 2> shard_shape = a.shard_spec().value().shape;

    const bool block_sharded = a.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED;
    const bool height_sharded = a.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED;

    // parallelization config
    CoreRangeSet input_cores = a.memory_config().shard_spec().value().grid;
    CoreRangeSet output_cores = output.memory_config().shard_spec().value().grid;
    TT_ASSERT(
        input_cores == output_cores || block_sharded,
        "For height sharded convs input and output cores must be the same");

    CoreRangeSet all_cores;
    if (height_sharded) {
        all_cores = CoreRangeSet(CoreRange(
            CoreCoord(0, 0),
            CoreCoord(parallelization_config.grid_size.x - 1, parallelization_config.grid_size.y - 1)));
    } else {
        all_cores = input_cores.merge(output_cores);
    }

    const uint32_t num_cores_x = parallelization_config.grid_size.x;
    const uint32_t num_cores_y = parallelization_config.grid_size.y;
    const uint32_t total_num_cores = all_cores.num_cores();

    const uint32_t per_core_out_matrix_width_ntiles = parallelization_config.per_core_out_matrix_width_ntile;
    const uint32_t per_core_out_matrix_height_ntiles = parallelization_config.per_core_out_matrix_height_ntile;

    const bool slice_inner_dim = (height_sharded && !enable_activation_reuse) || (block_sharded && !full_inner_dim);

    uint32_t conv_act_c_blocks = 1;
    uint32_t out_conv_c_blocks = 1;
    uint32_t input_channels_padded = shard_shape[1];
    if (block_sharded) {
        TT_ASSERT(input_cores.size() == 1, "Block sharded convs should have only one input core range!");
        const uint32_t in_num_cores_x = input_cores.bounding_box().end_coord.x + 1;
        const uint32_t in_num_cores_y = input_cores.bounding_box().end_coord.y + 1;

        conv_act_c_blocks =
            a.shard_spec().value().orientation == ShardOrientation::ROW_MAJOR ? in_num_cores_x : in_num_cores_y;
        out_conv_c_blocks =
            output.shard_spec().value().orientation == ShardOrientation::ROW_MAJOR ? num_cores_x : num_cores_y;
        if (transpose_mcast) {
            TT_FATAL(conv_act_c_blocks == in_num_cores_y, "Expected conv_act_c_blocks to be equal to height of grid");
            input_channels_padded = shard_shape[1] * in_num_cores_y;
        } else {
            TT_FATAL(conv_act_c_blocks == in_num_cores_x, "Expected conv_act_c_blocks to be equal to width of grid");
            input_channels_padded = shard_shape[1] * in_num_cores_x;
        }
    }

    const ttnn::Shape ashape_with_channels_padded({ashape[0], ashape[1], ashape[2], input_channels_padded});
    uint32_t conv_act_size_w = ashape_with_channels_padded[2];
    const uint32_t conv_act_size_c = ashape_with_channels_padded[3];
    const uint32_t filter_h = (uint32_t)sliding_window_config.window_hw.first;
    const uint32_t filter_w = (uint32_t)sliding_window_config.window_hw.second;
    uint32_t pad_w = (uint32_t)sliding_window_config.get_pad_w();
    const uint32_t dilation_h = (uint32_t)sliding_window_config.dilation_hw.first;
    const uint32_t dilation_w = (uint32_t)sliding_window_config.dilation_hw.second;
    const uint32_t stride_h = sliding_window_config.is_transpose ? 1 : (uint32_t)sliding_window_config.stride_hw.first;
    const uint32_t stride_w = sliding_window_config.is_transpose ? 1 : (uint32_t)sliding_window_config.stride_hw.second;

    if (sliding_window_config.is_transpose) {
        auto input_shape = sliding_window_config.get_transposed_full_input_shape();
        conv_act_size_w = input_shape[2];
        pad_w = 0;
    }

    const bool is_conv_1d_depthwise_conv =
        is_1d_depthwise_conv(groups, ashape[3], output_channels, filter_h, filter_w, ashape[1], has_bias);

    const bool enable_split_reader =
        is_split_reader_supported(a.memory_config().memory_layout(), is_conv_1d_depthwise_conv, act_block_h_ntiles) &&
        force_split_reader.value_or(is_split_reader_viable(
            a.memory_config().memory_layout(),
            act_block_h_ntiles,
            input_channels_padded,
            filter_w,
            tt::tt_metal::hal::get_arch(),
            a.dtype(),
            parallelization_config.per_core_out_matrix_width_ntile * block_config.act_block_w_ntiles,
            tt::tile_size(tt::tt_metal::datatype_to_dataformat_converter(b.dtype())),
            dilation_w,
            per_core_out_matrix_height_ntiles / block_config.act_block_h_ntiles,
            act_block_w_ntiles,
            fp32_dest_acc_en,
            output.dtype(),
            enable_activation_reuse));
    log_debug(
        tt::LogOp,
        "force_split_reader: {}, enable_split_reader: {}, num_blocks_act_h: {}, "
        "per_core_out_matrix_height_ntiles: {}, act_block_h_ntiles: {}",
        force_split_reader,
        enable_split_reader,
        per_core_out_matrix_height_ntiles / block_config.act_block_h_ntiles,
        per_core_out_matrix_height_ntiles,
        block_config.act_block_h_ntiles);

    // Activation reuse validation
    if (enable_activation_reuse) {
        TT_FATAL(!block_sharded, "Activation data reuse is not supported for block sharded");
        TT_FATAL(dilation_h == 1 && dilation_w == 1, "Activation data reuse is not supported for dilation > 1");
        TT_FATAL(stride_h == 1 && stride_w == 1, "Activation data reuse is not supported for stride > 1");
        TT_FATAL(enable_split_reader, "Activation data reuse requires split reader to be on");
    }

    TT_FATAL(input_channels_padded >= ashape[3], "Incorrect padding of input channels!");
    TT_FATAL(
        input_channels_padded % 8 == 0,
        "Expected input channels to be padded for 16 byte alignment in L1 ({} % 16 != 0)",
        input_channels_padded);

    const uint32_t act_matrix_height_ntiles = out_block_h_ntiles * parallelization_config.num_cores_nhw;
    const uint32_t act_matrix_height = act_matrix_height_ntiles * tt::constants::TILE_HEIGHT;

    if (has_bias) {
        if (is_conv_1d_depthwise_conv) {
            TT_THROW("Bias is not supported for depthwise conv1d");
        }
        TT_FATAL(bias.has_value(), "Bias tensor must be provided when has_bias is true");
        TT_FATAL(bias.value().buffer() != nullptr, "Bias tensor buffer must not be null");
        auto bias_shape_without_padding = bias.value().logical_shape();
        TT_FATAL(bias_shape_without_padding[0] == 1, "Bias should have batch == 1");
    }

    TT_FATAL(
        weight_matrix_height % tt::constants::TILE_HEIGHT == 0, "Height of weight matrix needs to be divisible by 32");
    TT_FATAL(
        weight_matrix_width % tt::constants::TILE_WIDTH == 0, "Width of weight matrix needs to be divisible by 32");

    TT_FATAL(
        a.storage_type() == StorageType::DEVICE && b.storage_type() == StorageType::DEVICE,
        "Operands to large matmul need to be on device!");
    TT_FATAL(a.device() == b.device(), "Operands to conv need to be on the same device!");
    TT_FATAL(
        a.buffer() != nullptr && b.buffer() != nullptr, "Operands to conv need to be allocated in buffers on device!");
    if (has_bias) {
        TT_FATAL(bias.value().storage_type() == StorageType::DEVICE, "Bias should be on device");
        TT_FATAL(bias.value().device() == a.device(), "Bias should be on the same device as act tensor");
    }

    TT_FATAL(
        act_matrix_height_ntiles % act_block_h_ntiles == 0,
        "act_matrix_height_ntiles {} should be divisible by act_block_h_ntiles {}",
        act_matrix_height_ntiles,
        act_block_h_ntiles);
    TT_FATAL(
        weight_matrix_width_ntiles % weight_block_w_ntiles == 0,
        "weight_matrix_width_ntiles {} should be divisible by weight_block_w_ntiles {}",
        weight_matrix_width_ntiles,
        weight_block_w_ntiles);
    TT_FATAL(
        act_matrix_height_ntiles % out_block_h_ntiles == 0,
        "act_matrix_height_ntiles {} should be divisible by out_block_h_ntiles {}",
        act_matrix_height_ntiles,
        out_block_h_ntiles);

    const uint32_t num_blocks_act_h = act_matrix_height_ntiles / act_block_h_ntiles;
    const uint32_t num_blocks_act_w = slice_inner_dim ? filter_h : 1;
    const uint32_t num_blocks_weight_w = weight_matrix_width_ntiles / weight_block_w_ntiles;

    // act block info
    uint32_t act_block_h_datums = act_matrix_height / num_blocks_act_h;

    uint32_t act_block_h_nsubblocks_split = block_config.act_block_h_ntiles;
    uint32_t act_block_h_nsubblocks_split_last = 0;
    if (enable_split_reader) {
        act_block_h_nsubblocks_split_last = block_config.act_block_h_ntiles / 2;
        act_block_h_nsubblocks_split = block_config.act_block_h_ntiles - act_block_h_nsubblocks_split_last;
    }
    uint32_t act_block_h_datums_split = act_block_h_nsubblocks_split * tt::constants::TILE_HEIGHT;
    (void)act_block_h_datums_split;  // Used in create_mesh_workload for config tensor generation

    uint32_t act_block_num_tiles_split = act_block_h_nsubblocks_split * act_block_w_ntiles;
    uint32_t act_block_num_tiles_split_last = act_block_h_nsubblocks_split_last * act_block_w_ntiles;

    // weight block info
    uint32_t weight_block_w_datums = weight_matrix_width / num_blocks_weight_w;
    TT_FATAL(
        weight_block_w_ntiles % out_subblock_w_ntiles == 0,
        "weight_block_w_ntiles {} should be divisible by weight_block_w_ntiles {}",
        weight_block_w_ntiles,
        out_subblock_w_ntiles);
    uint32_t weight_num_subblocks = weight_block_w_ntiles / out_subblock_w_ntiles;
    uint32_t weight_block_h_ntiles = is_conv_1d_depthwise_conv ? act_block_h_ntiles * filter_w : act_block_w_ntiles;
    uint32_t weight_block_num_tiles = weight_block_w_ntiles * weight_block_h_ntiles;

    uint32_t output_channels_padded_to_tile_width = tt::round_up(output_channels, tt::constants::TILE_WIDTH);
    TT_FATAL(
        output_channels_padded_to_tile_width <= weight_matrix_width,
        "output_channels_padded_to_tile_width {} should be less than or equal to weight_matrix_width {}",
        output_channels_padded_to_tile_width,
        weight_matrix_width);
    uint32_t last_block_width_datums = (output_channels_padded_to_tile_width % weight_block_w_datums == 0)
                                           ? weight_block_w_datums
                                           : (output_channels_padded_to_tile_width % weight_block_w_datums);
    TT_FATAL(
        last_block_width_datums % tt::constants::TILE_WIDTH == 0,
        "last_block_width_datums {} should be divisible by TILE_WIDTH",
        last_block_width_datums);

    TT_FATAL(output.is_sharded(), "Output buffer must be sharded!");

    // out
    uint32_t out_subblock_num_tiles = out_subblock_h_ntiles * out_subblock_w_ntiles;
    TT_FATAL(out_subblock_num_tiles <= 8, "Need to ensure that matmul partials fit in dst");

    TT_FATAL(
        act_block_h_ntiles % out_subblock_h_ntiles == 0,
        "act_block_h_ntiles {} should be divisible by out_subblock_h_ntiles {}",
        act_block_h_ntiles,
        out_subblock_h_ntiles);

    const uint32_t act_num_subblocks = act_block_h_ntiles / out_subblock_h_ntiles;
    const uint32_t act_block_num_tiles = act_block_h_ntiles * act_block_w_ntiles;
    const uint32_t act_subblock_h_ntiles = out_subblock_h_ntiles;
    const uint32_t act_subblock_num_tiles = act_subblock_h_ntiles * act_block_w_ntiles;

    const uint32_t in0_num_blocks_w = conv_act_c_blocks * num_blocks_act_w;

    // weight
    const uint32_t weight_dram_addr = b.buffer()->address();

    // bias
    tt::tt_metal::Buffer* bias_buffer = nullptr;
    uint32_t bias_dram_addr = 0;
    uint32_t bias_ntiles = 0;
    if (has_bias) {
        bias_buffer = bias.value().buffer();
        bias_dram_addr = bias_buffer->address();
        bias_ntiles = bias.value().padded_shape()[3] / tt::constants::TILE_WIDTH;
    }

    const uint32_t window_outer = num_blocks_act_w;
    const uint32_t window_inner = block_sharded ? filter_h : filter_h * filter_w / num_blocks_act_w;
    log_debug(tt::LogOp, "window_outer: {}, window_inner: {}", window_outer, window_inner);

    TT_FATAL(
        weight_matrix_width_ntiles % per_core_out_matrix_width_ntiles == 0,
        "weight_matrix_width_ntiles {} should be divisible by per_core_out_matrix_width_ntiles {}",
        weight_matrix_width_ntiles,
        per_core_out_matrix_width_ntiles);
    TT_FATAL(
        per_core_out_matrix_width_ntiles % weight_block_w_ntiles == 0,
        "per_core_out_matrix_width_ntiles {} should be divisible by weight_block_w_ntiles {}",
        per_core_out_matrix_width_ntiles,
        weight_block_w_ntiles);
    uint32_t num_blocks_weight_w_per_core = per_core_out_matrix_width_ntiles / weight_block_w_ntiles;
    if (height_sharded) {
        TT_FATAL(
            num_blocks_weight_w_per_core == num_blocks_weight_w,
            "num_blocks_weight_w_per_core {} should be equal to num_blocks_weight_w {}",
            num_blocks_weight_w_per_core,
            num_blocks_weight_w);
    }
    uint32_t num_weight_slices_width = weight_matrix_width_ntiles / per_core_out_matrix_width_ntiles;
    uint32_t total_num_cores_per_weight_slice = 0;
    if (block_sharded) {
        if (transpose_mcast) {
            TT_FATAL(
                num_cores_y % num_weight_slices_width == 0,
                "num_cores_y {} should be divisible by num_weight_slices_width {}",
                num_cores_y,
                num_weight_slices_width);
            uint32_t num_cores_y_per_weight_slice_width = num_cores_y / num_weight_slices_width;
            total_num_cores_per_weight_slice = num_cores_y_per_weight_slice_width * num_cores_x;
        } else {
            TT_FATAL(
                num_cores_x % num_weight_slices_width == 0,
                "num_cores_x {} should be divisible by num_weight_slices_width {}",
                num_cores_x,
                num_weight_slices_width);
            uint32_t num_cores_x_per_weight_slice_width = num_cores_x / num_weight_slices_width;
            total_num_cores_per_weight_slice = num_cores_x_per_weight_slice_width * num_cores_y;
        }
        TT_FATAL(
            total_num_cores_per_weight_slice * per_core_out_matrix_height_ntiles == act_matrix_height_ntiles,
            "total_num_cores_per_weight_slice {} * per_core_out_matrix_height_ntiles {} should be equal to "
            "act_matrix_height_ntiles {}",
            total_num_cores_per_weight_slice,
            per_core_out_matrix_height_ntiles,
            act_matrix_height_ntiles);
    } else {
        TT_FATAL(
            num_cores_y % num_weight_slices_width == 0,
            "num_cores_y {} should be divisible by num_weight_slices_width {}",
            num_cores_y,
            num_weight_slices_width);
        uint32_t num_cores_y_per_weight_slice_width = num_cores_y / num_weight_slices_width;
        total_num_cores_per_weight_slice = num_cores_y_per_weight_slice_width * num_cores_x;
        TT_FATAL(
            total_num_cores * per_core_out_matrix_height_ntiles >= act_matrix_height_ntiles,
            "total_num_cores {} * per_core_out_matrix_height_ntiles {} should be greater than or equal to "
            "act_matrix_height_ntiles {}",
            total_num_cores,
            per_core_out_matrix_height_ntiles,
            act_matrix_height_ntiles);
    }
    TT_FATAL(
        per_core_out_matrix_height_ntiles % act_block_h_ntiles == 0,
        "per_core_out_matrix_height_ntiles {} should be divisible by act_block_h_ntiles {}",
        per_core_out_matrix_height_ntiles,
        act_block_h_ntiles);
    uint32_t num_blocks_act_h_per_core = per_core_out_matrix_height_ntiles / act_block_h_ntiles;

    TT_FATAL(
        act_matrix_height_ntiles % per_core_out_matrix_height_ntiles == 0,
        "Activation matrix height in tiles ({}) must be divisible by per-core output matrix height in tiles ({})",
        act_matrix_height_ntiles,
        per_core_out_matrix_height_ntiles);
    uint32_t total_noop_cores = total_num_cores_per_weight_slice - parallelization_config.num_cores_nhw;
    uint32_t total_active_num_cores = parallelization_config.num_cores_nhw * num_weight_slices_width;
    TT_FATAL(!block_sharded || total_noop_cores == 0, "All cores should be active for block sharded convs");

    if (has_bias) {
        TT_FATAL(
            bias_ntiles == weight_matrix_width_ntiles,
            "Bias tiles ({}) must equal weight matrix width in tiles ({})",
            bias_ntiles,
            weight_matrix_width_ntiles);
    }
    uint32_t bias_ntiles_per_core = bias_ntiles / num_weight_slices_width;

    uint32_t conv_act_c_read_bytes = conv_act_size_c * a.element_size() / conv_act_c_blocks;
    uint32_t act_block_w_extra_align_bytes =
        !slice_inner_dim
            ? (tt::round_up(shard_shape[1] * filter_h * filter_w, tt::constants::TILE_WIDTH) -
               (shard_shape[1] * filter_h * filter_w)) *
                  a.element_size()
            : (tt::round_up(shard_shape[1] * filter_w, tt::constants::TILE_WIDTH) - (shard_shape[1] * filter_w)) *
                  a.element_size();
    const uint32_t act_block_w_extra_align_scalars = act_block_w_extra_align_bytes / a.element_size();
    const bool needs_act_block_zero_out =
        act_block_w_extra_align_scalars % 16 != 0 && tt::tt_metal::is_block_float(output.dtype());

    const uint32_t tilized_act_tile_size = tt::tile_size(tilized_act_df);

    const bool packer_l1_acc_en = determine_packer_l1_acc(packer_l1_acc, has_bias, in0_num_blocks_w);
    const uint32_t batch = sliding_window_config.get_output_shape()[0];
    const uint32_t output_image_width = sliding_window_config.get_output_shape()[2];
    const uint32_t output_image_height = sliding_window_config.get_output_shape()[1];
    const uint32_t total_output_height_ntiles =
        (batch * output_image_height * output_image_width) / tt::constants::TILE_HEIGHT;

    Conv2dConfig conv_config = Conv2dConfig{
        .weights_dtype = b.dtype(),
        .config_tensors_in_dram = config_tensors_in_dram,
        .shard_layout = a.memory_config().memory_layout(),
        .output_layout = (untilize_out ? Layout::ROW_MAJOR : Layout::TILE),
        .enable_act_double_buffer = enable_act_double_buffer,
        .enable_weights_double_buffer = enable_weights_double_buffer,
        .enable_activation_reuse = enable_activation_reuse,
        .force_split_reader = force_split_reader};
    std::vector<CBInfo> cb_info = get_cb_info(
        compute_kernel_config,
        block_config,
        parallelization_config,
        b.padded_shape(),
        {filter_h, filter_w},
        {sliding_window_config.input_hw.first, sliding_window_config.input_hw.second},
        {dilation_h, dilation_w},
        conv_config,
        a.dtype(),
        output.dtype(),
        shard_shape,
        output_image_width,
        has_bias,
        is_conv_1d_depthwise_conv,
        skip_activation_mcast,
        input_channels_padded);

    // Adjust READER_INDICES CB page size based on config tensor buffer
    if (config_tensor_buffer != nullptr) {
        if (config_tensors_in_dram) {
            TT_FATAL(
                access_cb_info_by_name(cb_info, Conv2dCb::READER_INDICES).page_size >=
                    config_tensor_buffer->page_size(),
                "CB page size {} should be greater than the config tensor page size {}",
                access_cb_info_by_name(cb_info, Conv2dCb::READER_INDICES).page_size,
                config_tensor_buffer->page_size());
        } else {
            access_cb_info_by_name(cb_info, Conv2dCb::READER_INDICES).page_size = config_tensor_buffer->page_size();
        }
    }

    // ---------------------------------------------------------------
    // Build CBDescriptors from cb_info
    // ---------------------------------------------------------------
    // First pass: assign sequential cb_index and build CBDescriptors for non-overlapped CBs
    // We also track the mapping from Conv2dCb name to CBDescriptor index for overlapped CBs
    uint32_t cb_index = 0;
    std::unordered_map<uint32_t, uint32_t> cb_info_idx_to_desc_idx;  // cb_info index -> desc.cbs index

    for (uint32_t ci = 0; ci < static_cast<uint32_t>(cb_info.size()); ++ci) {
        auto& cb = cb_info[ci];
        if (cb.num_pages == 0) {
            continue;
        }
        if (cb.overlapped_by_cb.has_value()) {
            // Will be merged in second pass
            cb.index = cb_index++;  // Still need a sequential index for the CB
            continue;
        }

        cb.index = cb_index++;

        // Determine the buffer pointer for globally allocated CBs
        Buffer* buffer = nullptr;
        if (cb.is_globally_allocated) {
            if (cb.name == Conv2dCb::ACT_SHARDED) {
                buffer = a.buffer();
            } else if (cb.name == Conv2dCb::OUT || cb.name == Conv2dCb::MATMUL_PARTIALS) {
                buffer = output.buffer();
            } else if (cb.name == Conv2dCb::READER_INDICES) {
                buffer = config_tensor_buffer;
            }
        }

        CBDescriptor cb_desc;
        cb_desc.total_size = cb.num_pages * cb.page_size;
        cb_desc.core_ranges = all_cores;
        cb_desc.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(cb.index),
            .data_format = cb.data_format,
            .page_size = cb.page_size,
        });
        cb_desc.buffer = buffer;

        cb_info_idx_to_desc_idx[ci] = static_cast<uint32_t>(desc.cbs.size());
        desc.cbs.push_back(std::move(cb_desc));
    }

    // Second pass: merge overlapped CBs into the CBDescriptor of the CB they overlap with
    for (uint32_t ci = 0; ci < static_cast<uint32_t>(cb_info.size()); ++ci) {
        auto& cb = cb_info[ci];
        if (cb.num_pages == 0 || !cb.overlapped_by_cb.has_value()) {
            continue;
        }

        // Find the CBInfo that this CB is overlapped by
        const CBInfo& overlapping_cb = get_cb_info_by_name(cb_info, cb.overlapped_by_cb.value());
        // Find cb_info index of overlapping CB
        uint32_t overlapping_ci = 0;
        for (uint32_t j = 0; j < static_cast<uint32_t>(cb_info.size()); ++j) {
            if (cb_info[j].name == overlapping_cb.name && cb_info[j].num_pages > 0 &&
                !cb_info[j].overlapped_by_cb.has_value()) {
                overlapping_ci = j;
                break;
            }
        }

        auto it = cb_info_idx_to_desc_idx.find(overlapping_ci);
        TT_FATAL(it != cb_info_idx_to_desc_idx.end(), "Could not find CBDescriptor for overlapping CB");
        uint32_t desc_idx = it->second;

        // Set the overlapped CB's index to match the overlapping CB's index
        cb.index = overlapping_cb.index;

        // Add format descriptor to the overlapping CB's descriptor
        desc.cbs[desc_idx].format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(cb.index),
            .data_format = cb.data_format,
            .page_size = cb.page_size,
        });
    }

    // ---------------------------------------------------------------
    // Compute physical core coordinates needed for runtime args
    // ---------------------------------------------------------------
    const uint32_t in_num_cores_x = input_cores.bounding_box().end_coord.x + 1;
    const uint32_t in_num_cores_y = input_cores.bounding_box().end_coord.y + 1;

    const CoreCoord top_left_core = {(std::size_t)0, (std::size_t)0};
    const CoreCoord top_left_core_plus_one = {(std::size_t)1, (std::size_t)1};
    const CoreCoord bottom_right_core = {(std::size_t)in_num_cores_x - 1, (std::size_t)in_num_cores_y - 1};
    const CoreCoord top_left_core_physical = device->worker_core_from_logical_core(top_left_core);
    const CoreCoord top_left_core_plus_one_physical = device->worker_core_from_logical_core(top_left_core_plus_one);
    const CoreCoord bottom_right_core_physical = device->worker_core_from_logical_core(bottom_right_core);

    // ---------------------------------------------------------------
    // Semaphores
    // ---------------------------------------------------------------
    CoreRangeSet mcast_sender_cores = CoreRangeSet(CoreRange(top_left_core, top_left_core));
    CoreRangeSet mcast_receiver_cores;
    uint32_t weights_mcast_sender_semaphore_id = 0;
    uint32_t weights_mcast_receiver_semaphore_id = 0;
    uint32_t act_mcast_sender_semaphore_id = 0;
    uint32_t act_mcast_receiver_semaphore_id = 0;
    uint32_t act_split_reader_reserve_done_semaphore_id = 0;
    uint32_t act_split_reader_write_done_semaphore_id = 0;

    const bool populate_skipped_work_cores =
        enable_split_reader && block_sharded && input_cores.num_cores() > output_cores.num_cores();

    const bool overlap_act_cb =
        get_cb_info_by_name(cb_info, Conv2dCb::ACT_ROW_MAJOR_BFLOAT16).overlapped_by_cb.has_value();
    const bool split_reader_cb_shared = enable_split_reader && overlap_act_cb && block_sharded;

    uint32_t next_semaphore_id = 0;

    if (block_sharded) {
        const CoreCoord out_bottom_right_core = {(std::size_t)num_cores_x - 1, (std::size_t)num_cores_y - 1};
        if (transpose_mcast) {
            mcast_sender_cores = CoreRangeSet(CoreRange(top_left_core, CoreCoord(0, num_cores_y - 1)));
            if (!skip_weights_mcast) {
                mcast_receiver_cores = CoreRange(CoreCoord(1, 0), out_bottom_right_core);
            }
        } else {
            mcast_sender_cores = CoreRangeSet(CoreRange(top_left_core, CoreCoord(num_cores_x - 1, 0)));
            if (!skip_weights_mcast) {
                mcast_receiver_cores = CoreRange(CoreCoord(0, 1), out_bottom_right_core);
            }
        }
        if (populate_skipped_work_cores) {
            mcast_sender_cores = input_cores.subtract(mcast_receiver_cores);
        }

        // act_mcast semaphores on all_cores
        act_mcast_sender_semaphore_id = next_semaphore_id++;
        desc.semaphores.push_back(SemaphoreDescriptor{
            .id = act_mcast_sender_semaphore_id, .core_ranges = all_cores, .initial_value = INVALID});
        act_mcast_receiver_semaphore_id = next_semaphore_id++;
        desc.semaphores.push_back(SemaphoreDescriptor{
            .id = act_mcast_receiver_semaphore_id, .core_ranges = all_cores, .initial_value = INVALID});

        if (split_reader_cb_shared) {
            weights_mcast_sender_semaphore_id = next_semaphore_id++;
            desc.semaphores.push_back(SemaphoreDescriptor{
                .id = weights_mcast_sender_semaphore_id, .core_ranges = all_cores, .initial_value = INVALID});
            weights_mcast_receiver_semaphore_id = next_semaphore_id++;
            desc.semaphores.push_back(SemaphoreDescriptor{
                .id = weights_mcast_receiver_semaphore_id, .core_ranges = all_cores, .initial_value = INVALID});
            act_split_reader_reserve_done_semaphore_id = next_semaphore_id++;
            desc.semaphores.push_back(SemaphoreDescriptor{
                .id = act_split_reader_reserve_done_semaphore_id, .core_ranges = all_cores, .initial_value = INVALID});
            act_split_reader_write_done_semaphore_id = next_semaphore_id++;
            desc.semaphores.push_back(SemaphoreDescriptor{
                .id = act_split_reader_write_done_semaphore_id, .core_ranges = all_cores, .initial_value = INVALID});
        } else {
            weights_mcast_sender_semaphore_id = next_semaphore_id++;
            desc.semaphores.push_back(SemaphoreDescriptor{
                .id = weights_mcast_sender_semaphore_id, .core_ranges = output_cores, .initial_value = INVALID});
            weights_mcast_receiver_semaphore_id = next_semaphore_id++;
            desc.semaphores.push_back(SemaphoreDescriptor{
                .id = weights_mcast_receiver_semaphore_id, .core_ranges = output_cores, .initial_value = INVALID});
        }
    } else {
        // 1D mcast
        if (!skip_weights_mcast) {
            mcast_receiver_cores = all_cores.subtract(mcast_sender_cores);
            weights_mcast_sender_semaphore_id = next_semaphore_id++;
            desc.semaphores.push_back(SemaphoreDescriptor{
                .id = weights_mcast_sender_semaphore_id, .core_ranges = output_cores, .initial_value = INVALID});
            weights_mcast_receiver_semaphore_id = next_semaphore_id++;
            desc.semaphores.push_back(SemaphoreDescriptor{
                .id = weights_mcast_receiver_semaphore_id, .core_ranges = output_cores, .initial_value = INVALID});
        }
    }

    const bool partials_cb_uses_output = get_cb_info_by_name(cb_info, Conv2dCb::MATMUL_PARTIALS).is_globally_allocated;
    log_debug(tt::LogOp, "partials_cb_uses_output: {}", partials_cb_uses_output);

    // ---------------------------------------------------------------
    // Kernel paths
    // ---------------------------------------------------------------
    std::string reader_kernel;
    std::string compute_kernel = "ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/conv_bmm_tilize.cpp";
    std::string writer_mcast_sender_kernel =
        "ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/"
        "reader_writer_tiled_out_1d_mcast_sender_conv_weights_tiled_col_to_rm_blocks.cpp";
    std::string writer_mcast_receiver_kernel =
        "ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/"
        "reader_writer_tiled_out_1d_mcast_receiver_conv_weights_tiled_col_to_rm_blocks.cpp";

    if (!is_conv_1d_depthwise_conv && block_sharded) {
        reader_kernel =
            "ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/"
            "reader_conv_activations_2d_mcast_padded_with_halo_3x3_weights_v2.cpp";
        writer_mcast_sender_kernel =
            "ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/"
            "writer_tiled_out_2d_mcast_sender_conv_weights_tiled_col_to_rm_blocks.cpp";
        writer_mcast_receiver_kernel =
            "ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/"
            "writer_tiled_out_2d_mcast_receiver_conv_weights_tiled_col_to_rm_blocks.cpp";
    } else if (is_conv_1d_depthwise_conv) {
        compute_kernel = "ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/compute_depthwise_conv1d.cpp";
        reader_kernel = "ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/reader_depthwise_conv1d.cpp";
    } else {
        reader_kernel =
            "ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/"
            "reader_conv_activations_padded_with_halo_3x3_weights_v2.cpp";
    }

    uint32_t reader_arg_act_block_h_datums = (enable_split_reader ? act_block_h_datums_split : act_block_h_datums);
    TT_FATAL(reader_arg_act_block_h_datums % 2 == 0, "2 Indices are packed in one uint32_t word.");

    ActivationReuseConfig activation_reuse_config;
    if (enable_activation_reuse) {
        activation_reuse_config = calculate_activation_reuse_params(
            output_image_height,
            output_image_width,
            filter_w,
            filter_h,
            conv_act_c_read_bytes,
            act_block_w_extra_align_bytes,
            act_block_h_nsubblocks_split,
            act_block_h_nsubblocks_split_last,
            tilized_act_tile_size,
            act_block_w_ntiles,
            act_block_h_ntiles,
            out_block_h_ntiles,
            total_output_height_ntiles,
            act_matrix_height_ntiles,
            cb_info,
            enable_split_reader,
            input_cores,
            batch);
    }

    // ---------------------------------------------------------------
    // Reader compile time args
    // ---------------------------------------------------------------
    std::vector<uint32_t> reader_compile_time_args = {
        (uint32_t)dilation_h,
        (uint32_t)dilation_w,
        (uint32_t)stride_w,
        (uint32_t)conv_act_c_read_bytes,
        (uint32_t)window_outer,
        (uint32_t)window_inner,
        (uint32_t)(enable_split_reader && !split_reader_cb_shared ? act_block_num_tiles_split : act_block_num_tiles),
        (uint32_t)filter_h,
        (uint32_t)filter_w,
        (uint32_t)conv_act_size_w + (pad_w),
        (uint32_t)act_block_w_extra_align_bytes,
        (uint32_t)num_blocks_act_h_per_core,
        (uint32_t)act_block_num_tiles,
        (uint32_t)conv_act_c_blocks,
        (uint32_t)(transpose_mcast ? num_cores_y - 1 : num_cores_x - 1),
        (uint32_t)(transpose_mcast ? num_cores_y - 1 : num_cores_x - 1),
        (uint32_t)act_mcast_sender_semaphore_id,
        (uint32_t)act_mcast_receiver_semaphore_id,
        (uint32_t)tilized_act_tile_size,
        (uint32_t)(transpose_mcast ? 1 : 0),
        (uint32_t)needs_act_block_zero_out,
        get_cb_info_by_name(cb_info, Conv2dCb::ACT).index,
        get_cb_info_by_name(cb_info, Conv2dCb::ACT_SHARDED).index,
        get_cb_info_by_name(cb_info, Conv2dCb::READER_INDICES).index,
        get_cb_info_by_name(cb_info, Conv2dCb::ACT_TILIZED).index,
        get_cb_info_by_name(cb_info, Conv2dCb::ACT_ROW_MAJOR_BFLOAT16).index,
        get_cb_info_by_name(cb_info, Conv2dCb::L1_ARRAY).index,
        (uint32_t)enable_split_reader,
        (uint32_t)enable_activation_reuse};

    // ---------------------------------------------------------------
    // Reader / writer defines
    // ---------------------------------------------------------------
    KernelDescriptor::Defines reader_defines_vec;
    KernelDescriptor::Defines writer_defines_vec;
    KernelDescriptor::Defines writer_mcast_sender_defines_vec;
    KernelDescriptor::Defines compute_defines_vec;

    // Helper to convert map to vector of pairs (for KernelDescriptor::Defines)
    std::map<std::string, std::string> reader_defines;
    std::map<std::string, std::string> writer_defines;
    std::map<std::string, std::string> writer_mcast_sender_defines;
    std::map<std::string, std::string> compute_defines;

    if (config_tensors_in_dram) {
        reader_defines["CONFIG_TENSOR_IN_DRAM"] = "1";
        writer_defines["CONFIG_TENSOR_IN_DRAM"] = "1";
        writer_mcast_sender_defines["CONFIG_TENSOR_IN_DRAM"] = "1";
        if (config_tensor_buffer != nullptr) {
            reader_compile_time_args.push_back(config_tensor_buffer->address());
            reader_compile_time_args.push_back(config_tensor_buffer->page_size());
            tt::tt_metal::TensorAccessorArgs(config_tensor_buffer).append_to(reader_compile_time_args);
        } else {
            reader_compile_time_args.push_back(0);
            reader_compile_time_args.push_back(0);
            reader_compile_time_args.push_back(0);
        }
    } else {
        reader_compile_time_args.push_back(0);
        reader_compile_time_args.push_back(0);
        reader_compile_time_args.push_back(0);
    }

    if (enable_activation_reuse) {
        std::vector<uint32_t> activation_reuse_args = {
            activation_reuse_config.act_cb_num_tiles_split,
            act_block_w_ntiles,
            static_cast<uint32_t>(activation_reuse_config.readers_process_full_image_widths),
            activation_reuse_config.image_width_tiles,
            output_image_width,
            activation_reuse_config.reuse_window_offset,
            static_cast<uint32_t>(activation_reuse_config.num_cores_with_non_meaningful_work > 0),
            static_cast<uint32_t>(activation_reuse_config.single_core_processes_multiple_batches)};

        reader_compile_time_args.insert(
            reader_compile_time_args.end(), activation_reuse_args.begin(), activation_reuse_args.end());
    } else if (height_sharded) {
        std::vector<uint32_t> activation_reuse_dummy_args(8, 0);
        reader_compile_time_args.insert(
            reader_compile_time_args.end(), activation_reuse_dummy_args.begin(), activation_reuse_dummy_args.end());
    }

    if (split_reader_cb_shared) {
        reader_compile_time_args.push_back(static_cast<uint32_t>(split_reader_cb_shared));
        reader_compile_time_args.push_back(act_split_reader_reserve_done_semaphore_id);
        reader_compile_time_args.push_back(act_split_reader_write_done_semaphore_id);
    } else if (block_sharded) {
        reader_compile_time_args.push_back(static_cast<uint32_t>(split_reader_cb_shared));
        reader_compile_time_args.push_back(0);
        reader_compile_time_args.push_back(0);
    }
    if (skip_activation_mcast) {
        reader_defines["SKIP_MCAST"] = "1";
    }
    if (skip_weights_mcast) {
        writer_mcast_sender_defines["SKIP_MCAST"] = "1";
    }
    bool pack_relu = fused_activation.has_value() && fused_activation.value().op_type == unary::UnaryOpType::RELU;
    if (fused_activation.has_value() && !pack_relu) {
        compute_defines.merge(ttnn::operations::unary::utils::get_defines(
            fused_activation.value().op_type, fused_activation.value().params, "ACTIVATION", "i"));
    }
    if (enable_split_reader) {
        compute_defines["SPLIT_READER"] = "1";
        reader_defines["SPLIT_READER"] = "1";
        writer_mcast_sender_defines["SPLIT_READER"] = "1";
        writer_defines["SPLIT_READER"] = "1";
    }

    if (enable_activation_reuse) {
        reader_defines["ACTIVATION_REUSE"] = "1";
        writer_mcast_sender_defines["ACTIVATION_REUSE"] = "1";
        writer_defines["ACTIVATION_REUSE"] = "1";
    }

    ttnn::operations::compute_throttle_utils::throttle_mm_perf(
        device->arch(), output_cores.num_cores(), compute_defines, ttnn::get_throttle_level(compute_kernel_config));

    for (auto elem : compute_defines) {
        log_trace(tt::LogOp, "compute_defines: {} = {}", elem.first, elem.second);
    }

    // ---------------------------------------------------------------
    // Writer compile time args
    // ---------------------------------------------------------------
    std::vector<uint32_t> writer_compile_time_args = {
        get_cb_info_by_name(cb_info, Conv2dCb::WEIGHTS).index,
        get_cb_info_by_name(cb_info, Conv2dCb::BIAS).index,
        (uint32_t)(bias_buffer == nullptr ? 0 : (bias_buffer->buffer_type() == BufferType::DRAM ? 1 : 0)),
        get_cb_info_by_name(cb_info, (split_reader_cb_shared ? Conv2dCb::ACT : Conv2dCb::ACT_SECOND_READER)).index,
        get_cb_info_by_name(cb_info, Conv2dCb::ACT_SHARDED).index,
        get_cb_info_by_name(cb_info, Conv2dCb::READER_INDICES).index,
        num_blocks_act_w,
        weight_block_num_tiles,
        conv_act_c_blocks,
        weight_block_h_ntiles,
        weight_block_w_ntiles,
        weight_matrix_width_ntiles,
        weight_matrix_width_ntiles * weight_block_h_ntiles,
        weight_block_w_ntiles,

        // bias
        bias_ntiles_per_core,

        num_blocks_act_h_per_core,
        num_blocks_weight_w_per_core,
        out_conv_c_blocks,
        (uint32_t)has_bias,
        (uint32_t)enable_split_reader,
        (uint32_t)(enable_activation_reuse && height_sharded)};

    std::vector<uint32_t> split_reader_args = {
        (uint32_t)act_block_num_tiles_split_last,
        (uint32_t)conv_act_c_read_bytes,
        (uint32_t)filter_w,
        (uint32_t)(conv_act_size_w + pad_w),
        (uint32_t)act_block_w_extra_align_bytes,
        (uint32_t)needs_act_block_zero_out,
        (uint32_t)dilation_h,
        (uint32_t)dilation_w,
        (uint32_t)stride_w,
        (uint32_t)filter_h};

    if (block_sharded) {
        split_reader_args.push_back(static_cast<uint32_t>(split_reader_cb_shared));
    }
    if (split_reader_cb_shared) {
        const tt::DataFormat img2col_cb_df = get_cb_info_by_name(cb_info, Conv2dCb::ACT_ROW_MAJOR_BFLOAT16).data_format;

        const uint32_t img2col_cb_tile_size = tt::tile_size(img2col_cb_df);
        const uint32_t act_cb_id_stride =
            skip_activation_mcast ? 1 : 1 + get_num_cores_channels_from_parallel_config(parallel_config);
        const uint32_t act_cb_block_cnt = get_cb_info_by_name(cb_info, Conv2dCb::ACT).num_pages /
                                          (act_block_num_tiles_split + act_block_num_tiles_split_last);

        const bool second_writer_two_addr = (act_cb_block_cnt > 1) && (act_cb_id_stride % 2 == 1);
        const uint32_t act_write_offset = act_block_num_tiles_split * img2col_cb_tile_size;
        const uint32_t act_write_offset_last =
            second_writer_two_addr
                ? (act_block_num_tiles_split_last + 2 * act_block_num_tiles_split) * img2col_cb_tile_size
                : act_write_offset;
        split_reader_args.push_back(act_split_reader_reserve_done_semaphore_id);
        split_reader_args.push_back(act_split_reader_write_done_semaphore_id);
        split_reader_args.push_back(act_write_offset);
        split_reader_args.push_back(act_write_offset_last);
    } else if (block_sharded) {
        split_reader_args.push_back(0);
        split_reader_args.push_back(0);
        split_reader_args.push_back(0);
        split_reader_args.push_back(0);
    }

    if (enable_activation_reuse && height_sharded) {
        std::vector<uint32_t> activation_reuse_args = {
            activation_reuse_config.act_cb_num_tiles_split_last,
            act_block_w_ntiles,
            static_cast<uint32_t>(activation_reuse_config.readers_process_full_image_widths),
            activation_reuse_config.image_width_tiles,
            output_image_width,
            activation_reuse_config.reuse_window_offset,
            static_cast<uint32_t>(activation_reuse_config.num_cores_with_non_meaningful_work > 0),
            static_cast<uint32_t>(activation_reuse_config.single_core_processes_multiple_batches)};
        split_reader_args.insert(split_reader_args.end(), activation_reuse_args.begin(), activation_reuse_args.end());
    } else if (height_sharded) {
        std::vector<uint32_t> activation_reuse_dummy_args(8, 0);
        split_reader_args.insert(
            split_reader_args.end(), activation_reuse_dummy_args.begin(), activation_reuse_dummy_args.end());
    }
    writer_compile_time_args.insert(writer_compile_time_args.end(), split_reader_args.begin(), split_reader_args.end());
    tt::tt_metal::TensorAccessorArgs(b.buffer()).append_to(writer_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(bias_buffer).append_to(writer_compile_time_args);

    // ---------------------------------------------------------------
    // Compute kernel args
    // ---------------------------------------------------------------
    const bool check_skip_compute = input_cores != output_cores;

    std::vector<uint32_t> compute_kernel_args = {
        act_block_w_ntiles,
        act_num_subblocks,
        act_block_num_tiles,
        act_subblock_num_tiles,
        enable_split_reader ? act_block_h_ntiles : act_subblock_h_ntiles * act_num_subblocks,
        weight_num_subblocks,
        weight_block_num_tiles,
        weight_block_w_ntiles,

        num_blocks_act_h_per_core,
        in0_num_blocks_w,
        num_blocks_weight_w_per_core,

        out_subblock_h_ntiles,
        out_subblock_w_ntiles,
        out_subblock_num_tiles,

        height_sharded,
        untilize_out,

        bias_ntiles_per_core,

        get_cb_info_by_name(cb_info, Conv2dCb::BIAS).index,
        get_cb_info_by_name(cb_info, Conv2dCb::ACT).index,
        get_cb_info_by_name(cb_info, Conv2dCb::WEIGHTS).index,
        get_cb_info_by_name(cb_info, Conv2dCb::ACT_ROW_MAJOR_BFLOAT16).index,
        get_cb_info_by_name(cb_info, Conv2dCb::ACT_SECOND_READER).index,
        get_cb_info_by_name(cb_info, Conv2dCb::MATMUL_PARTIALS).index,
        get_cb_info_by_name(cb_info, Conv2dCb::ACT_TILIZED).index,
        get_cb_info_by_name(cb_info, Conv2dCb::OUT).index,
        get_cb_info_by_name(cb_info, Conv2dCb::TEMP_SUM).index,
        partials_cb_uses_output,
        conv_act_c_blocks,
        check_skip_compute,
        pack_relu,
        weight_block_w_ntiles <= 8,  // packer_untilize
        packer_l1_acc_en,
        has_bias,
        enable_split_reader,
        enable_activation_reuse};

    if (enable_activation_reuse) {
        compute_kernel_args.push_back(activation_reuse_config.image_width_tiles);
        compute_kernel_args.push_back(activation_reuse_config.reuse_window_offset / COMPUTE_KERNEL_ADDRESS_DIVISOR);
        compute_kernel_args.push_back(activation_reuse_config.tilized_cb_row_offset / COMPUTE_KERNEL_ADDRESS_DIVISOR);
        compute_kernel_args.push_back(
            activation_reuse_config.tilized_cb_second_reader_offset / COMPUTE_KERNEL_ADDRESS_DIVISOR);
    } else {
        std::vector<uint32_t> activation_reuse_dummy_args = {0, 0, 0, 0};
        compute_kernel_args.insert(
            compute_kernel_args.end(), activation_reuse_dummy_args.begin(), activation_reuse_dummy_args.end());
    }
    compute_kernel_args.push_back(static_cast<uint32_t>(split_reader_cb_shared));

    // ---------------------------------------------------------------
    // Convert defines from map to vector<pair<string,string>>
    // ---------------------------------------------------------------
    for (const auto& [k, v] : reader_defines) {
        reader_defines_vec.emplace_back(k, v);
    }
    for (const auto& [k, v] : writer_defines) {
        writer_defines_vec.emplace_back(k, v);
    }
    for (const auto& [k, v] : writer_mcast_sender_defines) {
        writer_mcast_sender_defines_vec.emplace_back(k, v);
    }
    for (const auto& [k, v] : compute_defines) {
        compute_defines_vec.emplace_back(k, v);
    }

    // ---------------------------------------------------------------
    // Build KernelDescriptors
    // ---------------------------------------------------------------
    const tt::tt_metal::NOC writer_mcast_noc = tt::tt_metal::detail::preferred_noc_for_dram_read(device->arch());
    const tt::tt_metal::NOC reader_noc =
        writer_mcast_noc == tt::tt_metal::NOC::NOC_0 ? tt::tt_metal::NOC::NOC_1 : tt::tt_metal::NOC::NOC_0;

    // Kernel 0: writer_mcast_sender
    KernelDescriptor writer_mcast_sender_desc;
    writer_mcast_sender_desc.kernel_source = writer_mcast_sender_kernel;
    writer_mcast_sender_desc.core_ranges = mcast_sender_cores;
    writer_mcast_sender_desc.compile_time_args = writer_compile_time_args;
    writer_mcast_sender_desc.defines = writer_mcast_sender_defines_vec;
    writer_mcast_sender_desc.config = DataMovementConfigDescriptor{
        .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
        .noc = writer_mcast_noc,
    };

    // Kernel 1: writer_mcast_receiver (may be unused if skip_weights_mcast)
    KernelDescriptor writer_mcast_receiver_desc;
    bool has_mcast_receiver = !skip_weights_mcast;
    if (has_mcast_receiver) {
        writer_mcast_receiver_desc.kernel_source = writer_mcast_receiver_kernel;
        writer_mcast_receiver_desc.core_ranges = mcast_receiver_cores;
        writer_mcast_receiver_desc.compile_time_args = writer_compile_time_args;
        writer_mcast_receiver_desc.defines = writer_defines_vec;
        writer_mcast_receiver_desc.config = DataMovementConfigDescriptor{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = writer_mcast_noc,
        };
    }

    // Kernel 2 (or 1 if no receiver): reader
    KernelDescriptor reader_desc;
    reader_desc.kernel_source = reader_kernel;
    reader_desc.core_ranges = height_sharded ? input_cores : all_cores;
    reader_desc.compile_time_args = reader_compile_time_args;
    reader_desc.defines = reader_defines_vec;
    reader_desc.config = DataMovementConfigDescriptor{
        .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
        .noc = reader_noc,
    };

    // Kernel 3 (or 2 if no receiver): compute
    KernelDescriptor compute_desc;
    compute_desc.kernel_source = compute_kernel;
    compute_desc.core_ranges = height_sharded ? input_cores : all_cores;
    compute_desc.compile_time_args = compute_kernel_args;
    compute_desc.defines = compute_defines_vec;
    compute_desc.config = ComputeConfigDescriptor{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
    };

    // ---------------------------------------------------------------
    // Runtime args: helper lambda for mcast args
    // ---------------------------------------------------------------
    auto setup_mcast_args = [&](bool is_noc_0, uint32_t start_x, uint32_t start_y, uint32_t end_x, uint32_t end_y) {
        return is_noc_0 ? std::vector<uint32_t>{start_x, start_y, end_x, end_y}
                        : std::vector<uint32_t>{end_x, end_y, start_x, start_y};
    };

    // ---------------------------------------------------------------
    // Reader runtime args
    // ---------------------------------------------------------------
    if (block_sharded) {
        std::vector<uint32_t> act_mcast_noc_y;
        if (transpose_mcast) {
            act_mcast_noc_y.reserve(in_num_cores_y);
            for (uint32_t core_idx_y = 0; core_idx_y < in_num_cores_y; ++core_idx_y) {
                act_mcast_noc_y.push_back(device->worker_core_from_logical_core({0, core_idx_y}).y);
            }
        } else {
            act_mcast_noc_y.reserve(in_num_cores_x);
            for (int32_t core_idx_x = 0; core_idx_x < static_cast<int32_t>(in_num_cores_x); ++core_idx_x) {
                act_mcast_noc_y.push_back(device->worker_core_from_logical_core({core_idx_x, 0}).x);
            }
        }

        const CoreCoord out_bottom_right_core = {(std::size_t)num_cores_x - 1, (std::size_t)num_cores_y - 1};
        const CoreCoord out_bottom_right_core_physical = device->worker_core_from_logical_core(out_bottom_right_core);
        const bool reader_is_noc_0 = reader_noc == tt::tt_metal::NOC::NOC_0;

        for (const CoreRange& core_range : all_cores.ranges()) {
            for (const CoreCoord& core : core_range) {
                const bool is_receiver_core = output_cores.contains(core);
                const bool is_sender_core = input_cores.contains(core);
                std::vector<uint32_t> reader_rt_args;
                if (transpose_mcast) {
                    CoreCoord bottom_core = {(std::size_t)core.x, (std::size_t)num_cores_y - 1};
                    CoreCoord bottom_core_physical = device->worker_core_from_logical_core(bottom_core);

                    reader_rt_args = setup_mcast_args(
                        reader_is_noc_0,
                        bottom_core_physical.x,
                        top_left_core_physical.y,
                        bottom_core_physical.x,
                        bottom_core_physical.y);

                    reader_rt_args.push_back(core.y);
                    reader_rt_args.push_back(bottom_core_physical.x);
                } else {
                    CoreCoord core_physical = device->worker_core_from_logical_core(core);

                    reader_rt_args = setup_mcast_args(
                        reader_is_noc_0,
                        top_left_core_physical.x,
                        core_physical.y,
                        out_bottom_right_core_physical.x,
                        core_physical.y);
                    reader_rt_args.push_back(core.x);
                    reader_rt_args.push_back(core_physical.y);
                }
                reader_rt_args.push_back(static_cast<uint32_t>(is_receiver_core));
                reader_rt_args.push_back(static_cast<uint32_t>(is_sender_core));
                reader_rt_args.push_back(transpose_mcast ? core.x : core.y);
                reader_rt_args.insert(reader_rt_args.end(), act_mcast_noc_y.begin(), act_mcast_noc_y.end());
                reader_desc.runtime_args.emplace_back(core, std::move(reader_rt_args));
            }
        }
    } else {
        uint32_t core_index = 0;
        for (const CoreRange& core_range : input_cores.ranges()) {
            for (const CoreCoord& core : core_range) {
                uint32_t reader_remaining_tiles_to_push = 0;
                if (enable_activation_reuse) {
                    if (activation_reuse_config.has_partial_core && core == activation_reuse_config.partial_work_core) {
                        reader_remaining_tiles_to_push = activation_reuse_config.partial_core_reader_tiles_to_push;
                    } else if (activation_reuse_config.cores_with_non_meaningful_work.contains(core)) {
                        reader_remaining_tiles_to_push = act_block_h_nsubblocks_split;
                    }
                }
                std::vector<uint32_t> reader_rt_args{core_index, reader_remaining_tiles_to_push};
                reader_desc.runtime_args.emplace_back(core, std::move(reader_rt_args));
                core_index++;
            }
        }
    }

    // ---------------------------------------------------------------
    // Writer mcast sender runtime args
    // ---------------------------------------------------------------
    for (const CoreRange& core_range : mcast_sender_cores.ranges()) {
        for (const CoreCoord& core : core_range) {
            if (populate_skipped_work_cores && !output_cores.contains(core)) {
                std::vector<uint32_t> args = std::vector<uint32_t>(14, 0);
                args[10] = weights_mcast_sender_semaphore_id;
                args[11] = weights_mcast_receiver_semaphore_id;
                args[12] = static_cast<uint32_t>(true);
                args[13] = static_cast<uint32_t>(true);
                writer_mcast_sender_desc.runtime_args.emplace_back(core, std::move(args));
                continue;
            }
            uint32_t weight_slice_i = ((block_sharded && transpose_mcast) || !block_sharded) ? core.y : core.x;

            uint32_t out_start_tile_id_w = weight_slice_i * per_core_out_matrix_width_ntiles;
            uint32_t bias_tile_offset = out_start_tile_id_w;

            TT_FATAL(
                bias_tile_offset < bias_ntiles || !has_bias,
                "bias_tile_offset {} should be less than bias_ntiles {}",
                bias_tile_offset,
                bias_ntiles);
            std::vector<uint32_t> sender_rt_args = {
                weight_dram_addr, bias_dram_addr, out_start_tile_id_w, bias_tile_offset};
            if (block_sharded) {
                const bool is_sender_core = input_cores.contains(core);
                if (transpose_mcast) {
                    CoreCoord right_core = {(std::size_t)num_cores_x - 1, (std::size_t)core.y};
                    CoreCoord right_core_physical = device->worker_core_from_logical_core(right_core);
                    TT_FATAL(core.x == 0, "Expected core.x to be 0 for sender in 2D mcast setup");

                    std::vector<uint32_t> mcast_coords = setup_mcast_args(
                        writer_mcast_noc == tt::tt_metal::NOC::NOC_0,
                        top_left_core_plus_one_physical.x,
                        right_core_physical.y,
                        bottom_right_core_physical.x,
                        right_core_physical.y);

                    sender_rt_args.insert(sender_rt_args.end(), mcast_coords.begin(), mcast_coords.end());

                    sender_rt_args.insert(
                        sender_rt_args.end(),
                        {num_cores_x - 1,
                         num_cores_x - 1,
                         weights_mcast_sender_semaphore_id,
                         weights_mcast_receiver_semaphore_id,
                         static_cast<uint32_t>(is_sender_core),
                         static_cast<uint32_t>(false)});
                    writer_mcast_sender_desc.runtime_args.emplace_back(core, std::move(sender_rt_args));
                } else {
                    CoreCoord top_core = {(std::size_t)core.x, 0};
                    CoreCoord top_core_physical = device->worker_core_from_logical_core(top_core);
                    TT_FATAL(core.y == 0, "Expected core.y to be 0 for sender in 2D mcast setup");
                    std::vector<uint32_t> mcast_coords = setup_mcast_args(
                        writer_mcast_noc == tt::tt_metal::NOC::NOC_0,
                        top_core_physical.x,
                        top_left_core_plus_one_physical.y,
                        top_core_physical.x,
                        bottom_right_core_physical.y);

                    sender_rt_args.insert(sender_rt_args.end(), mcast_coords.begin(), mcast_coords.end());
                    sender_rt_args.insert(
                        sender_rt_args.end(),
                        {num_cores_y - 1,
                         num_cores_y - 1,
                         weights_mcast_sender_semaphore_id,
                         weights_mcast_receiver_semaphore_id,
                         static_cast<uint32_t>(is_sender_core),
                         static_cast<uint32_t>(false)});
                    writer_mcast_sender_desc.runtime_args.emplace_back(core, std::move(sender_rt_args));
                }
            } else {
                // 1D multicast setup
                std::vector<uint32_t> mcast_coords = setup_mcast_args(
                    writer_mcast_noc == tt::tt_metal::NOC::NOC_0,
                    top_left_core_physical.x,
                    top_left_core_physical.y,
                    bottom_right_core_physical.x,
                    bottom_right_core_physical.y);

                sender_rt_args.insert(sender_rt_args.end(), mcast_coords.begin(), mcast_coords.end());
                sender_rt_args.insert(
                    sender_rt_args.end(),
                    {total_active_num_cores - 1,
                     total_num_cores - 1,
                     weights_mcast_sender_semaphore_id,
                     weights_mcast_receiver_semaphore_id});
                if (enable_activation_reuse) {
                    uint32_t writer_remaining_tiles_to_push = 0;
                    if (activation_reuse_config.has_partial_core && core == activation_reuse_config.partial_work_core) {
                        writer_remaining_tiles_to_push =
                            activation_reuse_config.partial_core_writer_remaining_tiles_to_push_to_push;
                    } else if (activation_reuse_config.cores_with_non_meaningful_work.contains(core)) {
                        writer_remaining_tiles_to_push = act_block_h_nsubblocks_split_last;
                    }
                    sender_rt_args.push_back(writer_remaining_tiles_to_push);
                }
                writer_mcast_sender_desc.runtime_args.emplace_back(core, std::move(sender_rt_args));
            }
        }
    }

    // ---------------------------------------------------------------
    // Writer mcast receiver runtime args
    // ---------------------------------------------------------------
    if (has_mcast_receiver) {
        for (const CoreRange& core_range : mcast_receiver_cores.ranges()) {
            auto create_receiver_args = [&](uint32_t sender_noc_x, uint32_t sender_noc_y) {
                return std::vector<uint32_t>{
                    sender_noc_x, sender_noc_y, weights_mcast_sender_semaphore_id, weights_mcast_receiver_semaphore_id};
            };

            for (const CoreCoord& core : core_range) {
                std::vector<uint32_t> receiver_args;
                if (block_sharded) {
                    if (transpose_mcast) {
                        CoreCoord right_core = {(std::size_t)num_cores_x - 1, (std::size_t)core.y};
                        CoreCoord right_core_physical = device->worker_core_from_logical_core(right_core);
                        receiver_args = create_receiver_args(top_left_core_physical.x, right_core_physical.y);
                    } else {
                        CoreCoord top_core = {(std::size_t)core.x, 0};
                        CoreCoord top_core_physical = device->worker_core_from_logical_core(top_core);
                        receiver_args = create_receiver_args(top_core_physical.x, top_left_core_physical.y);
                    }
                    const bool is_sender_core = input_cores.contains(core);
                    receiver_args.push_back(static_cast<uint32_t>(is_sender_core));
                } else {
                    bool is_no_op_core = !input_cores.contains(core);
                    receiver_args = std::vector<uint32_t>{
                        static_cast<uint32_t>(is_no_op_core),
                        top_left_core_physical.x,
                        top_left_core_physical.y,
                        weights_mcast_sender_semaphore_id,
                        weights_mcast_receiver_semaphore_id};
                    if (enable_activation_reuse) {
                        uint32_t writer_remaining_tiles_to_push = 0;
                        if (activation_reuse_config.has_partial_core &&
                            core == activation_reuse_config.partial_work_core) {
                            writer_remaining_tiles_to_push =
                                activation_reuse_config.partial_core_writer_remaining_tiles_to_push_to_push;
                        } else if (activation_reuse_config.cores_with_non_meaningful_work.contains(core)) {
                            writer_remaining_tiles_to_push = act_block_h_nsubblocks_split_last;
                        }
                        receiver_args.push_back(writer_remaining_tiles_to_push);
                    }
                }
                writer_mcast_receiver_desc.runtime_args.emplace_back(core, std::move(receiver_args));
            }
        }
    }

    // ---------------------------------------------------------------
    // Compute runtime args (skip_compute flag for block sharded with mismatched grids)
    // ---------------------------------------------------------------
    if (input_cores != output_cores) {
        CoreCoord bottom_right_core_out = output_cores.bounding_box().end_coord;
        uint32_t end_coord_x = bottom_right_core_out.x;
        uint32_t end_coord_y = bottom_right_core_out.y;
        for (const CoreRange& range : all_cores.ranges()) {
            for (const CoreCoord& core : range) {
                bool skip_compute = transpose_mcast ? core.y > end_coord_y : core.x > end_coord_x;
                compute_desc.runtime_args.emplace_back(
                    core, KernelDescriptor::CoreRuntimeArgs{static_cast<uint32_t>(skip_compute)});
            }
        }
    }

    // ---------------------------------------------------------------
    // Push kernels in order: sender=0, receiver=1 (if present), reader=next, compute=next
    // ---------------------------------------------------------------
    desc.kernels.push_back(std::move(writer_mcast_sender_desc));
    if (has_mcast_receiver) {
        desc.kernels.push_back(std::move(writer_mcast_receiver_desc));
    }
    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(compute_desc));

    return desc;
}

// ---------------------------------------------------------------------------
// create_mesh_workload
// ---------------------------------------------------------------------------

Conv2dShardedDescriptorFactory::cached_mesh_workload_t Conv2dShardedDescriptorFactory::create_mesh_workload(
    const Conv2dParams& operation_attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const Conv2dInputs& tensor_args,
    Tensor& tensor_return_value) {
    using namespace tt::tt_metal;
    namespace sliding_window = ttnn::operations::sliding_window;

    tt::tt_metal::distributed::MeshWorkload mesh_workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;

    // ---------------------------------------------------------------
    // Collect buffer pointers for slot detection
    // ---------------------------------------------------------------
    std::vector<tt::tt_metal::Buffer*> bufs;
    auto collect = [&](const auto& obj) {
        tt::stl::reflection::visit_object_of_type<Tensor>(
            [&](const Tensor& t) {
                if (t.buffer()) {
                    bufs.push_back(t.buffer());
                }
            },
            obj);
    };
    collect(tensor_args);
    collect(tensor_return_value);

    std::unordered_map<uint32_t, uint16_t> addr_to_id;
    std::unordered_map<tt::tt_metal::Buffer*, uint16_t> buf_to_id;
    bool has_addr_collision = false;
    for (uint16_t i = 0; i < static_cast<uint16_t>(bufs.size()); ++i) {
        auto [it, inserted] = addr_to_id.emplace(bufs[i]->address(), i);
        if (!inserted) {
            has_addr_collision = true;
        }
        buf_to_id.emplace(bufs[i], i);
    }

    // ---------------------------------------------------------------
    // Create config tensor (sliding window reader indices)
    // This must happen outside create_descriptor.
    // ---------------------------------------------------------------
    const auto& a = tensor_args.a;
    const auto& sliding_window_config = operation_attributes.sliding_window_config;
    const auto& block_config = operation_attributes.block_config;
    const auto& parallelization_config = operation_attributes.parallelization_config;
    const bool config_tensors_in_dram = operation_attributes.config_tensors_in_dram;

    const bool block_sharded = a.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED;
    const bool height_sharded = a.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED;

    const std::array<uint32_t, 2> shard_shape = a.shard_spec().value().shape;
    uint32_t input_channels_padded = shard_shape[1];
    if (block_sharded) {
        const bool transpose_mcast = a.shard_spec().value().orientation == ShardOrientation::COL_MAJOR;
        CoreRangeSet input_cores = a.memory_config().shard_spec().value().grid;
        const uint32_t in_num_cores_x = input_cores.bounding_box().end_coord.x + 1;
        const uint32_t in_num_cores_y = input_cores.bounding_box().end_coord.y + 1;
        if (transpose_mcast) {
            input_channels_padded = shard_shape[1] * in_num_cores_y;
        } else {
            input_channels_padded = shard_shape[1] * in_num_cores_x;
        }
    }

    const auto& ashape = ttnn::Shape(operation_attributes.input_tensor_shape);
    const uint32_t output_channels = operation_attributes.output_channels;
    const uint32_t groups = operation_attributes.groups;
    const bool has_bias = operation_attributes.has_bias;
    const auto& force_split_reader = operation_attributes.force_split_reader;
    const auto enable_activation_reuse = operation_attributes.enable_activation_reuse;

    const uint32_t filter_h = (uint32_t)sliding_window_config.window_hw.first;
    const uint32_t filter_w = (uint32_t)sliding_window_config.window_hw.second;
    const uint32_t dilation_w = (uint32_t)sliding_window_config.dilation_hw.second;
    const uint32_t stride_w = sliding_window_config.is_transpose ? 1 : (uint32_t)sliding_window_config.stride_hw.second;

    const uint32_t act_block_h_ntiles = block_config.act_block_h_ntiles;
    const uint32_t act_block_w_ntiles = block_config.act_block_w_ntiles;
    const uint32_t per_core_out_matrix_height_ntiles = parallelization_config.per_core_out_matrix_height_ntile;

    const bool is_conv_1d_depthwise_conv_flag =
        is_1d_depthwise_conv(groups, ashape[3], output_channels, filter_h, filter_w, ashape[1], has_bias);

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(a.device()->arch(), operation_attributes.compute_kernel_config);

    const auto& b = tensor_args.b;
    const bool enable_split_reader =
        is_split_reader_supported(
            a.memory_config().memory_layout(), is_conv_1d_depthwise_conv_flag, act_block_h_ntiles) &&
        force_split_reader.value_or(is_split_reader_viable(
            a.memory_config().memory_layout(),
            act_block_h_ntiles,
            input_channels_padded,
            filter_w,
            tt::tt_metal::hal::get_arch(),
            a.dtype(),
            parallelization_config.per_core_out_matrix_width_ntile * block_config.act_block_w_ntiles,
            tt::tile_size(tt::tt_metal::datatype_to_dataformat_converter(b.dtype())),
            dilation_w,
            per_core_out_matrix_height_ntiles / block_config.act_block_h_ntiles,
            act_block_w_ntiles,
            fp32_dest_acc_en,
            tensor_return_value.dtype(),
            enable_activation_reuse));

    // Compute act_block_h_datums for split reader
    const uint32_t out_block_h_ntiles_local = per_core_out_matrix_height_ntiles;
    const uint32_t act_matrix_height_ntiles = out_block_h_ntiles_local * parallelization_config.num_cores_nhw;
    const uint32_t act_matrix_height = act_matrix_height_ntiles * tt::constants::TILE_HEIGHT;
    // slice_inner_dim used only for force_split_reader check below
    [[maybe_unused]] const bool slice_inner_dim =
        (height_sharded && !enable_activation_reuse) || (block_sharded && !operation_attributes.full_inner_dim);
    const uint32_t num_blocks_act_h = act_matrix_height_ntiles / act_block_h_ntiles;
    uint32_t act_block_h_datums = act_matrix_height / num_blocks_act_h;

    uint32_t act_block_h_nsubblocks_split = block_config.act_block_h_ntiles;
    uint32_t act_block_h_nsubblocks_split_last = 0;
    if (enable_split_reader) {
        act_block_h_nsubblocks_split_last = block_config.act_block_h_ntiles / 2;
        act_block_h_nsubblocks_split = block_config.act_block_h_ntiles - act_block_h_nsubblocks_split_last;
    }
    uint32_t act_block_h_datums_split = act_block_h_nsubblocks_split * tt::constants::TILE_HEIGHT;
    uint32_t act_block_h_datums_split_last = act_block_h_nsubblocks_split_last * tt::constants::TILE_HEIGHT;

    // Generate sliding window metadata
    std::vector<uint32_t> op_trace_metadata =
        ttnn::operations::sliding_window::generate_op_trace_metadata(sliding_window_config);
    std::vector<sliding_window::ShardBoundary> shard_boundaries =
        ttnn::operations::sliding_window::generate_shard_boundaries(sliding_window_config);

    std::vector<std::vector<uint16_t>> conv_sharded_input_top_left_indices =
        ttnn::operations::sliding_window::generate_sliding_window_op_config(
            op_trace_metadata,
            shard_boundaries,
            stride_w,
            true,
            enable_split_reader ? act_block_h_datums_split : act_block_h_datums,
            enable_split_reader ? act_block_h_datums_split_last : 0);

    // Create sharded config tensors
    sliding_window::ParallelConfig input_parallel_config = {
        .grid = a.memory_config().shard_spec().value().grid,
        .shard_scheme = a.memory_config().memory_layout(),
        .shard_orientation = a.memory_config().shard_spec().value().orientation,
    };

    Tensor conv_reader_indices_tensor = ttnn::operations::sliding_window::construct_on_host_config_tensor(
        conv_sharded_input_top_left_indices, input_parallel_config, config_tensors_in_dram);
    conv_reader_indices_tensor = ttnn::operations::sliding_window::move_config_tensor_to_device(
        conv_reader_indices_tensor, input_parallel_config, block_sharded, a.device(), config_tensors_in_dram);

    log_trace(tt::LogOp, "Conv2D Config Tensor : {}", conv_reader_indices_tensor);
    const tt::tt_metal::DeviceStorage& conv_reader_indices_storage = conv_reader_indices_tensor.device_storage();
    tt::tt_metal::Buffer* config_tensor_buffer = conv_reader_indices_storage.get_buffer();

    // ---------------------------------------------------------------
    // Build programs for each mesh coordinate range
    // ---------------------------------------------------------------
    for (const auto& range : tensor_coords.ranges()) {
        auto descriptor =
            create_descriptor(operation_attributes, tensor_args, tensor_return_value, config_tensor_buffer);

        // Scan runtime args for address slots
        std::vector<AddressSlot> address_slots;
        if (!has_addr_collision) {
            for (uint32_t k = 0; k < static_cast<uint32_t>(descriptor.kernels.size()); ++k) {
                for (const auto& [core, args] : descriptor.kernels[k].runtime_args) {
                    for (uint32_t i = 0; i < static_cast<uint32_t>(args.size()); ++i) {
                        auto it = addr_to_id.find(args[i]);
                        if (it != addr_to_id.end()) {
                            address_slots.push_back({k, core, i, it->second});
                        }
                    }
                }
            }
        }

        tt::tt_metal::Program program{descriptor};

        // Scan for CB slots
        std::vector<CBSlot> cb_slots;
        auto program_cbs = program.circular_buffers();
        for (uint32_t ci = 0; ci < static_cast<uint32_t>(descriptor.cbs.size()); ++ci) {
            if (descriptor.cbs[ci].buffer) {
                auto it = buf_to_id.find(descriptor.cbs[ci].buffer);
                if (it != buf_to_id.end()) {
                    cb_slots.push_back({program_cbs[ci]->id(), it->second});
                }
            }
        }

        mesh_workload.add_program(range, std::move(program));

        shared_variables[range] = shared_variables_t{
            .address_slots = std::move(address_slots),
            .cb_slots = std::move(cb_slots),
            .conv_reader_indices_storage = conv_reader_indices_storage,
        };
    }

    return cached_mesh_workload_t{std::move(mesh_workload), std::move(shared_variables)};
}

// ---------------------------------------------------------------------------
// override_runtime_arguments
// ---------------------------------------------------------------------------

void Conv2dShardedDescriptorFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const Conv2dParams& /*operation_attributes*/,
    const Conv2dInputs& tensor_args,
    Tensor& tensor_return_value) {
    for (auto& [coordinate_range, program] : cached_workload.workload.get_programs()) {
        const auto& sv = cached_workload.shared_variables.at(coordinate_range);

        // Collect buffer pointers in the same deterministic order as create_mesh_workload
        std::vector<tt::tt_metal::Buffer*> bufs;
        tt::stl::reflection::visit_object_of_type<Tensor>(
            [&](const Tensor& t) {
                if (t.buffer()) {
                    bufs.push_back(t.buffer());
                }
            },
            tensor_args);
        tt::stl::reflection::visit_object_of_type<Tensor>(
            [&](const Tensor& t) {
                if (t.buffer()) {
                    bufs.push_back(t.buffer());
                }
            },
            tensor_return_value);

        // Patch runtime arg address slots
        for (const auto& slot : sv.address_slots) {
            auto& args = tt::tt_metal::GetRuntimeArgs(program, slot.kernel_handle, slot.core);
            args[slot.arg_index] = bufs[slot.buffer_id]->address();
        }

        // Patch dynamic CB addresses
        for (const auto& cb_slot : sv.cb_slots) {
            tt::tt_metal::UpdateDynamicCircularBufferAddress(program, cb_slot.cb_handle, *bufs[cb_slot.buffer_id]);
        }
    }
}

}  // namespace ttnn::prim::conv2d_new_detail
