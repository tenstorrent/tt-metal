// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <tt_stl/assert.hpp>
#include "tt-metalium/circular_buffer_config.hpp"
#include "tt-metalium/core_coord.hpp"
#include "tt-metalium/kernel_types.hpp"
#include "tt-metalium/tt_backend_api_types.hpp"
#include "ttnn/operations/conv/conv2d/conv2d_op_program_factory_common.hpp"
#include "ttnn/operations/conv/conv2d/conv2d_utils.hpp"
#include "ttnn/operations/conv/conv2d/device/conv2d_op_sharded_program_factory.hpp"
#include "ttnn/operations/conv/conv2d/device/conv2d_device_operation_types.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_utils.hpp"
#include "ttnn/operations/matmul/device/config/matmul_auto_tuner.hpp"
#include "ttnn/operations/sliding_window/sliding_window.hpp"
#include "ttnn/tensor/types.hpp"
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/workload_descriptor.hpp>
#include "ttnn/operations/compute_throttle_utils.hpp"

namespace ttnn::prim {

namespace unary = ttnn::operations::unary;
using ttnn::operations::conv::conv_skip_mcast;
using ttnn::operations::conv::get_num_cores_channels_from_parallel_config;
using ttnn::operations::conv::is_1d_depthwise_conv;
using ttnn::operations::conv::should_coalesce_1d_depthwise_conv_reads;
using ttnn::operations::conv::SkipMcast;

// Compute kernel addressing mode divides addresses with 16
constexpr uint32_t COMPUTE_KERNEL_ADDRESS_DIVISOR = 16;

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
    // TODO(sjovic): avoid this assumption
    config.act_cb_num_tiles_split = get_cb_info_by_name(cb_info, Conv2dCb::ACT).num_pages;
    if (enable_split_reader) {
        config.act_cb_num_tiles_split_last = get_cb_info_by_name(cb_info, Conv2dCb::ACT_SECOND_READER).num_pages;
    }

    // Number of bytes to move the CB read pointer when passing on to the new output image row;
    // We need to skip the first kernel_w*in_channels_padded elements
    config.reuse_window_offset = filter_w * conv_act_c_read_bytes;
    // In case the output image width is not a multiple of the tile height, we need to skip the additional elements
    // we read to fill in the tile height
    if (image_width_tile_leftover) {
        config.reuse_window_offset +=
            (filter_w * filter_h * conv_act_c_read_bytes + act_block_w_extra_align_bytes) * image_width_tile_leftover;
    }

    // Precompute happy path for the feature - if each reader processes full image rows only, we can skip many if
    // conditions in the kernel. There are two cases which can affect this:
    // - shards are split in such way that one shard ends in the middle of the image width
    // - shards contain full image widths only, but split reader splits shard in the middle of the image width
    // - output image width is not a multiple of the tile height, so we need to process more than one image width at
    // once
    config.readers_process_full_image_widths = act_block_h_nsubblocks_split % config.image_width_tiles == 0 &&
                                               act_block_h_nsubblocks_split_last % config.image_width_tiles == 0 &&
                                               image_width_tile_leftover == 0;

    // Compute kernel interleaves tilizing data coming from two readers so it needs to calculate the address in the
    // tilized CB
    config.tilized_cb_row_offset = tilized_act_tile_size * act_block_w_ntiles;
    config.tilized_cb_second_reader_offset = tilized_act_tile_size * act_block_h_nsubblocks_split * act_block_w_ntiles;

    // Last cores sometime have less work to do, but we still need to push the same number of tiles
    // to avoid blocking compute kernels; Here we compute how many cores will be pushing the remaining tiles
    uint32_t total_remaining_tiles_to_push = padded_total_output_height_ntiles - total_output_height_ntiles;

    config.num_cores_with_non_meaningful_work = tt::div_up(total_remaining_tiles_to_push, single_core_height_ntiles);

    std::vector<CoreCoord> all_input_cores;
    for (const CoreRange& range : input_cores.ranges()) {
        for (const CoreCoord& core : range) {
            all_input_cores.push_back(core);
        }
    }

    // Calculate tiles for the partial core (the one core that may have less than full work)
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

    // Put all cores with non-meaningful work to the set
    uint32_t start_idx = all_input_cores.size() - config.num_cores_with_non_meaningful_work;
    for (uint32_t i = start_idx; i < all_input_cores.size(); i++) {
        config.cores_with_non_meaningful_work.insert(all_input_cores[i]);
    }

    // If we have cores processing data from more than 1 batch, we need to trigger a check inside the kernel
    // to 'restart' the optimization and fill in the whole output image width
    // before the reuse for the new batch starts
    const uint32_t per_core_out_hw = single_core_height_ntiles * tt::constants::TILE_HEIGHT;
    const uint32_t total_batch_out_hw = output_image_height * output_image_width;
    config.single_core_processes_multiple_batches = (batch > 1) && (total_batch_out_hw % per_core_out_hw != 0);

    return config;
}

namespace {

// The intermediate conv_reader_indices tensor is allocated once by
// create_workload_descriptor and parked on the WorkloadDescriptor; we receive
// the raw Buffer* here and wire it into the READER_INDICES CB and reader CT args.
tt::tt_metal::ProgramDescriptor build_program_descriptor_sharded(
    const Conv2dParams& operation_attributes,
    const Conv2dInputs& tensor_args,
    Tensor& output_tensor,
    tt::tt_metal::Buffer* conv_reader_indices_buffer) {
    using tt::tt_metal::CBDescriptor;
    using tt::tt_metal::CBFormatDescriptor;
    using tt::tt_metal::ComputeConfigDescriptor;
    using tt::tt_metal::DataMovementConfigDescriptor;
    using tt::tt_metal::KernelDescriptor;
    using tt::tt_metal::ProgramDescriptor;
    using tt::tt_metal::SemaphoreDescriptor;

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

    std::vector<uint32_t> op_trace_metadata =
        ttnn::operations::sliding_window::generate_op_trace_metadata(sliding_window_config);
    std::vector<sliding_window::ShardBoundary> shard_boundaries =
        ttnn::operations::sliding_window::generate_shard_boundaries(sliding_window_config);

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
    // Non-const: an eligible HEIGHT_SHARDED no-bias conv may override these with a larger (relaxed)
    // subblock for the TileRowMajor (non-pin) path — see the CONV_TILE_PACK_ROW_MAJOR eligibility block
    // below. The override re-derives all subblock-dependent quantities.
    uint32_t out_subblock_h_ntiles = block_config.out_subblock_h_ntiles;
    uint32_t out_subblock_w_ntiles = block_config.out_subblock_w_ntiles;

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

    // Tensor b has weights and it should be tiled layout after converting conv weights into weight matrix
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
    const uint32_t filter_h = (uint32_t)sliding_window_config.window_hw.first;   // filter_h
    const uint32_t filter_w = (uint32_t)sliding_window_config.window_hw.second;  // filter_W
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
        is_1d_depthwise_conv(groups, ashape[3], output_channels, filter_h, ashape[1], has_bias);
    const uint32_t conv_act_c_read_bytes = conv_act_size_c * a.element_size() / conv_act_c_blocks;
    const bool coalesce_1d_depthwise_kw_reads = should_coalesce_1d_depthwise_conv_reads(
        is_conv_1d_depthwise_conv,
        a.memory_config().memory_layout(),
        input_channels_padded,
        filter_w,
        dilation_w,
        a.dtype());
    TT_FATAL(
        !coalesce_1d_depthwise_kw_reads || filter_h == 1,
        "Coalesced 1D depthwise reads require filter_h == 1, got {}",
        filter_h);

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
    if (is_conv_1d_depthwise_conv && height_sharded) {
        const uint32_t expected_act_block_w_ntiles =
            tt::round_up(
                input_channels_padded * (coalesce_1d_depthwise_kw_reads ? filter_w : 1), tt::constants::TILE_WIDTH) /
            tt::constants::TILE_WIDTH;
        TT_FATAL(
            act_block_w_ntiles == expected_act_block_w_ntiles,
            "1D depthwise activation block width mismatch. Got {} tiles, expected {} tiles",
            act_block_w_ntiles,
            expected_act_block_w_ntiles);
    }
    // check is for 16-byte alignment
    TT_FATAL(
        // Since fp16 is smalleset data format used for halo output, 8 input_channels is enough for 16 byte alignment
        input_channels_padded % 8 == 0,
        "Expected input channels to be padded for 16 byte alignment in L1 ({} % 16 != 0)",
        input_channels_padded);

    const uint32_t act_matrix_height_ntiles = out_block_h_ntiles * parallelization_config.num_cores_nhw;
    const uint32_t act_matrix_height = act_matrix_height_ntiles * tt::constants::TILE_HEIGHT;

    if (has_bias) {
        if (is_conv_1d_depthwise_conv) {
            TT_THROW("Bias is not supported for depthwise conv1d");
        }
        // Tensor bias is of shape {output_channels}
        TT_FATAL(bias.has_value(), "Bias tensor must be provided when has_bias is true");
        TT_FATAL(bias.value().buffer() != nullptr, "Bias tensor buffer must not be null");
        auto bias_shape_without_padding = bias.value().logical_shape();
        TT_FATAL(bias_shape_without_padding[0] == 1, "Bias should have batch == 1");
    }

    // Tile size divisibility checks
    TT_FATAL(
        weight_matrix_height % tt::constants::TILE_HEIGHT == 0, "Height of weight matrix needs to be divisible by 32");
    TT_FATAL(
        weight_matrix_width % tt::constants::TILE_WIDTH == 0, "Width of weight matrix needs to be divisible by 32");

    // Device compatibility checks
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
    const uint32_t num_blocks_act_w = is_conv_1d_depthwise_conv
                                          ? (coalesce_1d_depthwise_kw_reads ? 1 : filter_h * filter_w)
                                          : (slice_inner_dim ? filter_h : 1);
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
    uint32_t weight_block_h_ntiles = is_conv_1d_depthwise_conv
                                         ? act_block_h_ntiles * (coalesce_1d_depthwise_kw_reads ? filter_w : 1)
                                         : act_block_w_ntiles;
    uint32_t weight_block_num_tiles = weight_block_w_ntiles * weight_block_h_ntiles;

    // writer of conv op partially removes padding on the width
    // it removes the padding done for block width but it doesn't remove padding done for tiled width
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

    // Non-const: re-derived by the CONV_TILE_PACK_ROW_MAJOR override below if the conv is eligible.
    uint32_t act_num_subblocks = act_block_h_ntiles / out_subblock_h_ntiles;
    const uint32_t act_block_num_tiles = act_block_h_ntiles * act_block_w_ntiles;
    uint32_t act_subblock_h_ntiles = out_subblock_h_ntiles;
    uint32_t act_subblock_num_tiles = act_subblock_h_ntiles * act_block_w_ntiles;

    const uint32_t in0_num_blocks_w = conv_act_c_blocks * num_blocks_act_w;

    // weight
    tt::tt_metal::Buffer* weights_buffer = b.buffer();

    // bias
    tt::tt_metal::Buffer* bias_buffer = nullptr;
    uint32_t bias_ntiles = 0;
    if (has_bias) {
        bias_buffer = bias.value().buffer();
        bias_ntiles =
            bias.value().padded_shape()[3] / tt::constants::TILE_WIDTH;  // TODO: support non tile multiple sizes
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

    const uint32_t act_block_w_logical_scalars =
        is_conv_1d_depthwise_conv
            ? shard_shape[1] * (coalesce_1d_depthwise_kw_reads ? filter_w : 1)
            : (!slice_inner_dim ? shard_shape[1] * filter_h * filter_w : shard_shape[1] * filter_w);
    uint32_t act_block_w_extra_align_bytes =
        (tt::round_up(act_block_w_logical_scalars, tt::constants::TILE_WIDTH) - act_block_w_logical_scalars) *
        a.element_size();
    const uint32_t act_block_w_extra_align_scalars = act_block_w_extra_align_bytes / a.element_size();
    // When using block float format, we must handle cases where the data doesn't align to 16-scalar boundaries.
    // If act_block_w_extra_align_bytes contains a number of scalars that isn't a multiple of 16,
    // we need to zero out the temporary circular buffers used during the tiling process.
    // Failing to do this could allow residual junk data in L1 memory to corrupt valid input data.
    const bool needs_act_block_zero_out =
        act_block_w_extra_align_scalars % 16 != 0 && tt::tt_metal::is_block_float(output.dtype());

    const uint32_t tilized_act_tile_size = tt::tile_size(tilized_act_df);

    // Only enable packer l1 accumulation when there are in0_num_blocks_w > 2, otherwise
    // unnecessary overhead for reconfigs are added. Last iteration of l1 accumulation
    // does a spill and reload, so need more than 2 blocks to use l1 acc for packer
    // For bias, last iteration of l1 acc remains in intermediate buffer, does not spill and reload
    const bool packer_l1_acc_en = determine_packer_l1_acc(packer_l1_acc, has_bias, in0_num_blocks_w);

    // ── TileRowMajor eligibility (NON-pin dedicated-partials base — M2) ────────────────────────────────
    // The matmul helper can pack the interm/out row-major (TileRowMajor, pack_subblock_row_strided) on the
    // SAME non-pin dedicated-partials FIFO the SubblockMajor path uses (M1 dedicated matmul_partials_cb to
    // one output block; its FIFO wraps to base per output block). TileRowMajor lifts the SubblockMajor
    // constraint out_subblock_w == per_core_N, letting the tuner pick a LARGER (relaxed) subblock when
    // per_core_N exceeds DST capacity (so SubblockMajor was stranded at out_subblock_h == 1, e.g.
    // per_core_N 9/11/13). A larger subblock means fewer matmul_block_init / pack passes per output block.
    // Per output layout (NO pin, NO partials alias — partials is always dedicated since M1):
    //   • TILE (Out / OutWithRelu target): TileRowMajor packs straight into out_cb — row-major tile order
    //     IS the tiled shard layout, so no reblock, no untilize. Software spill/reload (l1_acc off) and
    //     per-address L1_ACC (l1_acc on) both work on the dedicated FIFO; the helper's strided reload
    //     (copy_subblock_row_strided) mirrors the strided spill for the l1_acc-on Out path. Supported for
    //     BOTH l1_acc on and off.
    //   • ROW_MAJOR (untilize_out, Interm target): TileRowMajor packs the interm row-major, then plain
    //     `untilize` reads it (no reblock gather). Supported for BOTH l1_acc on and off (M3). With l1_acc
    //     ON the helper accumulates per-address across K-blocks. With l1_acc OFF the software spill/reload
    //     drains the prior K-block's full output block AND reserves the next M-row-group in the SAME
    //     matmul_partials_cb; the last-block reserve_back lands one M-row-group ON TOP of a still-fronted
    //     full block of spills (the reload pop is sequenced after the reserve), so the dedicated partials
    //     CB is sized to 2*out_block_num_tiles for the l1_acc-OFF case (get_cb_info MATMUL_PARTIALS) — that
    //     headroom clears the otherwise-deadlocking reserve_back (was Watcher UPAW). No degrade to SBM.
    // The kernel keys all of this off the tile_pack_row_major compile arg (39) (see conv_bmm_tilize.cpp).
    // When ineligible we emit byte-identical SubblockMajor.
    //
    // Eligible iff ALL hold:
    //   • HEIGHT_SHARDED                 (width-sharded keeps SubblockMajor)
    //   • !has_bias                      (TileRowMajor + bias deadlocks the partials CB — see kernel note)
    //   • weights bf16 or fp32           (the lever is inert on bf8 — same packed-tile layout either way)
    //   • not 1d-depthwise               (0-page partials; the lever does not apply)
    //   • (no acc/layout gate — M3 sized the l1_acc-OFF partials CB for the ROW_MAJOR reload, so all four
    //     {l1_acc on,off}×{TILE,ROW_MAJOR} quadrants are supported)
    //   • the relaxed subblock differs from the SBM subblock AND is larger: relaxed.out_subblock_h > 1
    //     (captures "per_core_N stranded" — SBM fell back to h==1,w==per_core_N) AND act_block_h_ntiles
    //     (per_core_M) divisible by relaxed.out_subblock_h.
    bool tile_pack_row_major = false;
    // In-place-pack class (formerly "caller_owns"): the deep-K packer_l1_acc + bias convs that pin used
    // to capture (rn50 DS2/DS3/L3a/L3b/L4a/L4b) now route through the matmul helper's in-place pack path
    // (TileRowMajor + packer_l1_acc + Interm — the helper does its own single reserve/push; pin has been
    // deleted — GH#45995). These convs all satisfy out_subblock_w == per_core_N (in1_num_subblocks == 1),
    // so TileRowMajor is BYTE-IDENTICAL to SubblockMajor — we KEEP the SBM-derived subblock (do NOT
    // re-derive the relaxed/taller subblock) and only flip the layout flag (CONV_TILE_PACK_ROW_MAJOR). The
    // kernel derives the in-place-pack behavior itself from (TileRowMajor + packer_l1_acc + Interm); there
    // is no longer a separate host define. The decision is finalized after get_cb_info (where
    // partials_cb_uses_output is known); see the pin_class_caller_owns block following the CB-info call.
    {
        // Shared with get_cb_info() (via conv2d_utils) so the partials-CB alias decision is derived
        // identically on the allocation path and the L1-usage prediction path — see the helper's banner.
        const tt::DataFormat weights_df = tt::tt_metal::datatype_to_dataformat_converter(b.dtype());
        const auto trm = ttnn::operations::conv::auto_select_tile_pack_row_major(
            height_sharded,
            has_bias,
            weights_df,
            is_conv_1d_depthwise_conv,
            fp32_dest_acc_en,
            act_block_h_ntiles,
            weight_block_w_ntiles);
        if (trm.selected) {
            tile_pack_row_major = true;
            out_subblock_h_ntiles = trm.relaxed_out_subblock_h;
            out_subblock_w_ntiles = trm.relaxed_out_subblock_w;
            // Re-derive every subblock-dependent quantity from the relaxed shape. These were computed
            // from the SBM shape above; the relaxed shape satisfies the same TT_FATALs (vol <= DST cap
            // via the tuner, weight_block_w % w == 0 via the tuner's divisibility, act_block_h % h == 0
            // guaranteed by the helper's divisibility gate), so the SBM-shape validations remain correct.
            weight_num_subblocks = weight_block_w_ntiles / out_subblock_w_ntiles;
            out_subblock_num_tiles = out_subblock_h_ntiles * out_subblock_w_ntiles;
            act_num_subblocks = act_block_h_ntiles / out_subblock_h_ntiles;
            act_subblock_h_ntiles = out_subblock_h_ntiles;
            act_subblock_num_tiles = act_subblock_h_ntiles * act_block_w_ntiles;
            log_info(
                tt::LogOp,
                "conv2d TileRowMajor (non-pin) ELIGIBLE: per_core_M={} per_core_N={} fp32_accum={} "
                "packer_l1_acc={} out_layout={} | SBM subblock={}x{} relaxed subblock={}x{} "
                "(auto-selected TileRowMajor; no bias, weights bf16/fp32)",
                act_block_h_ntiles,
                weight_block_w_ntiles,
                fp32_dest_acc_en,
                packer_l1_acc_en,
                untilize_out ? "ROW_MAJOR" : "TILE",
                trm.sbm_out_subblock_h,
                trm.sbm_out_subblock_w,
                trm.relaxed_out_subblock_h,
                trm.relaxed_out_subblock_w);
        }
    }

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
    // The conv_reader_indices tensor itself is allocated once in
    // create_workload_descriptor and parked on workload_descriptor.buffers; we
    // receive the raw Buffer* and read its page size for the CB sizing below.
    // Pass the actual DRAM/L1-small config buffer page size into get_cb_info so the predicted
    // READER_INDICES CB footprint matches the CB this factory creates. Without this, the in-DRAM
    // path was sized to the worst case (1 uint16 per output row), holding spare L1 that is never
    // populated by the reader kernel.
    const uint32_t reader_indices_actual_page_size = conv_reader_indices_buffer->page_size();
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
        input_channels_padded,
        reader_indices_actual_page_size);

    // Emit CBDescriptors directly onto the ProgramDescriptor.  The framework
    // tracks the globally-allocated CBs (ACT_SHARDED on input, OUT/PARTIALS on
    // output, READER_INDICES on the workload-scoped indices buffer) and patches
    // their addresses on cache hits — no per-op UpdateDynamicCircularBufferAddress
    // override needed.
    emit_cb_descriptors(cb_info, desc, all_cores, a.buffer(), output.buffer(), conv_reader_indices_buffer);

    const uint32_t in_num_cores_x = input_cores.bounding_box().end_coord.x + 1;
    const uint32_t in_num_cores_y = input_cores.bounding_box().end_coord.y + 1;

    const CoreCoord top_left_core = {(std::size_t)0, (std::size_t)0};
    const CoreCoord top_left_core_plus_one = {(std::size_t)1, (std::size_t)1};
    const CoreCoord bottom_right_core = {(std::size_t)in_num_cores_x - 1, (std::size_t)in_num_cores_y - 1};
    const CoreCoord top_left_core_physical = device->worker_core_from_logical_core(top_left_core);
    const CoreCoord top_left_core_plus_one_physical = device->worker_core_from_logical_core(top_left_core_plus_one);
    const CoreCoord bottom_right_core_physical = device->worker_core_from_logical_core(bottom_right_core);

    CoreRangeSet mcast_sender_cores =
        CoreRangeSet(CoreRange(top_left_core, top_left_core));  // If single core, this kernel doesn't do mcasting
    CoreRangeSet mcast_receiver_cores;
    uint32_t weights_mcast_sender_semaphore_id = 0;
    uint32_t weights_mcast_receiver_semaphore_id = 0;
    uint32_t act_mcast_sender_semaphore_id = 0;
    uint32_t act_mcast_receiver_semaphore_id = 0;
    uint32_t act_split_reader_reserve_done_semaphore_id = 0;
    uint32_t act_split_reader_write_done_semaphore_id = 0;

    // Check if we should run BRISC kernels on cores that are not in the output grid ( when split reader is enabled and
    // the output grid is smaller than the input grid)
    const bool populate_skipped_work_cores =
        enable_split_reader && block_sharded && input_cores.num_cores() > output_cores.num_cores();

    const bool overlap_act_cb =
        get_cb_info_by_name(cb_info, Conv2dCb::ACT_ROW_MAJOR_BFLOAT16).overlapped_by_cb.has_value();
    // When split reader is enabled with overlapped CBs, both readers write to the same circular buffer.
    // This requires synchronization between the main reader and the second reader to prevent race conditions.
    const bool split_reader_cb_shared = enable_split_reader && overlap_act_cb && block_sharded;

    // Sequential semaphore IDs match the order in which CreateSemaphore would have
    // allocated them; the framework realises desc.semaphores in this order on cache miss.
    auto push_semaphore = [&desc](const CoreRangeSet& cores) -> uint32_t {
        uint32_t id = static_cast<uint32_t>(desc.semaphores.size());
        desc.semaphores.push_back(SemaphoreDescriptor{
            .id = id,
            .core_ranges = cores,
            .initial_value = 0,  // 0 == INVALID
        });
        return id;
    };

    if (block_sharded) {
        const CoreCoord out_bottom_right_core = {(std::size_t)num_cores_x - 1, (std::size_t)num_cores_y - 1};
        // 2D mcast
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
        act_mcast_sender_semaphore_id = push_semaphore(all_cores);
        act_mcast_receiver_semaphore_id = push_semaphore(all_cores);

        if (split_reader_cb_shared) {
            weights_mcast_sender_semaphore_id = push_semaphore(all_cores);
            weights_mcast_receiver_semaphore_id = push_semaphore(all_cores);
            act_split_reader_reserve_done_semaphore_id = push_semaphore(all_cores);
            act_split_reader_write_done_semaphore_id = push_semaphore(all_cores);
        } else {
            weights_mcast_sender_semaphore_id = push_semaphore(output_cores);
            weights_mcast_receiver_semaphore_id = push_semaphore(output_cores);
        }
    } else {
        // 1D mcast
        if (!skip_weights_mcast) {
            mcast_receiver_cores = all_cores.subtract(mcast_sender_cores);
            weights_mcast_sender_semaphore_id = push_semaphore(output_cores);
            weights_mcast_receiver_semaphore_id = push_semaphore(output_cores);
        }
    }

    // 1D depthwise compute uses dest-reuse for accumulation — no MATMUL_PARTIALS CB is allocated.
    const bool partials_cb_uses_output =
        !is_conv_1d_depthwise_conv && get_cb_info_by_name(cb_info, Conv2dCb::MATMUL_PARTIALS).is_globally_allocated;
    log_debug(tt::LogOp, "partials_cb_uses_output: {}", partials_cb_uses_output);

    // ── in-place-pack class (DEFAULT — replaces the deleted pin for the deep-K INTERM-target class) ────
    // The compute kernel used to pin (the now-deleted pin_interm_to_captured_base) exactly when:
    //     packer_l1_acc && SubblockMajor && (fuse_bias || untilize_out || !partials_cb_uses_output)
    // Pin has been removed (GH#45995); that class now routes through the matmul helper's in-place pack
    // path (TileRowMajor + packer_l1_acc + Interm — the helper does its own single reserve/push and skips
    // the per-block reserve/push/drain). It requires TileRowMajor layout + DEDICATED (non-aliased)
    // partials. For the engaged convs out_subblock_w == per_core_N (in1_num_subblocks == 1), so
    // TileRowMajor is BYTE-IDENTICAL to SubblockMajor — we KEEP the SBM-derived subblock (no relaxed
    // re-derive) and only flip the layout flag (CONV_TILE_PACK_ROW_MAJOR); the kernel derives the
    // in-place-pack behavior from the template args, so there is no longer a separate host define.
    //
    // Engage only when the production no-bias TRM-relaxed path above did NOT already claim this conv
    // (tile_pack_row_major still false): that path re-derives a taller subblock, which is incompatible
    // with the byte-identical SBM-subblock requirement here.
    //
    // Scope: the INTERM-target members of the former pin class — the matmul packs its last K-block to the
    // DEDICATED matmul_partials_cb (LastBlockTarget::Interm), and the kernel's caller_owns reserve/push
    // is on matmul_partials_cb. The downstream consumer reads partials and writes a DISTINCT out_cb:
    //   • fuse_bias  (step a): bias-add reads partials → out_cb. Validated on rn50 DS2/DS3/L3a/L3b/L4a/L4b
    //     (bit-identical to the old SBM+pin).
    //   • untilize_out, no bias (step a′): the untilize phase reads the row-major partials via plain
    //     `untilize` (WaitMode::WaitBlock) → out_cb — the SAME helper instantiation CCL all_gather and the
    //     bias path use (LastBlockTarget::Interm + TileRowMajor + caller_owns_pack_target). The +12.7%
    //     win on these (vu_512_30x40_BS ROW_MAJOR) is preserved without pin.
    // The OUT target (no bias, no untilize, dedicated multi-output-block partials) is NOT migrated to
    // caller_owns: it runs the helper's NON-pin FIFO. Rationale: it has no production-trace vehicle (every
    // real no-bias TILE conv is single-output-block → partials aliased onto OUT → already non-pin;
    // multi-output-block OUT was only reachable via a synthetic act_block_h override), and caller_owns +
    // the Out-target software-reload would need new shared-helper logic untested by any shipped caller
    // (CCL/bias/untilize all use the Interm target). The synthetic-only non-pin perf delta is accepted.
    //
    // caller_owns keeps the SBM-derived subblock (no relaxed re-derive). For in1_num_subblocks == 1
    // (out_subblock_w == per_core_N) TileRowMajor is BYTE-IDENTICAL to SBM; for in1_num_subblocks > 1
    // TileRowMajor is a different-but-valid layout (the helper's row-strided pack + the consumer's
    // row-major read agree), proven correct by the step-(a) bias convs (SD per_core_N=10, in1=2) and the
    // d7b9fb1ad1c absolute-offset fix for in0_num_subblocks > 1. So NO in1-subblock guard.
    //
    // fp32+bias+untilize is NOT excluded. The former exclusion existed because the kernel aliased the
    // bias-add output onto matmul_partials_cb when untilize_out (untilize_mode_out_cb_id ==
    // matmul_partials_cb). The TileRowMajor bias-add issues reserve_back BEFORE pop_front, so with
    // fp32_dest_acc_en the fp32-sized partials CB (4B/elem) had no slack — the full out_block filled it and
    // reserve_back deadlocked. The kernel now does the bias-add IN PLACE for this class (bias_in_place in
    // conv_bmm_tilize.cpp): the matmul packed the whole block into matmul_partials_cb with a single
    // reserve/push, and the bias-add read-modify-writes that fronted block at its original positions with
    // NO reserve_back / push_back / pop_front. With no reserve at all the reserve-before-pop circular wait
    // cannot occur for ANY dtype, so fp32+bias+untilize takes caller_owns + TileRowMajor like every other
    // bias+untilize conv (bf16 included — test_sd_conv_wh), and NO staging CB is needed — the untilize phase
    // reads the biased block straight out of matmul_partials_cb and pops it. The dedicated UNTILIZE_STAGING
    // CB (see conv2d_op_program_factory_common.cpp) is now 0 pages for this class (its full output-block of
    // L1 is recovered); it is retained only for the SubblockMajor (!packer_l1_acc) bias+untilize path, whose
    // per-subblock FIFO does not fit the in-place read-modify-write model.
    const bool pin_class_caller_owns = packer_l1_acc_en && (has_bias || untilize_out);
    if (!tile_pack_row_major && pin_class_caller_owns) {
        tile_pack_row_major = true;
        log_debug(
            tt::LogOp,
            "conv2d CALLER_OWNS (default, was pin): per_core_M={} per_core_N={} out_subblock={}x{} "
            "in0_num_subblocks={} in1_num_subblocks={} packer_l1_acc={} has_bias={} untilize_out={} "
            "partials_dedicated={}",
            act_block_h_ntiles,
            weight_block_w_ntiles,
            out_subblock_h_ntiles,
            out_subblock_w_ntiles,
            act_block_h_ntiles / out_subblock_h_ntiles,
            weight_block_w_ntiles / out_subblock_w_ntiles,
            packer_l1_acc_en,
            has_bias,
            untilize_out,
            !partials_cb_uses_output);
    }

    std::string reader_kernel;
    std::string compute_kernel = "ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/conv_bmm_tilize.cpp";
    std::string writer_mcast_sender_kernel =
        "ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/"
        "reader_writer_tiled_out_1d_mcast_sender_conv_weights_tiled_col_to_rm_blocks.cpp";
    std::string writer_mcast_receiver_kernel =
        "ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/"
        "reader_writer_tiled_out_1d_mcast_receiver_conv_weights_tiled_col_to_rm_blocks.cpp";

    if (!is_conv_1d_depthwise_conv && block_sharded) {
        // Block sharded conv
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
        // 1D Depthwise Conv (height sharded)
        compute_kernel = "ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/compute_depthwise_conv1d.cpp";
        reader_kernel = "ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/reader_depthwise_conv1d.cpp";
    } else {
        // Height sharded conv
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
        (uint32_t)act_block_w_extra_align_bytes,                          // only used for 1d systolic variant
        (uint32_t)num_blocks_act_h_per_core,                              // act_num_blocks_h
        (uint32_t)act_block_num_tiles,                                    // act_block_num_tiles
        (uint32_t)conv_act_c_blocks,                                      // act_w_num_outer
        (uint32_t)(transpose_mcast ? num_cores_y - 1 : num_cores_x - 1),  // act_mcast_num_dests
        (uint32_t)(transpose_mcast ? num_cores_y - 1 : num_cores_x - 1),  // act_mcast_num_cores
        (uint32_t)act_mcast_sender_semaphore_id,
        (uint32_t)act_mcast_receiver_semaphore_id,
        (uint32_t)tilized_act_tile_size,  // act_mcast_tile_size_bytes
        (uint32_t)(transpose_mcast ? 1 : 0),
        (uint32_t)needs_act_block_zero_out,  // zero_out_act_cb
        get_cb_info_by_name(cb_info, Conv2dCb::ACT).index,
        get_cb_info_by_name(cb_info, Conv2dCb::ACT_SHARDED).index,
        get_cb_info_by_name(cb_info, Conv2dCb::READER_INDICES).index,
        get_cb_info_by_name(cb_info, Conv2dCb::ACT_TILIZED).index,
        get_cb_info_by_name(cb_info, Conv2dCb::ACT_ROW_MAJOR_BFLOAT16).index,
        get_cb_info_by_name(cb_info, Conv2dCb::L1_ARRAY).index,
        (uint32_t)enable_split_reader,
        (uint32_t)(is_conv_1d_depthwise_conv ? coalesce_1d_depthwise_kw_reads : enable_activation_reuse)};

    std::map<std::string, std::string> reader_defines;
    std::map<std::string, std::string> writer_defines;
    std::map<std::string, std::string> writer_mcast_sender_defines;
    std::map<std::string, std::string> compute_defines;

    if (config_tensors_in_dram) {
        reader_defines["CONFIG_TENSOR_IN_DRAM"] = "1";
        writer_defines["CONFIG_TENSOR_IN_DRAM"] = "1";               // Needed for split reader
        writer_mcast_sender_defines["CONFIG_TENSOR_IN_DRAM"] = "1";  // Needed for split reader
        reader_compile_time_args.push_back(conv_reader_indices_buffer->address());
        reader_compile_time_args.push_back(conv_reader_indices_buffer->page_size());
        tt::tt_metal::TensorAccessorArgs(conv_reader_indices_buffer).append_to(reader_compile_time_args);
    } else {
        // Put enough 0s so that the offsets of activation reuse args are the same.
        // TensorAccessorArgs appends 2 CT args (is_dram + aligned_page_size), so we need 4 zeros total
        // (address + page_size + 2 TensorAccessorArgs) to match the DRAM path.
        reader_compile_time_args.push_back(0);
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
        // Add dummy activation reuse arguments when not enabled
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

    // TileRowMajor (non-pin) path: the compute kernel packs the interm/out row-major
    // (pack_subblock_row_strided) on the dedicated-partials FIFO, and (ROW_MAJOR output) untilizes it with
    // plain `untilize` instead of reblock_and_untilize. Eligibility decided above; the relaxed
    // out_subblock_{h,w} were already folded into the compute compile args.
    // TileRowMajor is passed to the compute kernel as compile arg 39 (see compute_kernel_args), not a
    // define. The deep-K fuse_bias/untilize + packer_l1_acc INTERM-target convs (formerly the
    // caller_owns / pin class) derive their in-place-pack behavior from the kernel's own template args
    // (TileRowMajor + packer_l1_acc + Interm), so no separate signal is needed.

    ttnn::operations::compute_throttle_utils::throttle_mm_perf(
        device->arch(), output_cores.num_cores(), compute_defines, ttnn::get_throttle_level(compute_kernel_config));

    for (auto elem : compute_defines) {
        log_trace(tt::LogOp, "compute_defines: {} = {}", elem.first, elem.second);
    }

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
        (uint32_t)filter_w,                       // weight_size_w
        (uint32_t)(conv_act_size_w + pad_w),      // conv_act_size_w_padded
        (uint32_t)act_block_w_extra_align_bytes,  // only used for 1d systolic variant
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
        // get total number of blocks in the ACT CB so that we can compute the loop around when moving write
        // pointer in second reader
        const uint32_t act_cb_block_cnt = get_cb_info_by_name(cb_info, Conv2dCb::ACT).num_pages /
                                          (act_block_num_tiles_split + act_block_num_tiles_split_last);

        // In cases where the overlapped buffer is double buffered and number of push_backs done on NCRISC is odd,
        // there are two addresses that need to be written to on the BRISC side for the second reader.
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
        // Add dummy activation reuse arguments when not enabled
        std::vector<uint32_t> activation_reuse_dummy_args(8, 0);
        split_reader_args.insert(
            split_reader_args.end(), activation_reuse_dummy_args.begin(), activation_reuse_dummy_args.end());
    }
    writer_compile_time_args.insert(writer_compile_time_args.end(), split_reader_args.begin(), split_reader_args.end());
    tt::tt_metal::TensorAccessorArgs(b.buffer()).append_to(writer_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(bias ? bias->buffer() : nullptr).append_to(writer_compile_time_args);

    const bool check_skip_compute = input_cores != output_cores;

    std::vector<uint32_t> compute_kernel_args;
    if (is_conv_1d_depthwise_conv) {
        // compute_depthwise_conv1d.cpp uses a specialized dest-reuse accumulation path. The last
        // two args select the coalesced kernel-width activation layout when the reader can fetch
        // all kernel-width sticks as one NoC packet.
        compute_kernel_args = {
            act_block_w_ntiles,                                         // 0: in0_block_w
            act_num_subblocks,                                          // 1: in0_num_subblocks
            act_block_num_tiles,                                        // 2: in0_block_num_tiles
            num_blocks_act_h_per_core,                                  // 3: in0_num_blocks_h
            in0_num_blocks_w,                                           // 4: in0_num_blocks_w
            get_cb_info_by_name(cb_info, Conv2dCb::ACT).index,          // 5: in0_cb_id (raw RM)
            get_cb_info_by_name(cb_info, Conv2dCb::WEIGHTS).index,      // 6: in1_cb_id
            get_cb_info_by_name(cb_info, Conv2dCb::ACT_TILIZED).index,  // 7: tilized_in0_cb_id
            get_cb_info_by_name(cb_info, Conv2dCb::OUT).index,          // 8: out_cb_id
            filter_w,                                                   // 9: kernel_width
            coalesce_1d_depthwise_kw_reads,                             // 10: coalesced activation block
        };
    } else {
        compute_kernel_args = {
            act_block_w_ntiles,
            act_num_subblocks,
            act_block_num_tiles,
            act_subblock_num_tiles,
            enable_split_reader ? act_block_h_ntiles
                                : act_subblock_h_ntiles * act_num_subblocks,  // reader_num_h_subblocks
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
            // Slot 25 (was the now-dead partials_cb_uses_output) now carries the UNTILIZE_STAGING CB
            // index: the distinct one-block buffer the fuse_bias + untilize_out bias-add writes to (and
            // the untilize phase reads from) instead of aliasing onto matmul_partials_cb. Only allocated
            // (valid index) when untilize_out && has_bias; otherwise kInvalidCBIndex (kernel never uses it).
            get_cb_info_by_name(cb_info, Conv2dCb::UNTILIZE_STAGING).index,
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
            compute_kernel_args.push_back(
                activation_reuse_config.tilized_cb_row_offset / COMPUTE_KERNEL_ADDRESS_DIVISOR);
            compute_kernel_args.push_back(
                activation_reuse_config.tilized_cb_second_reader_offset / COMPUTE_KERNEL_ADDRESS_DIVISOR);
        } else {
            std::vector<uint32_t> activation_reuse_dummy_args = {0, 0, 0, 0};
            compute_kernel_args.insert(
                compute_kernel_args.end(), activation_reuse_dummy_args.begin(), activation_reuse_dummy_args.end());
        }
        compute_kernel_args.push_back(static_cast<uint32_t>(split_reader_cb_shared));
        compute_kernel_args.push_back(static_cast<uint32_t>(tile_pack_row_major));  // 39: TileRowMajor layout
    }

    // "Dedicate partials, match matmul" (GH#45995): every conv routes through the matmul-helper kernel
    // (conv_bmm_tilize.cpp). It uses a DEDICATED one-block matmul_partials_cb (is_globally_allocated=false)
    // instead of a bespoke pin scheme, so its non-pin FIFO wraps to a fixed base each K-block and
    // packer_l1_acc lands at a fixed address for multi-output-block convs too. The non-l1_acc path likewise
    // uses the helper's plain FIFO reserve/push/pop (no L1_ACC, so no per-address accumulation requirement).

    const tt::tt_metal::NOC writer_mcast_noc = tt::tt_metal::detail::preferred_noc_for_dram_read(device->arch());
    const tt::tt_metal::NOC reader_noc =
        writer_mcast_noc == tt::tt_metal::NOC::NOC_0 ? tt::tt_metal::NOC::NOC_1 : tt::tt_metal::NOC::NOC_0;

    // Build the writer_mcast_sender kernel descriptor (placed on mcast_sender_cores).
    // We build runtime_args below after kernel placement is fixed.
    KernelDescriptor writer_mcast_sender_desc;
    writer_mcast_sender_desc.kernel_source = writer_mcast_sender_kernel;
    writer_mcast_sender_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_mcast_sender_desc.core_ranges = mcast_sender_cores;
    writer_mcast_sender_desc.compile_time_args = writer_compile_time_args;
    for (const auto& [k, v] : writer_mcast_sender_defines) {
        writer_mcast_sender_desc.defines.emplace_back(k, v);
    }
    writer_mcast_sender_desc.config = DataMovementConfigDescriptor{
        .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
        .noc = writer_mcast_noc,
    };

    // Optional writer_mcast_receiver kernel (created only when weights mcast is not skipped).
    KernelDescriptor writer_mcast_receiver_desc;
    const bool create_writer_mcast_receiver = !skip_weights_mcast;
    if (create_writer_mcast_receiver) {
        writer_mcast_receiver_desc.kernel_source = writer_mcast_receiver_kernel;
        writer_mcast_receiver_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
        writer_mcast_receiver_desc.core_ranges = mcast_receiver_cores;
        writer_mcast_receiver_desc.compile_time_args = writer_compile_time_args;
        for (const auto& [k, v] : writer_defines) {
            writer_mcast_receiver_desc.defines.emplace_back(k, v);
        }
        writer_mcast_receiver_desc.config = DataMovementConfigDescriptor{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = writer_mcast_noc,
        };
    }

    KernelDescriptor reader_desc;
    reader_desc.kernel_source = reader_kernel;
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = height_sharded ? input_cores : all_cores;
    reader_desc.compile_time_args = std::move(reader_compile_time_args);
    for (const auto& [k, v] : reader_defines) {
        reader_desc.defines.emplace_back(k, v);
    }
    reader_desc.config = DataMovementConfigDescriptor{
        .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
        .noc = reader_noc,
    };

    KernelDescriptor compute_kernel_desc;
    compute_kernel_desc.kernel_source = compute_kernel;
    compute_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_kernel_desc.core_ranges = height_sharded ? input_cores : all_cores;
    compute_kernel_desc.compile_time_args = std::move(compute_kernel_args);
    for (const auto& [k, v] : compute_defines) {
        compute_kernel_desc.defines.emplace_back(k, v);
    }
    compute_kernel_desc.config = ComputeConfigDescriptor{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
    };

    // Helper lambda to setup mcast arguments
    auto setup_mcast_args = [&](bool is_noc_0, uint32_t start_x, uint32_t start_y, uint32_t end_x, uint32_t end_y) {
        return is_noc_0 ? std::vector<uint32_t>{start_x, start_y, end_x, end_y}
                        : std::vector<uint32_t>{end_x, end_y, start_x, start_y};
    };

    // Setup reader runtime arguments
    if (block_sharded) {
        const uint32_t in_num_cores_x = input_cores.bounding_box().end_coord.x + 1;
        const uint32_t in_num_cores_y = input_cores.bounding_box().end_coord.y + 1;
        std::vector<uint32_t> act_mcast_noc_y;
        if (transpose_mcast) {
            act_mcast_noc_y.reserve(in_num_cores_y);
            for (uint32_t core_idx_y = 0; core_idx_y < in_num_cores_y; ++core_idx_y) {
                act_mcast_noc_y.push_back(device->worker_core_from_logical_core({0, core_idx_y}).y);
            }
        } else {
            // NOTE: using same var for x as well, this is intentional
            act_mcast_noc_y.reserve(in_num_cores_x);
            for (int32_t core_idx_x = 0; core_idx_x < in_num_cores_x; ++core_idx_x) {
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

                    reader_rt_args.push_back(core.y);                  // act_mcast_sender_id
                    reader_rt_args.push_back(bottom_core_physical.x);  // act_mcast_sender_noc_x
                } else {
                    CoreCoord core_physical = device->worker_core_from_logical_core(core);

                    reader_rt_args = setup_mcast_args(
                        reader_is_noc_0,
                        top_left_core_physical.x,
                        core_physical.y,
                        out_bottom_right_core_physical.x,
                        core_physical.y);
                    reader_rt_args.push_back(core.x);           // act_mcast_sender_id
                    reader_rt_args.push_back(core_physical.y);  // act_mcast_sender_noc_x
                }
                reader_rt_args.push_back(static_cast<uint32_t>(is_receiver_core));  // is_receiver_core
                reader_rt_args.push_back(static_cast<uint32_t>(is_sender_core));    // is_receiver_core
                reader_rt_args.push_back(transpose_mcast ? core.x : core.y);        // dram config reader index
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

    // Setup writer mcast arguments
    // Setup sender args first.  runtime_args[0] is the weight buffer address and
    // runtime_args[1] is the (optional) bias buffer address; both are registered
    // as Buffer* bindings via emplace_runtime_args so the framework's fast
    // cache-hit path patches addresses without GetRuntimeArgs.  When bias is
    // absent we embed a literal 0 — has_bias gates the bias read on the kernel
    // side.
    for (const CoreRange& core_range : mcast_sender_cores.ranges()) {
        for (const CoreCoord& core : core_range) {
            if (populate_skipped_work_cores && !output_cores.contains(core)) {
                // Pad-out path: 14 zeros with only the semaphore/bias-flag slots
                // populated.  weight/bias addresses are unused for skipped work
                // cores so we keep them as literal zeros (no buffer binding).
                KernelDescriptor::RTArgList args;
                args.reserve(14);
                args.push_back(uint32_t{0});  // 0: weight addr (unused for skipped cores)
                args.push_back(uint32_t{0});  // 1: bias addr (unused)
                for (int i = 2; i < 10; ++i) {
                    args.push_back(uint32_t{0});
                }
                args.push_back(weights_mcast_sender_semaphore_id);
                args.push_back(weights_mcast_receiver_semaphore_id);
                args.push_back(uint32_t{1});  // is_sender_core, always true for input_cores
                args.push_back(uint32_t{1});  // skip_work
                writer_mcast_sender_desc.emplace_runtime_args(core, args);
                continue;
            }
            // Calculate weight slice indices
            uint32_t weight_slice_i = ((block_sharded && transpose_mcast) || !block_sharded) ? core.y : core.x;

            uint32_t out_start_tile_id_w = weight_slice_i * per_core_out_matrix_width_ntiles;
            uint32_t bias_tile_offset = out_start_tile_id_w;

            TT_FATAL(
                bias_tile_offset < bias_ntiles || !has_bias,
                "bias_tile_offset {} should be less than bias_ntiles {}",
                bias_tile_offset,
                bias_ntiles);

            KernelDescriptor::RTArgList sender_rt_args;
            sender_rt_args.push_back(weights_buffer);  // 0: weight buffer address (Buffer* binding)
            if (bias_buffer != nullptr) {
                sender_rt_args.push_back(bias_buffer);  // 1: bias buffer address (Buffer* binding)
            } else {
                sender_rt_args.push_back(uint32_t{0});  // 1: bias placeholder (has_bias gates the read)
            }
            sender_rt_args.push_back(out_start_tile_id_w);  // 2
            sender_rt_args.push_back(bias_tile_offset);     // 3

            if (block_sharded) {
                const bool is_sender_core = input_cores.contains(core);
                // 2D multicast setup
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

                    sender_rt_args.append(mcast_coords);
                    sender_rt_args.push_back(num_cores_x - 1);  // mcast_num_dests
                    sender_rt_args.push_back(num_cores_x - 1);  // mcast_num_cores
                    sender_rt_args.push_back(weights_mcast_sender_semaphore_id);
                    sender_rt_args.push_back(weights_mcast_receiver_semaphore_id);
                    sender_rt_args.push_back(static_cast<uint32_t>(is_sender_core));
                    sender_rt_args.push_back(uint32_t{0});  // skip_work
                    writer_mcast_sender_desc.emplace_runtime_args(core, sender_rt_args);
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

                    sender_rt_args.append(mcast_coords);
                    sender_rt_args.push_back(num_cores_y - 1);  // mcast_num_dests
                    sender_rt_args.push_back(num_cores_y - 1);  // mcast_num_cores
                    sender_rt_args.push_back(weights_mcast_sender_semaphore_id);
                    sender_rt_args.push_back(weights_mcast_receiver_semaphore_id);
                    sender_rt_args.push_back(static_cast<uint32_t>(is_sender_core));
                    sender_rt_args.push_back(uint32_t{0});  // skip_work
                    writer_mcast_sender_desc.emplace_runtime_args(core, sender_rt_args);
                }
            } else {
                // 1D multicast setup
                std::vector<uint32_t> mcast_coords = setup_mcast_args(
                    writer_mcast_noc == tt::tt_metal::NOC::NOC_0,
                    top_left_core_physical.x,
                    top_left_core_physical.y,
                    bottom_right_core_physical.x,
                    bottom_right_core_physical.y);

                sender_rt_args.append(mcast_coords);
                sender_rt_args.push_back(total_active_num_cores - 1);  // mcast_num_dests
                sender_rt_args.push_back(total_num_cores - 1);         // mcast_num_cores
                sender_rt_args.push_back(weights_mcast_sender_semaphore_id);
                sender_rt_args.push_back(weights_mcast_receiver_semaphore_id);
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
                writer_mcast_sender_desc.emplace_runtime_args(core, sender_rt_args);
            }
        }
    }

    // Setup receiver args second
    if (create_writer_mcast_receiver) {
        for (const CoreRange& core_range : mcast_receiver_cores.ranges()) {
            // Helper lambda to create receiver runtime args
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

    if (input_cores != output_cores) {
        CoreCoord bottom_right_core_out = output_cores.bounding_box().end_coord;
        uint32_t end_coord_x = bottom_right_core_out.x;
        uint32_t end_coord_y = bottom_right_core_out.y;
        for (const CoreRange range : all_cores.ranges()) {
            for (const CoreCoord core : range) {
                bool skip_compute = transpose_mcast ? core.y > end_coord_y : core.x > end_coord_x;
                compute_kernel_desc.runtime_args.emplace_back(
                    core, std::vector<uint32_t>{static_cast<uint32_t>(skip_compute)});
            }
        }
    }

    // Descriptor-path CB-size check (mirrors the legacy post_conv2d_op_memory_checks
    // CB-sizing equality).  The L1-allocator delta half of the legacy check requires a
    // realised Program and so isn't reachable here: the framework realises the Program
    // after this function returns.
    post_conv2d_op_memory_checks_descriptor(desc, operation_attributes, tensor_args, reader_indices_actual_page_size);

    desc.kernels.push_back(std::move(writer_mcast_sender_desc));
    if (create_writer_mcast_receiver) {
        desc.kernels.push_back(std::move(writer_mcast_receiver_desc));
    }
    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(compute_kernel_desc));
    return desc;
}

}  // namespace

tt::tt_metal::WorkloadDescriptor Conv2dShardedProgramFactory::create_workload_descriptor(
    const Conv2dParams& operation_attributes,
    const Conv2dInputs& tensor_args,
    Tensor& output_tensor,
    const ttnn::MeshCoordinateRangeSet& tensor_coords) {
    tt::tt_metal::WorkloadDescriptor workload_descriptor;

    // Allocate the conv_reader_indices tensor on device once for the whole
    // workload.  Wrap it in a shared_ptr<Tensor> so ~Tensor doesn't fire when
    // the local goes out of scope: ~Tensor calls DeviceStorage::deallocate
    // which force-frees the device buffer regardless of any shared_ptr
    // ownership of the MeshBuffer (see WorkloadBuffer rationale in
    // workload_descriptor.hpp).  Holding the Tensor in the WorkloadDescriptor
    // defers destruction until the cached workload is evicted.
    const auto& a = tensor_args.a;
    const auto& sliding_window_config = operation_attributes.sliding_window_config;
    const auto& block_config = operation_attributes.block_config;
    const auto& parallelization_config = operation_attributes.parallelization_config;
    const auto enable_activation_reuse = operation_attributes.enable_activation_reuse;
    const auto force_split_reader = operation_attributes.force_split_reader;
    const bool config_tensors_in_dram = operation_attributes.config_tensors_in_dram;
    const bool block_sharded = a.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED;

    sliding_window::ParallelConfig input_parallel_config = {
        .grid = a.memory_config().shard_spec().value().grid,
        .shard_scheme = a.memory_config().memory_layout(),
        .shard_orientation = a.memory_config().shard_spec().value().orientation,
    };

    std::vector<uint32_t> op_trace_metadata =
        ttnn::operations::sliding_window::generate_op_trace_metadata(sliding_window_config);
    std::vector<sliding_window::ShardBoundary> shard_boundaries =
        ttnn::operations::sliding_window::generate_shard_boundaries(sliding_window_config);

    const uint32_t stride_w = sliding_window_config.is_transpose ? 1 : (uint32_t)sliding_window_config.stride_hw.second;
    const uint32_t act_block_h_ntiles = block_config.act_block_h_ntiles;

    // Determine the (act_block_h_datums, last) pair for the host-side index
    // generator.  This must agree exactly with the enable_split_reader path
    // computed by build_program_descriptor_sharded(), since the generated index tensor
    // is consumed by the reader kernel.
    const auto& b = tensor_args.b;
    const auto output_channels = operation_attributes.output_channels;
    const auto groups = operation_attributes.groups;
    const auto has_bias = operation_attributes.has_bias;
    const uint32_t filter_h = (uint32_t)sliding_window_config.window_hw.first;
    const uint32_t filter_w = (uint32_t)sliding_window_config.window_hw.second;
    const uint32_t dilation_w = (uint32_t)sliding_window_config.dilation_hw.second;
    const ttnn::Shape ashape(operation_attributes.input_tensor_shape);
    const std::array<uint32_t, 2> shard_shape = a.shard_spec().value().shape;

    uint32_t input_channels_padded = shard_shape[1];
    if (block_sharded) {
        const uint32_t in_num_cores_x = input_parallel_config.grid.bounding_box().end_coord.x + 1;
        const uint32_t in_num_cores_y = input_parallel_config.grid.bounding_box().end_coord.y + 1;
        const bool transpose_mcast = a.shard_spec().value().orientation == ShardOrientation::COL_MAJOR;
        input_channels_padded = shard_shape[1] * (transpose_mcast ? in_num_cores_y : in_num_cores_x);
    }
    const bool is_conv_1d_depthwise_conv =
        is_1d_depthwise_conv(groups, ashape[3], output_channels, filter_h, ashape[1], has_bias);

    auto compute_kernel_config_args =
        get_compute_kernel_config_args(a.device()->arch(), operation_attributes.compute_kernel_config);
    const bool fp32_dest_acc_en = std::get<2>(compute_kernel_config_args);

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
            parallelization_config.per_core_out_matrix_height_ntile / block_config.act_block_h_ntiles,
            block_config.act_block_w_ntiles,
            fp32_dest_acc_en,
            output_tensor.dtype(),
            enable_activation_reuse));

    uint32_t act_block_h_nsubblocks_split = block_config.act_block_h_ntiles;
    uint32_t act_block_h_nsubblocks_split_last = 0;
    if (enable_split_reader) {
        act_block_h_nsubblocks_split_last = block_config.act_block_h_ntiles / 2;
        act_block_h_nsubblocks_split = block_config.act_block_h_ntiles - act_block_h_nsubblocks_split_last;
    }
    const uint32_t act_block_h_datums = act_block_h_ntiles * tt::constants::TILE_HEIGHT;
    const uint32_t act_block_h_datums_split = act_block_h_nsubblocks_split * tt::constants::TILE_HEIGHT;
    const uint32_t act_block_h_datums_split_last = act_block_h_nsubblocks_split_last * tt::constants::TILE_HEIGHT;

    std::vector<std::vector<uint16_t>> conv_sharded_input_top_left_indices =
        ttnn::operations::sliding_window::generate_sliding_window_op_config(
            op_trace_metadata,
            shard_boundaries,
            stride_w,
            true,
            enable_split_reader ? act_block_h_datums_split : act_block_h_datums,
            enable_split_reader ? act_block_h_datums_split_last : 0);

    Tensor conv_reader_indices_tensor = ttnn::operations::sliding_window::construct_on_host_config_tensor(
        conv_sharded_input_top_left_indices, input_parallel_config, config_tensors_in_dram);
    conv_reader_indices_tensor = ttnn::operations::sliding_window::move_config_tensor_to_device(
        conv_reader_indices_tensor, input_parallel_config, block_sharded, a.device(), config_tensors_in_dram);

    log_trace(tt::LogOp, "Conv2D Config Tensor : {}", conv_reader_indices_tensor);

    auto conv_reader_indices_tensor_owner = std::make_shared<Tensor>(std::move(conv_reader_indices_tensor));
    tt::tt_metal::Buffer* conv_reader_indices_buffer = conv_reader_indices_tensor_owner->buffer();
    workload_descriptor.buffers.push_back({conv_reader_indices_tensor_owner, conv_reader_indices_buffer});

    // Single-device op: per-coord program is structurally identical for every
    // coord in `tensor_coords` (conv2d doesn't depend on cluster position).
    // Build the ProgramDescriptor once and copy into each range entry.
    tt::tt_metal::ProgramDescriptor desc =
        build_program_descriptor_sharded(operation_attributes, tensor_args, output_tensor, conv_reader_indices_buffer);

    auto ranges = tensor_coords.ranges();
    workload_descriptor.programs.reserve(ranges.size());
    for (size_t i = 0; i + 1 < ranges.size(); ++i) {
        workload_descriptor.programs.push_back({ranges[i], desc});
    }
    if (!ranges.empty()) {
        workload_descriptor.programs.push_back({ranges.back(), std::move(desc)});
    }
    return workload_descriptor;
}

}  // namespace ttnn::prim
