// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>
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
#include "ttnn/operations/sliding_window/sliding_window.hpp"
#include "ttnn/tensor/types.hpp"
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/host_api.hpp>
#include "ttnn/operations/compute_throttle_utils.hpp"

#include "ttnn/metal2_artifacts.hpp"
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

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

namespace m2 = tt::tt_metal::experimental;

// Shared Metal 2.0 DFB / TensorParameter names + dfb_name_for() live in
// conv2d_op_program_factory_common.hpp (inline) to avoid a unity-build ODR clash
// with the sibling Conv2dWidthShardedProgramFactory .cpp. Only the per-factory
// kernel and semaphore names are declared here.
const m2::KernelSpecName K_READER{"reader"};
const m2::KernelSpecName K_WRITER_SENDER{"writer_mcast_sender"};
const m2::KernelSpecName K_WRITER_RECEIVER{"writer_mcast_receiver"};
const m2::KernelSpecName K_HS_COMPUTE{"compute"};

const m2::SemaphoreSpecName SEM_WEIGHTS_SENDER{"weights_mcast_sender"};
const m2::SemaphoreSpecName SEM_WEIGHTS_RECEIVER{"weights_mcast_receiver"};

}  // namespace

ttnn::device_operation::ProgramArtifacts Conv2dShardedProgramFactory::create_program_spec(
    const Conv2dParams& operation_attributes, const Conv2dInputs& tensor_args, Tensor& output_tensor) {
    const auto& a = tensor_args.a;
    const auto& b = tensor_args.b;

    const auto& ashape = ttnn::Shape(operation_attributes.input_tensor_shape);
    const auto& bias = tensor_args.bias;
    const auto& sliding_window_config = operation_attributes.sliding_window_config;

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

    // ---- Metal 2.0 scope gates ----
    // The in-DRAM config-tensor path threads conv_reader_indices_buffer->address()
    // and ->page_size() through CTAs + a TensorAccessorArgs(buffer) into the reader
    // (consumed under #ifdef CONFIG_TENSOR_IN_DRAM). A raw buffer address through a
    // CTA is an enumerated Metal 2.0 framework blocker; only the L1 config path is
    // ported.
    TT_FATAL(
        !config_tensors_in_dram,
        "Conv2dShardedProgramFactory::create_program_spec only supports the L1 config-tensor path "
        "(config_tensors_in_dram == false); the in-DRAM path remains a Metal 2.0 blocker.");

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
    [[maybe_unused]] uint32_t out_conv_c_blocks = 1;
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

    // ---- Metal 2.0 scope gates (continued) ----
    // BLOCK_SHARDED uses a different reader (the *_2d_mcast_padded* source) and 2D
    // weights mcast; the 1D-depthwise path uses dedicated reader/compute kernels and
    // dest-reuse accumulation. Neither path is ported here. The input_cores !=
    // output_cores case enables the legacy "skip_compute" noop-core RTA path on the
    // compute kernel (height_sharded keeps input_cores == output_cores per the
    // TT_ASSERT above). Gate them all out so the spec build below only handles the
    // dense / height-sharded 1D-mcast path.
    TT_FATAL(
        height_sharded && !block_sharded,
        "Conv2dShardedProgramFactory::create_program_spec only supports the HEIGHT_SHARDED 1D-mcast path; "
        "the BLOCK_SHARDED path remains a Metal 2.0 port follow-up.");
    TT_FATAL(
        !is_conv_1d_depthwise_conv,
        "Conv2dShardedProgramFactory::create_program_spec does not support the 1D-depthwise path "
        "(dedicated reader/compute kernels); it remains a Metal 2.0 port follow-up.");
    TT_FATAL(
        input_cores == output_cores,
        "Conv2dShardedProgramFactory::create_program_spec requires input_cores == output_cores; the "
        "noop-core skip-compute path remains a Metal 2.0 port follow-up.");
    // The legacy 1D path can place the reader/compute on input_cores while the writer
    // mcast receiver also covers inactive cores in all_cores (the receiver early-returns
    // via its `noop` RTA). Reproducing that with the shared-local-DFB invariant (which
    // forces reader+compute+writer to share a node set) requires extra noop-only work
    // units; to keep this first port faithful and tight, require all_cores == input_cores
    // (no inactive grid cores) so the two work units below exactly cover input_cores.
    TT_FATAL(
        all_cores == input_cores,
        "Conv2dShardedProgramFactory::create_program_spec requires all_cores == input_cores (no inactive "
        "grid cores); that noop-core mcast-receiver path remains a Metal 2.0 port follow-up.");

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

    const uint32_t act_num_subblocks = act_block_h_ntiles / out_subblock_h_ntiles;
    const uint32_t act_block_num_tiles = act_block_h_ntiles * act_block_w_ntiles;
    const uint32_t act_subblock_h_ntiles = out_subblock_h_ntiles;
    const uint32_t act_subblock_num_tiles = act_subblock_h_ntiles * act_block_w_ntiles;

    const uint32_t in0_num_blocks_w = conv_act_c_blocks * num_blocks_act_w;

    // bias
    uint32_t bias_ntiles = 0;
    if (has_bias) {
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
    [[maybe_unused]] uint32_t total_num_cores_per_weight_slice = 0;
    {
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
    uint32_t total_active_num_cores = parallelization_config.num_cores_nhw * num_weight_slices_width;

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

    // ---- Op-owned conv_reader_indices tensor ----
    // The host-populated index table that backs the READER_INDICES borrowed-memory
    // DFB. Allocated + uploaded here (L1-sharded in the L1 config path) and parked in
    // ProgramArtifacts::op_owned_tensors so the adapter keeps it alive (stable
    // address) for the cached Program's life. The (act_block_h_datums, last) split
    // must agree exactly with the reader kernel's split, since the generated index
    // tensor is consumed by the reader.
    sliding_window::ParallelConfig input_parallel_config = {
        .grid = a.memory_config().shard_spec().value().grid,
        .shard_scheme = a.memory_config().memory_layout(),
        .shard_orientation = a.memory_config().shard_spec().value().orientation,
    };
    std::vector<uint32_t> op_trace_metadata =
        ttnn::operations::sliding_window::generate_op_trace_metadata(sliding_window_config);
    std::vector<sliding_window::ShardBoundary> shard_boundaries =
        ttnn::operations::sliding_window::generate_shard_boundaries(sliding_window_config);
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

    std::vector<tt::tt_metal::Tensor> op_owned_tensors;
    op_owned_tensors.reserve(1);
    op_owned_tensors.push_back(std::move(conv_reader_indices_tensor));
    // Build all tensor bindings against the VECTOR ELEMENT (identity footgun).
    const Tensor& conv_reader_indices = op_owned_tensors[0];
    tt::tt_metal::Buffer* conv_reader_indices_buffer = conv_reader_indices.buffer();
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

    // 1D depthwise compute uses dest-reuse for accumulation — no MATMUL_PARTIALS CB is allocated.
    const bool partials_cb_uses_output =
        !is_conv_1d_depthwise_conv && get_cb_info_by_name(cb_info, Conv2dCb::MATMUL_PARTIALS).is_globally_allocated;
    log_debug(tt::LogOp, "partials_cb_uses_output: {}", partials_cb_uses_output);

    const bool has_act_second_reader = get_cb_info_by_name(cb_info, Conv2dCb::ACT_SECOND_READER).num_pages != 0;

    // ---- DataflowBufferSpecs ----
    // One DataflowBufferSpec per non-zero-page CBInfo. Globally-allocated CBs become
    // borrowed-memory DFBs (borrowed_from the backing TensorParameter); the rest are
    // Program-local L1 DFBs. CB index assignment is irrelevant here — Metal 2.0 derives
    // indices from bindings — but per-CB entry_size / num_entries / data_format are
    // taken verbatim from CBInfo.
    m2::ProgramSpec spec;
    spec.name = "conv2d_height_sharded";
    for (const auto& cb : cb_info) {
        // ACT_ROW_MAJOR has 0 pages on the height-sharded path (it is the width-/block-
        // sharded pre-tilize buffer). The shared compute kernel conv_bmm_tilize_m2.cpp
        // still references dfb::act_row_major at file scope (inside a dead
        // `if constexpr (!height_sharded)` branch), so the token must exist. Emit a
        // 1-entry placeholder DFB and self-loop it on compute; the kernel never reads/
        // writes it on this path. All other 0-page CBs are skipped, matching
        // emit_cb_descriptors / allocate_cbs.
        const bool act_row_major_placeholder = cb.name == Conv2dCb::ACT_ROW_MAJOR_BFLOAT16 && cb.num_pages == 0;
        if (cb.num_pages == 0 && !act_row_major_placeholder) {
            continue;
        }
        m2::DataflowBufferSpec dfb{
            .unique_id = dfb_name_for(cb.name),
            .entry_size = cb.page_size,
            .num_entries = act_row_major_placeholder ? 1u : cb.num_pages,
            .data_format_metadata = cb.data_format};
        if (cb.is_globally_allocated) {
            switch (cb.name) {
                case Conv2dCb::ACT_SHARDED: dfb.borrowed_from = TP_ACT_SHARDED; break;
                case Conv2dCb::OUT:
                case Conv2dCb::MATMUL_PARTIALS: dfb.borrowed_from = TP_OUT; break;
                case Conv2dCb::READER_INDICES: dfb.borrowed_from = TP_READER_INDICES; break;
                default:
                    TT_THROW("Unexpected globally-allocated CB {} in height-sharded spec", static_cast<int>(cb.name));
            }
        }
        spec.dataflow_buffers.push_back(std::move(dfb));
    }

    // ---- Semaphores (program-local 1D weights mcast) ----
    // The legacy 1D path allocates the weights mcast sender/receiver semaphores on
    // output_cores (== all active cores for height sharded). Skip when weights mcast
    // is skipped (single-core), mirroring the legacy push_semaphore gating.
    if (!skip_weights_mcast) {
        spec.semaphores = {
            m2::SemaphoreSpec{.unique_id = SEM_WEIGHTS_SENDER, .target_nodes = output_cores},
            m2::SemaphoreSpec{.unique_id = SEM_WEIGHTS_RECEIVER, .target_nodes = output_cores},
        };
    }

    // ---- Per-kernel defines (height-sharded 1D path) ----
    std::map<std::string, std::string> reader_defines;
    std::map<std::string, std::string> writer_defines;
    std::map<std::string, std::string> writer_mcast_sender_defines;
    std::map<std::string, std::string> compute_defines;

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

    const uint32_t reader_arg_act_block_h_datums =
        (enable_split_reader ? act_block_h_datums_split : act_block_h_datums);
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

    // mcast NoC selection (1D path)
    const tt::tt_metal::NOC writer_mcast_noc = tt::tt_metal::detail::preferred_noc_for_dram_read(device->arch());
    const tt::tt_metal::NOC reader_noc =
        writer_mcast_noc == tt::tt_metal::NOC::NOC_0 ? tt::tt_metal::NOC::NOC_1 : tt::tt_metal::NOC::NOC_0;

    // mcast rect physical coords (1D)
    const CoreCoord top_left_core = {(std::size_t)0, (std::size_t)0};
    const uint32_t in_num_cores_x = input_cores.bounding_box().end_coord.x + 1;
    const uint32_t in_num_cores_y = input_cores.bounding_box().end_coord.y + 1;
    const CoreCoord bottom_right_core = {(std::size_t)in_num_cores_x - 1, (std::size_t)in_num_cores_y - 1};
    const CoreCoord top_left_core_physical = device->worker_core_from_logical_core(top_left_core);
    const CoreCoord bottom_right_core_physical = device->worker_core_from_logical_core(bottom_right_core);

    auto setup_mcast_args = [&](bool is_noc_0, uint32_t start_x, uint32_t start_y, uint32_t end_x, uint32_t end_y) {
        return is_noc_0 ? std::array<uint32_t, 4>{start_x, start_y, end_x, end_y}
                        : std::array<uint32_t, 4>{end_x, end_y, start_x, start_y};
    };
    const std::array<uint32_t, 4> weights_mcast_coords = setup_mcast_args(
        writer_mcast_noc == tt::tt_metal::NOC::NOC_0,
        top_left_core_physical.x,
        top_left_core_physical.y,
        bottom_right_core_physical.x,
        bottom_right_core_physical.y);

    // ======================= KernelSpecs =======================

    // ---- Reader (RISCV_1, READER) ----
    m2::KernelSpec reader_kernel{
        .unique_id = K_READER,
        .source = std::filesystem::path{"ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/"
                                        "reader_conv_activations_padded_with_halo_3x3_weights_v2_m2.cpp"},
        .compile_time_args =
            {{"dilation_h", dilation_h},
             {"dilation_w", dilation_w},
             {"stride_w", stride_w},
             {"conv_act_c_read_bytes", conv_act_c_read_bytes},
             {"window_outer", window_outer},
             // Reader fills act_block_num_tiles_split when split-reader is on (it does
             // the first half; the writer does the second half). split_reader_cb_shared
             // is block-sharded only, so it is always false on this path.
             {"act_block_num_tiles", enable_split_reader ? act_block_num_tiles_split : act_block_num_tiles},
             {"weight_size_h", filter_h},
             {"weight_size_w", filter_w},
             {"conv_act_size_w_padded", conv_act_size_w + pad_w},
             {"act_block_w_extra_align_bytes", act_block_w_extra_align_bytes},
             {"act_num_blocks_h", num_blocks_act_h_per_core},
             {"needs_act_block_zero_out", (uint32_t)needs_act_block_zero_out},
             {"split_reader_enabled", (uint32_t)enable_split_reader},
             {"activation_reuse_enabled", (uint32_t)enable_activation_reuse},
             // activation-reuse args (dummy 0s when disabled, mirroring the legacy
             // height-sharded path which still emits 8 reuse slots)
             {"act_reuse_cb_tiles", enable_activation_reuse ? activation_reuse_config.act_cb_num_tiles_split : 0u},
             {"act_block_w_tiles", enable_activation_reuse ? act_block_w_ntiles : 0u},
             {"readers_process_full_image_widths",
              enable_activation_reuse ? (uint32_t)activation_reuse_config.readers_process_full_image_widths : 0u},
             {"image_width_tiles", enable_activation_reuse ? activation_reuse_config.image_width_tiles : 0u},
             {"output_image_width", enable_activation_reuse ? output_image_width : 0u},
             {"window_reuse_offset", enable_activation_reuse ? activation_reuse_config.reuse_window_offset : 0u},
             {"need_to_push_remaining_tiles",
              enable_activation_reuse ? (uint32_t)(activation_reuse_config.num_cores_with_non_meaningful_work > 0)
                                      : 0u},
             {"single_core_processes_multiple_batches",
              enable_activation_reuse ? (uint32_t)activation_reuse_config.single_core_processes_multiple_batches : 0u}},
        .runtime_arg_schema = {.runtime_arg_names = {"core_index", "remaining_tiles_to_push"}},
        .hw_config = m2::DataMovementHardwareConfig{.role = m2::DataMovementRoleHint::READER},
    };
    reader_kernel.dfb_bindings = {
        // ACT produced by the reader; consumed by compute.
        m2::DFBBinding{
            .dfb_spec_name = DFB_ACT, .accessor_name = "act", .endpoint_type = m2::DFBEndpointType::PRODUCER},
        // ACT_SHARDED / READER_INDICES: base-pointer address sources (no FIFO peer) ->
        // self-loop to satisfy the producer+consumer invariant.
        m2::DFBBinding{
            .dfb_spec_name = DFB_ACT_SHARDED,
            .accessor_name = "act_sharded",
            .endpoint_type = m2::DFBEndpointType::PRODUCER},
        m2::DFBBinding{
            .dfb_spec_name = DFB_ACT_SHARDED,
            .accessor_name = "act_sharded",
            .endpoint_type = m2::DFBEndpointType::CONSUMER},
        m2::DFBBinding{
            .dfb_spec_name = DFB_READER_INDICES,
            .accessor_name = "reader_indices",
            .endpoint_type = m2::DFBEndpointType::PRODUCER},
        m2::DFBBinding{
            .dfb_spec_name = DFB_READER_INDICES,
            .accessor_name = "reader_indices",
            .endpoint_type = m2::DFBEndpointType::CONSUMER},
        // L1_ARRAY is an allocated 1-page scratch CB that no height-sharded kernel
        // references (legacy CTA 26 is skipped by this reader). Self-loop it on the
        // reader purely to satisfy the producer+consumer invariant; the kernel never
        // constructs dfb::l1_array. Candidate for the forthcoming Metal 2.0 scratchpad.
        m2::DFBBinding{
            .dfb_spec_name = DFB_L1_ARRAY, .accessor_name = "l1_array", .endpoint_type = m2::DFBEndpointType::PRODUCER},
        m2::DFBBinding{
            .dfb_spec_name = DFB_L1_ARRAY, .accessor_name = "l1_array", .endpoint_type = m2::DFBEndpointType::CONSUMER},
    };
    reader_kernel.tensor_bindings = {
        m2::TensorBinding{.tensor_parameter_name = TP_ACT_SHARDED, .accessor_name = "act_sharded"},
        m2::TensorBinding{.tensor_parameter_name = TP_READER_INDICES, .accessor_name = "reader_indices"},
    };
    {
        m2::KernelSpec::CompilerOptions::Defines d;
        for (const auto& [k, v] : reader_defines) {
            d.insert({k, v});
        }
        reader_kernel.compiler_options.defines = std::move(d);
    }

    // ---- Weights/writer mcast sender (RISCV_0, WRITER) ----
    // Shared writer CT args (sender + receiver). Only the slots each kernel reads are
    // emitted as named CTAs (the legacy positional list left gaps the kernels skip).
    auto add_writer_ctas = [&](m2::KernelSpec& k) {
        k.compile_time_args = {
            {"num_blocks_weight_h", num_blocks_act_w},
            {"weight_block_num_tiles", weight_block_num_tiles},
            {"weight_block_height_num_outer", conv_act_c_blocks},
            {"weight_block_height_ntiles", weight_block_h_ntiles},
            {"weight_block_width_ntiles", weight_block_w_ntiles},
            {"weight_stride_h", weight_matrix_width_ntiles},
            {"weight_next_block_stride_h", weight_matrix_width_ntiles * weight_block_h_ntiles},
            {"bias_ntiles", bias_ntiles_per_core},
            {"out_num_blocks_h", num_blocks_act_h_per_core},
            {"fuse_bias", (uint32_t)has_bias},
            {"split_reader_enabled", (uint32_t)enable_split_reader},
            {"activation_reuse_enabled", (uint32_t)(enable_activation_reuse && height_sharded)},
            // Split reader args
            {"act_block_num_tiles", act_block_num_tiles_split_last},
            {"conv_act_c_read_bytes", conv_act_c_read_bytes},
            {"weight_size_w", filter_w},
            {"conv_act_size_w_padded", conv_act_size_w + pad_w},
            {"act_block_w_extra_align_bytes", act_block_w_extra_align_bytes},
            {"needs_act_block_zero_out", (uint32_t)needs_act_block_zero_out},
            {"dilation_h", dilation_h},
            {"dilation_w", dilation_w},
            {"stride_w", stride_w},
            {"weights_size_h", filter_h},
            // Activation reuse args (dummy 0s when disabled; mirrors legacy height-sharded)
            {"act_reuse_cb_tiles", enable_activation_reuse ? activation_reuse_config.act_cb_num_tiles_split_last : 0u},
            {"act_block_w_tiles", enable_activation_reuse ? act_block_w_ntiles : 0u},
            {"readers_process_full_image_widths",
             enable_activation_reuse ? (uint32_t)activation_reuse_config.readers_process_full_image_widths : 0u},
            {"image_width_tiles", enable_activation_reuse ? activation_reuse_config.image_width_tiles : 0u},
            {"output_image_width", enable_activation_reuse ? output_image_width : 0u},
            {"window_reuse_offset", enable_activation_reuse ? activation_reuse_config.reuse_window_offset : 0u},
            {"need_to_push_remaining_tiles",
             enable_activation_reuse ? (uint32_t)(activation_reuse_config.num_cores_with_non_meaningful_work > 0) : 0u},
            {"single_core_processes_multiple_batches",
             enable_activation_reuse ? (uint32_t)activation_reuse_config.single_core_processes_multiple_batches : 0u}};
    };

    // Conditional DFB / tensor bindings shared by sender + receiver writers.
    auto add_writer_bindings = [&](m2::KernelSpec& k, bool is_sender) {
        k.dfb_bindings = {
            m2::DFBBinding{
                .dfb_spec_name = DFB_WEIGHTS,
                .accessor_name = "weights",
                .endpoint_type = m2::DFBEndpointType::PRODUCER},
            // ACT_SHARDED / READER_INDICES base-pointer address sources only when split
            // reader is enabled; self-loop to satisfy producer+consumer invariant.
            m2::DFBBinding{
                .dfb_spec_name = DFB_ACT_SHARDED,
                .accessor_name = "act_sharded",
                .endpoint_type = m2::DFBEndpointType::PRODUCER},
            m2::DFBBinding{
                .dfb_spec_name = DFB_ACT_SHARDED,
                .accessor_name = "act_sharded",
                .endpoint_type = m2::DFBEndpointType::CONSUMER},
            m2::DFBBinding{
                .dfb_spec_name = DFB_READER_INDICES,
                .accessor_name = "reader_indices",
                .endpoint_type = m2::DFBEndpointType::PRODUCER},
            m2::DFBBinding{
                .dfb_spec_name = DFB_READER_INDICES,
                .accessor_name = "reader_indices",
                .endpoint_type = m2::DFBEndpointType::CONSUMER},
        };
        // BIAS / ACT_SECOND_READER CBs are produced by the writer kernels when bound
        // (conditionally, gated by FUSE_BIAS / SECOND_READER_PRESENT defines on the
        // kernel side). Both writers reserve/push these CBs.
        if (has_bias) {
            k.dfb_bindings.push_back(m2::DFBBinding{
                .dfb_spec_name = DFB_BIAS, .accessor_name = "bias", .endpoint_type = m2::DFBEndpointType::PRODUCER});
        }
        if (has_act_second_reader) {
            k.dfb_bindings.push_back(m2::DFBBinding{
                .dfb_spec_name = DFB_ACT_SECOND_READER,
                .accessor_name = "act_second_reader",
                .endpoint_type = m2::DFBEndpointType::PRODUCER});
            k.dfb_bindings.push_back(m2::DFBBinding{
                .dfb_spec_name = DFB_ACT_SECOND_READER,
                .accessor_name = "act_second_reader",
                .endpoint_type = m2::DFBEndpointType::CONSUMER});
        }
        // ACT_SHARDED / READER_INDICES base pointers are read by both writers (split
        // reader); the weights/bias DRAM TensorAccessors are constructed only by the
        // sender (the receiver gets its weights via mcast), so bind those only there.
        k.tensor_bindings = {
            m2::TensorBinding{.tensor_parameter_name = TP_ACT_SHARDED, .accessor_name = "act_sharded"},
            m2::TensorBinding{.tensor_parameter_name = TP_READER_INDICES, .accessor_name = "reader_indices"},
        };
        if (is_sender) {
            k.tensor_bindings.push_back(
                m2::TensorBinding{.tensor_parameter_name = TP_WEIGHTS, .accessor_name = "weights"});
            if (has_bias) {
                k.tensor_bindings.push_back(
                    m2::TensorBinding{.tensor_parameter_name = TP_BIAS, .accessor_name = "bias"});
            }
        }
        if (!skip_weights_mcast) {
            k.semaphore_bindings = {
                m2::SemaphoreBinding{
                    .semaphore_spec_name = SEM_WEIGHTS_SENDER, .accessor_name = "weights_mcast_sender"},
                m2::SemaphoreBinding{
                    .semaphore_spec_name = SEM_WEIGHTS_RECEIVER, .accessor_name = "weights_mcast_receiver"},
            };
        }
    };

    m2::KernelSpec writer_sender_kernel{
        .unique_id = K_WRITER_SENDER,
        .source =
            std::filesystem::path{"ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/"
                                  "reader_writer_tiled_out_1d_mcast_sender_conv_weights_tiled_col_to_rm_blocks_m2.cpp"},
        .runtime_arg_schema =
            {.runtime_arg_names =
                 {"out_start_tile_id_w",
                  "bias_tile_offset",
                  "mcast_noc_x_start",
                  "mcast_noc_y_start",
                  "mcast_noc_x_end",
                  "mcast_noc_y_end",
                  "weights_mcast_num_dests",
                  "weights_mcast_num_cores"}},
        .hw_config = m2::DataMovementHardwareConfig{.role = m2::DataMovementRoleHint::WRITER},
    };
    add_writer_ctas(writer_sender_kernel);
    add_writer_bindings(writer_sender_kernel, /*is_sender=*/true);
    if (enable_activation_reuse) {
        writer_sender_kernel.runtime_arg_schema.runtime_arg_names.push_back("remaining_tiles_to_push");
    }
    {
        m2::KernelSpec::CompilerOptions::Defines d;
        for (const auto& [k, v] : writer_mcast_sender_defines) {
            d.insert({k, v});
        }
        if (has_bias) {
            d.insert({"FUSE_BIAS", "1"});
        }
        if (has_act_second_reader) {
            d.insert({"SECOND_READER_PRESENT", "1"});
        }
        writer_sender_kernel.compiler_options.defines = std::move(d);
    }

    m2::KernelSpec writer_receiver_kernel{
        .unique_id = K_WRITER_RECEIVER,
        .source =
            std::filesystem::path{
                "ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/"
                "reader_writer_tiled_out_1d_mcast_receiver_conv_weights_tiled_col_to_rm_blocks_m2.cpp"},
        .runtime_arg_schema =
            {.runtime_arg_names = {"noop", "weights_mcast_sender_noc_x", "weights_mcast_sender_noc_y"}},
        .hw_config = m2::DataMovementHardwareConfig{.role = m2::DataMovementRoleHint::WRITER},
    };
    add_writer_ctas(writer_receiver_kernel);
    add_writer_bindings(writer_receiver_kernel, /*is_sender=*/false);
    if (enable_activation_reuse) {
        writer_receiver_kernel.runtime_arg_schema.runtime_arg_names.push_back("remaining_tiles_to_push");
    }
    {
        m2::KernelSpec::CompilerOptions::Defines d;
        for (const auto& [k, v] : writer_defines) {
            d.insert({k, v});
        }
        if (has_bias) {
            d.insert({"FUSE_BIAS", "1"});
        }
        if (has_act_second_reader) {
            d.insert({"SECOND_READER_PRESENT", "1"});
        }
        writer_receiver_kernel.compiler_options.defines = std::move(d);
    }

    // ---- Compute (forked: conv_bmm_tilize_m2.cpp) ----
    m2::KernelSpec compute_kernel{
        .unique_id = K_HS_COMPUTE,
        .source = std::filesystem::path{"ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/conv_bmm_tilize_m2.cpp"},
        .compile_time_args =
            {{"in0_block_w", act_block_w_ntiles},
             {"in0_num_subblocks", act_num_subblocks},
             {"in0_block_num_tiles", act_block_num_tiles},
             {"in0_subblock_num_tiles", act_subblock_num_tiles},
             {"reader_num_h_subblocks",
              enable_split_reader ? act_block_h_ntiles : act_subblock_h_ntiles * act_num_subblocks},
             {"in1_num_subblocks", weight_num_subblocks},
             {"in1_block_num_tiles", weight_block_num_tiles},
             {"in1_block_w", weight_block_w_ntiles},
             {"in0_num_blocks_h", num_blocks_act_h_per_core},
             {"in0_num_blocks_w", in0_num_blocks_w},
             {"in1_num_blocks_w", num_blocks_weight_w_per_core},
             {"out_subblock_h", out_subblock_h_ntiles},
             {"out_subblock_w", out_subblock_w_ntiles},
             {"out_subblock_num_tiles", out_subblock_num_tiles},
             {"height_sharded", (uint32_t)height_sharded},
             {"untilize_out", (uint32_t)untilize_out},
             {"bias_ntiles_w", bias_ntiles_per_core},
             {"partials_cb_uses_output", (uint32_t)partials_cb_uses_output},
             {"in0_nblocks_w_tilize", conv_act_c_blocks},
             {"check_skip_compute", 0u},  // input_cores == output_cores is enforced above
             {"pack_relu", (uint32_t)pack_relu},
             {"packer_untilize", (uint32_t)(weight_block_w_ntiles <= 8)},
             {"packer_l1_acc", (uint32_t)packer_l1_acc_en},
             {"fuse_bias", (uint32_t)has_bias},
             {"split_reader", (uint32_t)enable_split_reader},
             {"activation_reuse", (uint32_t)enable_activation_reuse},
             {"image_width_in_tiles", enable_activation_reuse ? activation_reuse_config.image_width_tiles : 0u},
             {"window_reuse_offset",
              enable_activation_reuse ? activation_reuse_config.reuse_window_offset / COMPUTE_KERNEL_ADDRESS_DIVISOR
                                      : 0u},
             {"tilized_cb_row_offset",
              enable_activation_reuse ? activation_reuse_config.tilized_cb_row_offset / COMPUTE_KERNEL_ADDRESS_DIVISOR
                                      : 0u},
             {"tilized_cb_second_reader_offset",
              enable_activation_reuse
                  ? activation_reuse_config.tilized_cb_second_reader_offset / COMPUTE_KERNEL_ADDRESS_DIVISOR
                  : 0u},
             {"split_reader_cb_shared", 0u}},  // block-sharded only
        .hw_config =
            m2::ComputeHardwareConfig{
                .math_fidelity = math_fidelity,
                .fp32_dest_acc_en = fp32_dest_acc_en,
            },
    };
    compute_kernel.dfb_bindings = {
        m2::DFBBinding{
            .dfb_spec_name = DFB_ACT, .accessor_name = "act", .endpoint_type = m2::DFBEndpointType::CONSUMER},
        // ACT_ROW_MAJOR: dead on the height-sharded path (placeholder DFB) -> self-loop
        // on compute to satisfy the producer+consumer invariant.
        m2::DFBBinding{
            .dfb_spec_name = DFB_ACT_ROW_MAJOR,
            .accessor_name = "act_row_major",
            .endpoint_type = m2::DFBEndpointType::PRODUCER},
        m2::DFBBinding{
            .dfb_spec_name = DFB_ACT_ROW_MAJOR,
            .accessor_name = "act_row_major",
            .endpoint_type = m2::DFBEndpointType::CONSUMER},
        m2::DFBBinding{
            .dfb_spec_name = DFB_ACT_TILIZED,
            .accessor_name = "act_tilized",
            .endpoint_type = m2::DFBEndpointType::PRODUCER},
        m2::DFBBinding{
            .dfb_spec_name = DFB_ACT_TILIZED,
            .accessor_name = "act_tilized",
            .endpoint_type = m2::DFBEndpointType::CONSUMER},
        m2::DFBBinding{
            .dfb_spec_name = DFB_WEIGHTS, .accessor_name = "weights", .endpoint_type = m2::DFBEndpointType::CONSUMER},
        // MATMUL_PARTIALS is a compute-local accumulator (produced and consumed by compute).
        m2::DFBBinding{
            .dfb_spec_name = DFB_MATMUL_PARTIALS,
            .accessor_name = "matmul_partials",
            .endpoint_type = m2::DFBEndpointType::PRODUCER},
        m2::DFBBinding{
            .dfb_spec_name = DFB_MATMUL_PARTIALS,
            .accessor_name = "matmul_partials",
            .endpoint_type = m2::DFBEndpointType::CONSUMER},
        // OUT is borrowed onto the (sharded) output tensor: compute packs results
        // directly into it; no separate writer/FIFO consumer -> self-loop on compute.
        m2::DFBBinding{
            .dfb_spec_name = DFB_OUT, .accessor_name = "out", .endpoint_type = m2::DFBEndpointType::PRODUCER},
        m2::DFBBinding{
            .dfb_spec_name = DFB_OUT, .accessor_name = "out", .endpoint_type = m2::DFBEndpointType::CONSUMER},
    };
    {
        m2::KernelSpec::CompilerOptions::Defines d;
        for (const auto& [k, v] : compute_defines) {
            d.insert({k, v});
        }
        if (has_bias) {
            d.insert({"FUSE_BIAS", "1"});
            compute_kernel.dfb_bindings.push_back(m2::DFBBinding{
                .dfb_spec_name = DFB_BIAS, .accessor_name = "bias", .endpoint_type = m2::DFBEndpointType::CONSUMER});
        }
        if (has_act_second_reader) {
            d.insert({"SECOND_READER_PRESENT", "1"});
            compute_kernel.dfb_bindings.push_back(m2::DFBBinding{
                .dfb_spec_name = DFB_ACT_SECOND_READER,
                .accessor_name = "act_second_reader",
                .endpoint_type = m2::DFBEndpointType::CONSUMER});
        }
        compute_kernel.compiler_options.defines = std::move(d);
    }

    spec.kernels = {reader_kernel, writer_sender_kernel, compute_kernel};
    if (!skip_weights_mcast) {
        spec.kernels.push_back(writer_receiver_kernel);
    }

    // ---- Tensor parameters ----
    spec.tensor_parameters = {
        m2::TensorParameter{.unique_id = TP_ACT_SHARDED, .spec = a.tensor_spec()},
        m2::TensorParameter{.unique_id = TP_WEIGHTS, .spec = b.tensor_spec()},
        m2::TensorParameter{.unique_id = TP_OUT, .spec = output.tensor_spec()},
        m2::TensorParameter{.unique_id = TP_READER_INDICES, .spec = conv_reader_indices.tensor_spec()},
    };
    if (has_bias) {
        spec.tensor_parameters.push_back(m2::TensorParameter{.unique_id = TP_BIAS, .spec = bias.value().tensor_spec()});
    }

    // ---- WorkUnits ----
    // Local DFBs (ACT/ACT_ROW_MAJOR/ACT_TILIZED/WEIGHTS/MATMUL_PARTIALS/OUT/BIAS) are
    // shared by reader+compute and (for weights) the writer kernels, so every kernel
    // that hosts a local DFB must share a WorkUnitSpec covering the same node set.
    // Legacy 1D height-sharded placement: reader on input_cores, sender on the single
    // top-left mcast-sender core, receiver on the rest, compute on input_cores. For the
    // shared-DFB invariant the reader/compute/writer-sender share input_cores; the
    // writer-receiver (when present) covers the remaining cores.
    const CoreRangeSet sender_cores{CoreRange(top_left_core, top_left_core)};
    const CoreRangeSet receiver_cores = skip_weights_mcast ? CoreRangeSet{} : all_cores.subtract(sender_cores);

    std::vector<m2::WorkUnitSpec> work_units;
    work_units.push_back(m2::WorkUnitSpec{
        .name = "conv2d_hs_sender",
        .kernels = {K_READER, K_WRITER_SENDER, K_HS_COMPUTE},
        .target_nodes = sender_cores});
    if (!skip_weights_mcast && receiver_cores.num_cores() > 0) {
        work_units.push_back(m2::WorkUnitSpec{
            .name = "conv2d_hs_receiver",
            .kernels = {K_READER, K_WRITER_RECEIVER, K_HS_COMPUTE},
            .target_nodes = receiver_cores});
    }
    spec.work_units = std::move(work_units);

    // ======================= ProgramRunArgs =======================
    m2::ProgramRunArgs run;
    m2::KernelRunArgs reader_run{.kernel = K_READER};
    m2::KernelRunArgs sender_run{.kernel = K_WRITER_SENDER};
    m2::KernelRunArgs receiver_run{.kernel = K_WRITER_RECEIVER};
    m2::KernelRunArgs compute_run{.kernel = K_HS_COMPUTE};

    // Reader per-core RTAs: core_index (sequential over input_cores) + remaining tiles.
    {
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
                reader_run.runtime_arg_values.push_back(
                    {core, {{"core_index", core_index}, {"remaining_tiles_to_push", reader_remaining_tiles_to_push}}});
                core_index++;
            }
        }
    }

    // Writer-sender RTAs (1D mcast). weight/bias addresses now flow via TensorBinding,
    // so they are dropped from the RTA list. out_start_tile_id_w / bias_tile_offset
    // and the mcast rect + counts + (optional) remaining tiles remain.
    for (const CoreRange& core_range : sender_cores.ranges()) {
        for (const CoreCoord& core : core_range) {
            const uint32_t weight_slice_i = core.y;  // 1D path
            const uint32_t out_start_tile_id_w = weight_slice_i * per_core_out_matrix_width_ntiles;
            const uint32_t bias_tile_offset = out_start_tile_id_w;
            TT_FATAL(
                bias_tile_offset < bias_ntiles || !has_bias,
                "bias_tile_offset {} should be less than bias_ntiles {}",
                bias_tile_offset,
                bias_ntiles);

            m2::KernelRunArgs::RuntimeArgValues vals = {
                {"out_start_tile_id_w", out_start_tile_id_w},
                {"bias_tile_offset", bias_tile_offset},
                {"mcast_noc_x_start", weights_mcast_coords[0]},
                {"mcast_noc_y_start", weights_mcast_coords[1]},
                {"mcast_noc_x_end", weights_mcast_coords[2]},
                {"mcast_noc_y_end", weights_mcast_coords[3]},
                {"weights_mcast_num_dests", total_active_num_cores - 1},
                {"weights_mcast_num_cores", total_num_cores - 1}};
            if (enable_activation_reuse) {
                uint32_t writer_remaining_tiles_to_push = 0;
                if (activation_reuse_config.has_partial_core && core == activation_reuse_config.partial_work_core) {
                    writer_remaining_tiles_to_push =
                        activation_reuse_config.partial_core_writer_remaining_tiles_to_push_to_push;
                } else if (activation_reuse_config.cores_with_non_meaningful_work.contains(core)) {
                    writer_remaining_tiles_to_push = act_block_h_nsubblocks_split_last;
                }
                vals.insert({"remaining_tiles_to_push", writer_remaining_tiles_to_push});
            }
            sender_run.runtime_arg_values.push_back({core, std::move(vals)});
        }
    }

    // Writer-receiver RTAs.
    if (!skip_weights_mcast) {
        for (const CoreRange& core_range : receiver_cores.ranges()) {
            for (const CoreCoord& core : core_range) {
                const bool is_no_op_core = !input_cores.contains(core);
                m2::KernelRunArgs::RuntimeArgValues vals = {
                    {"noop", (uint32_t)is_no_op_core},
                    {"weights_mcast_sender_noc_x", top_left_core_physical.x},
                    {"weights_mcast_sender_noc_y", top_left_core_physical.y}};
                if (enable_activation_reuse) {
                    uint32_t writer_remaining_tiles_to_push = 0;
                    if (activation_reuse_config.has_partial_core && core == activation_reuse_config.partial_work_core) {
                        writer_remaining_tiles_to_push =
                            activation_reuse_config.partial_core_writer_remaining_tiles_to_push_to_push;
                    } else if (activation_reuse_config.cores_with_non_meaningful_work.contains(core)) {
                        writer_remaining_tiles_to_push = act_block_h_nsubblocks_split_last;
                    }
                    vals.insert({"remaining_tiles_to_push", writer_remaining_tiles_to_push});
                }
                receiver_run.runtime_arg_values.push_back({core, std::move(vals)});
            }
        }
    }

    (void)compute_run;  // compute kernel has no per-core RTAs on this path
    // reader_noc was the legacy reader DataMovementConfig NoC; Metal 2.0 derives the
    // reader's NoC from its DataMovementRoleHint, so the explicit value is unused here.
    (void)reader_noc;

    run.kernel_run_args = {reader_run, sender_run, compute_run};
    if (!skip_weights_mcast) {
        run.kernel_run_args.push_back(receiver_run);
    }
    run.tensor_args = {
        {TP_ACT_SHARDED, a.mesh_tensor()},
        {TP_WEIGHTS, b.mesh_tensor()},
        {TP_OUT, output.mesh_tensor()},
        {TP_READER_INDICES, conv_reader_indices.mesh_tensor()},
    };
    if (has_bias) {
        run.tensor_args.insert({TP_BIAS, bias.value().mesh_tensor()});
    }

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec), .run_params = std::move(run), .op_owned_tensors = std::move(op_owned_tensors)};
}

}  // namespace ttnn::prim
