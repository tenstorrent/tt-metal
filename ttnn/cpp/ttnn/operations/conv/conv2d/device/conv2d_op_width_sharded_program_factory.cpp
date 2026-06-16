// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "ttnn/operations/conv/conv2d/conv2d_op_program_factory_common.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_utils.hpp"
#include "ttnn/operations/conv/conv2d/device/conv2d_op_width_sharded_program_factory.hpp"
#include "ttnn/operations/conv/conv2d/device/conv2d_device_operation_types.hpp"
#include "ttnn/operations/sliding_window/sliding_window.hpp"
#include "ttnn/operations/conv/conv2d/conv2d_utils.hpp"
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
namespace m2 = tt::tt_metal::experimental;
using ttnn::operations::conv::conv_skip_mcast;
using ttnn::operations::conv::SkipMcast;

std::pair<std::vector<uint32_t>, std::vector<uint32_t>> compute_opt_conv_activation_as_mm_shape(
    const ttnn::Shape& conv_activation_shape,
    const ttnn::operations::sliding_window::SlidingWindowConfig& sliding_window_config,
    uint32_t num_cores_nhw,
    uint32_t act_block_h_ntiles) {
    uint32_t filter_h = (uint32_t)sliding_window_config.window_hw.first;   // filter_h
    uint32_t filter_w = (uint32_t)sliding_window_config.window_hw.second;  // filter_W
    auto output_shape = sliding_window_config.get_output_shape();
    uint32_t batch_size = output_shape[0];
    uint32_t conv_output_h = output_shape[1];
    uint32_t conv_output_w = output_shape[2];

    // pad height
    uint32_t num_rows = (uint32_t)batch_size * conv_output_h * conv_output_w;
    uint32_t act_block_h_datums = act_block_h_ntiles * tt::constants::TILE_HEIGHT;
    uint32_t num_rows_padded = tt::round_up(num_rows, num_cores_nhw * act_block_h_datums);
    uint32_t num_cols = conv_activation_shape[3] * filter_h * filter_w;
    uint32_t num_cols_padded = tt::round_up(conv_activation_shape[3] * filter_w, tt::constants::TILE_WIDTH) * filter_h;
    return {{1, num_rows_padded, num_cols_padded}, {1, num_rows, num_cols}};
}

namespace {

// Metal 2.0 DFB unique-ids, one per (non-zero-page) Conv2dCb the width-sharded
// kernels reference. Naming mirrors the kernel-side dfb:: accessor tokens.
const m2::DFBSpecName DFB_ACT_SHARDED{"act_sharded"};
const m2::DFBSpecName DFB_ACT{"act"};
const m2::DFBSpecName DFB_ACT_ROW_MAJOR{"act_row_major"};
const m2::DFBSpecName DFB_ACT_TILIZED{"act_tilized"};
const m2::DFBSpecName DFB_WEIGHTS{"weights"};
const m2::DFBSpecName DFB_BIAS{"bias"};
const m2::DFBSpecName DFB_READER_INDICES{"reader_indices"};
const m2::DFBSpecName DFB_L1_ARRAY{"l1_array"};
const m2::DFBSpecName DFB_MATMUL_PARTIALS{"matmul_partials"};
const m2::DFBSpecName DFB_OUT{"out"};

const m2::TensorParamName TP_WEIGHTS{"weights"};
const m2::TensorParamName TP_BIAS{"bias"};
const m2::TensorParamName TP_READER_INDICES{"reader_indices"};
const m2::TensorParamName TP_ACT_SHARDED{"act_sharded"};
const m2::TensorParamName TP_OUT{"out"};

const m2::KernelSpecName K_ACT{"act_reader"};
const m2::KernelSpecName K_WEIGHTS{"weights_reader"};
const m2::KernelSpecName K_COMPUTE{"compute"};

const m2::SemaphoreSpecName SEM_SENDER{"act_mcast_sender"};
const m2::SemaphoreSpecName SEM_RECEIVER{"act_mcast_receiver"};

// Map a Conv2dCb to the DFBSpecName the kernels reference. Used only for the
// (non-zero-page) CBs the width-sharded path actually allocates.
m2::DFBSpecName dfb_name_for(Conv2dCb cb) {
    switch (cb) {
        case Conv2dCb::ACT_SHARDED: return DFB_ACT_SHARDED;
        case Conv2dCb::ACT: return DFB_ACT;
        case Conv2dCb::ACT_ROW_MAJOR_BFLOAT16: return DFB_ACT_ROW_MAJOR;
        case Conv2dCb::ACT_TILIZED: return DFB_ACT_TILIZED;
        case Conv2dCb::WEIGHTS: return DFB_WEIGHTS;
        case Conv2dCb::BIAS: return DFB_BIAS;
        case Conv2dCb::READER_INDICES: return DFB_READER_INDICES;
        case Conv2dCb::L1_ARRAY: return DFB_L1_ARRAY;
        case Conv2dCb::MATMUL_PARTIALS: return DFB_MATMUL_PARTIALS;
        case Conv2dCb::OUT: return DFB_OUT;
        default: TT_THROW("Unexpected Conv2dCb in width-sharded Metal 2.0 spec: {}", static_cast<int>(cb));
    }
}

}  // namespace

ttnn::device_operation::ProgramArtifacts Conv2dWidthShardedProgramFactory::create_program_spec(
    const Conv2dParams& operation_attributes, const Conv2dInputs& tensor_args, Tensor& output_tensor) {
    const auto& a = tensor_args.a;
    const auto& b = tensor_args.b;

    const auto& ashape = ttnn::Shape(operation_attributes.input_tensor_shape);
    const auto& bias = tensor_args.bias;
    const auto& sliding_window_config = operation_attributes.sliding_window_config;

    const auto output_channels = operation_attributes.output_channels;
    const auto untilize_out = operation_attributes.untilize_out;
    const auto has_bias = operation_attributes.has_bias;
    const auto& fused_activation = operation_attributes.activation;
    const auto& parallelization_config = operation_attributes.parallelization_config;
    const auto& block_config = operation_attributes.block_config;
    auto& output = output_tensor;
    const auto& compute_kernel_config = operation_attributes.compute_kernel_config;
    const auto enable_act_double_buffer = operation_attributes.enable_act_double_buffer;
    const auto enable_weights_double_buffer = operation_attributes.enable_weights_double_buffer;
    const auto config_tensors_in_dram = operation_attributes.config_tensors_in_dram;

    // The in-DRAM config-tensor path threads conv_reader_indices_buffer->address()
    // and ->page_size() through CTAs into the activation kernel (consumed under
    // #ifdef CONFIG_TENSOR_IN_DRAM). A raw buffer address through a CTA is an
    // enumerated Metal 2.0 framework blocker; only the L1 config path is ported.
    TT_FATAL(
        !config_tensors_in_dram,
        "Conv2dWidthShardedProgramFactory::create_program_spec only supports the L1 config-tensor path "
        "(config_tensors_in_dram == false); the in-DRAM path remains a Metal 2.0 blocker.");

    tt::tt_metal::IDevice* device = a.device();
    TT_FATAL(a.layout() == tt::tt_metal::Layout::ROW_MAJOR, "Conv activation should be in row major layout");
    TT_FATAL(a.memory_config().is_sharded(), "Conv activation must be sharded.");
    TT_FATAL(output_channels <= b.padded_shape()[3], "Invalid weight shape. Incorrect weight tensor.");
    uint32_t act_block_h_ntiles = block_config.act_block_h_ntiles;
    uint32_t act_block_w_ntiles = block_config.act_block_w_ntiles;
    uint32_t weight_block_w_ntiles = parallelization_config.per_core_out_matrix_width_ntile;
    uint32_t out_block_h_ntiles = parallelization_config.per_core_out_matrix_height_ntile;
    uint32_t out_subblock_h_ntiles = block_config.out_subblock_h_ntiles;
    uint32_t out_subblock_w_ntiles = block_config.out_subblock_w_ntiles;

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
    TT_FATAL(b.layout() == tt::tt_metal::Layout::TILE, "Conv weights should be in tiled layout");
    TT_FATAL(b.padded_shape()[0] == 1, "Conv weight matrix shape is invalid");
    TT_FATAL(b.padded_shape()[1] == 1, "Conv weight matrix shape is invalid");
    uint32_t weight_matrix_height = b.padded_shape()[2];
    uint32_t weight_matrix_width = b.padded_shape()[3];
    uint32_t weight_matrix_width_ntiles = weight_matrix_width / tt::constants::TILE_WIDTH;

    const auto shard_shape = a.shard_spec().value().shape;

    CoreRangeSet input_cores = a.memory_config().shard_spec().value().grid;
    CoreRangeSet output_cores = output.memory_config().shard_spec().value().grid;
    CoreRangeSet all_cores = output.memory_config().shard_spec().value().grid;
    if (input_cores.num_cores() > output_cores.num_cores()) {
        all_cores = input_cores;
    }
    CoreRange all_reader_cores = all_cores.bounding_box();
    auto input_num_cores = input_cores.num_cores();
    auto output_num_cores = output_cores.num_cores();

    // parallelization config
    const auto& p_config = parallelization_config;
    uint32_t input_channels_padded = shard_shape[1] * input_num_cores;
    TT_FATAL(input_channels_padded >= ashape[3], "Incorrect padding of input channels!");
    // check is for 16-byte alignment
    TT_FATAL(
        input_channels_padded % 16 == 0,
        "Expected input channels to be padded for 16 byte alignment in L1");  // TODO: For bfp16, check if its divisible
                                                                              // by 8 not 16.

    ttnn::Shape ashape_with_channels_padded({ashape[0], ashape[1], ashape[2], input_channels_padded});

    uint32_t conv_act_size_w = ashape_with_channels_padded[2];
    uint32_t conv_act_size_c = ashape_with_channels_padded[3];

    const uint32_t filter_h = (uint32_t)sliding_window_config.window_hw.first;   // filter_h
    const uint32_t filter_w = (uint32_t)sliding_window_config.window_hw.second;  // filter_W
    const uint32_t stride_w = sliding_window_config.is_transpose ? 1 : (uint32_t)sliding_window_config.stride_hw.second;
    const uint32_t dilation_h = (uint32_t)sliding_window_config.dilation_hw.first;
    const uint32_t dilation_w = (uint32_t)sliding_window_config.dilation_hw.second;

    uint32_t pad_w = (uint32_t)sliding_window_config.get_pad_w();

    [[maybe_unused]] uint32_t input_size_w = conv_act_size_w + pad_w;
    if (sliding_window_config.is_transpose) {
        auto input_shape = sliding_window_config.get_transposed_full_input_shape();
        input_size_w = input_shape[2];
    }

    // Compute the 2d matrix shape
    auto [act_matrix_shape, act_matrix_shape_unpadded] = compute_opt_conv_activation_as_mm_shape(
        ashape_with_channels_padded, sliding_window_config, parallelization_config.num_cores_nhw, out_block_h_ntiles);
    TT_FATAL(
        act_matrix_shape.size() == 3,
        "Activation matrix shape must have 3 dimensions but got {}",
        act_matrix_shape.size());
    TT_FATAL(act_matrix_shape[0] == 1, "Activation matrix first dimension must be 1 but got {}", act_matrix_shape[0]);
    uint32_t act_matrix_height = (uint32_t)act_matrix_shape[1];
    uint32_t act_matrix_width = (uint32_t)act_matrix_shape[2];

    // TODO: Move all these TT_FATALs/checks to validate?

    if (has_bias) {
        // Tensor bias is of shape {output_channels}
        TT_FATAL(bias.has_value(), "Bias tensor must be provided when has_bias is true");
        TT_FATAL(bias.value().buffer() != nullptr, "Bias tensor buffer must not be null");
        auto bias_shape_without_padding = bias.value().logical_shape();
        TT_FATAL(bias_shape_without_padding[0] == 1, "Bias should have batch == 1");
    }

    // Normal matrix shape check
    TT_FATAL(act_matrix_width == weight_matrix_height, "The width of tensor a needs to match the height of tensor b");

    // Tile size divisibility checks
    TT_FATAL(
        act_matrix_height % tt::constants::TILE_HEIGHT == 0, "Height of activation matrix needs to be divisible by 32");
    TT_FATAL(
        act_matrix_width % tt::constants::TILE_WIDTH == 0, "Width of activation matrix needs to be divisible by 32");
    TT_FATAL(
        weight_matrix_height % tt::constants::TILE_HEIGHT == 0, "Height of weight matrix needs to be divisible by 32");
    TT_FATAL(
        weight_matrix_width % tt::constants::TILE_WIDTH == 0, "Width of weight matrix needs to be divisible by 32");

    // Device compatibility checks
    TT_FATAL(
        a.storage_type() == tt::tt_metal::StorageType::DEVICE && b.storage_type() == tt::tt_metal::StorageType::DEVICE,
        "Operands to large matmul need to be on device!");
    TT_FATAL(a.device() == b.device(), "Operands to conv need to be on the same device!");
    TT_FATAL(
        a.buffer() != nullptr && b.buffer() != nullptr, "Operands to conv need to be allocated in buffers on device!");
    if (has_bias) {
        TT_FATAL(bias.value().storage_type() == tt::tt_metal::StorageType::DEVICE, "Bias should be on device");
        TT_FATAL(bias.value().device() == a.device(), "Bias should be on the same device as act tensor");
    }

    // Convert tensor dims to tile dims
    uint32_t act_matrix_height_ntiles = act_matrix_height / tt::constants::TILE_HEIGHT;
    uint32_t act_matrix_width_ntiles = act_matrix_width / tt::constants::TILE_WIDTH;

    TT_FATAL(
        act_matrix_height_ntiles % act_block_h_ntiles == 0,
        "act_matrix_height_ntiles {} should be divisible by act_block_h_ntiles {}",
        act_matrix_height_ntiles,
        act_block_h_ntiles);
    TT_FATAL(
        act_matrix_width_ntiles % act_block_w_ntiles == 0,
        "act_matrix_width_ntiles {} should be divisible by act_block_w_ntiles {}",
        act_matrix_width_ntiles,
        act_block_w_ntiles);
    TT_FATAL(
        weight_matrix_width_ntiles % weight_block_w_ntiles == 0,
        "weight_+matrix_width_ntiles {} should be divisible by weight_block_w_ntiles {}",
        weight_matrix_width_ntiles,
        weight_block_w_ntiles);
    TT_FATAL(
        act_matrix_height_ntiles % out_block_h_ntiles == 0,
        "act_matrix_height_ntiles {} should be divisible by out_block_h_ntiles {}",
        act_matrix_height_ntiles,
        out_block_h_ntiles);

    uint32_t num_blocks_act_h = act_matrix_height_ntiles / act_block_h_ntiles;
    uint32_t num_blocks_act_w = act_matrix_width_ntiles / act_block_w_ntiles;
    uint32_t num_blocks_weight_w = weight_matrix_width_ntiles / weight_block_w_ntiles;

    TT_FATAL(
        num_blocks_act_w % input_num_cores == 0,
        "Number of Act Blocks along the Width {} should be divisible by the number of cores {}",
        num_blocks_act_w,
        input_num_cores);

    TT_FATAL(
        num_blocks_act_w % input_num_cores == 0,
        "Number of Act Blocks along the Width {} should be divisible by the number of cores {}",
        num_blocks_act_w,
        input_num_cores);
    uint32_t per_core_num_blocks_act_w = num_blocks_act_w / input_num_cores;

    // act block info
    uint32_t act_block_h_datums = act_matrix_height / num_blocks_act_h;

    const uint32_t act_block_num_tiles = act_block_h_ntiles * act_block_w_ntiles;

    // weight block info
    uint32_t weight_block_w_datums = weight_matrix_width / num_blocks_weight_w;
    TT_FATAL(
        weight_block_w_ntiles % out_subblock_w_ntiles == 0,
        "weight_block_w_ntiles {} should be divisible by out_subblock_w_ntiles {}",
        weight_block_w_ntiles,
        out_subblock_w_ntiles);
    uint32_t weight_num_subblocks = weight_block_w_ntiles / out_subblock_w_ntiles;
    uint32_t weight_block_num_tiles = weight_block_w_ntiles * act_block_w_ntiles;
    uint32_t weight_block_in_channels_ntiles =
        input_channels_padded / (32 * input_num_cores * per_core_num_blocks_act_w);
    TT_FATAL(
        input_channels_padded >= (tt::constants::TILE_HEIGHT * input_num_cores),
        "input_channels_padded {} should be greater than or equal to TILE_HEIGHT * input_num_cores {}",
        input_channels_padded,
        tt::constants::TILE_HEIGHT * input_num_cores);
    TT_FATAL(
        input_channels_padded % (tt::constants::TILE_HEIGHT * input_num_cores) == 0,
        "input_channels_padded {} should be divisible by TILE_HEIGHT * input_num_cores {}",
        input_channels_padded,
        tt::constants::TILE_HEIGHT * input_num_cores);

    // writer of conv op partially removes padding on the width
    // it removes the padding done for block width but it doesn't remove padding done for tiled width
    uint32_t output_channels_padded_to_tile_width =
        tt::round_up(output_channels, output_num_cores * tt::constants::TILE_WIDTH);
    TT_FATAL(
        output_channels_padded_to_tile_width <= weight_matrix_width,
        "output_channels_padded_to_tile_width {} should be less than or equal to weight_matrix_width {}",
        output_channels_padded_to_tile_width,
        weight_matrix_width);
    uint32_t num_blocks_output_w =
        (uint32_t)std::ceil((double)output_channels_padded_to_tile_width / (double)weight_block_w_datums);
    uint32_t last_block_width_datums = (output_channels_padded_to_tile_width % weight_block_w_datums == 0)
                                           ? weight_block_w_datums
                                           : (output_channels_padded_to_tile_width % weight_block_w_datums);
    TT_FATAL(
        last_block_width_datums % tt::constants::TILE_WIDTH == 0,
        "last_block_width_datums {} should be divisible by TILE_WIDTH {}",
        last_block_width_datums,
        tt::constants::TILE_WIDTH);

    // sanity check
    TT_FATAL(
        num_blocks_output_w == num_blocks_weight_w,
        "num_blocks_output_w {} should be equal to num_blocks_weight_w {}",
        num_blocks_output_w,
        num_blocks_weight_w);

    TT_FATAL(output.buffer() != nullptr, "Output buffer should be allocated on device!");

    uint32_t out_subblock_num_tiles = out_subblock_h_ntiles * out_subblock_w_ntiles;
    TT_FATAL(out_subblock_num_tiles <= 8, "Need to ensure that matmul partials fit in dst");

    TT_FATAL(
        act_block_h_ntiles % out_subblock_h_ntiles == 0,
        "act_block_h_ntiles {} should be divisible by out_subblock_h_ntiles {}",
        act_block_h_ntiles,
        out_subblock_h_ntiles);
    uint32_t act_num_subblocks = act_block_h_ntiles / out_subblock_h_ntiles;
    uint32_t act_subblock_h_ntiles = out_subblock_h_ntiles;
    uint32_t act_subblock_num_tiles = act_subblock_h_ntiles * act_block_w_ntiles;

    // bias
    uint32_t bias_ntiles = 0;
    if (has_bias) {
        bias_ntiles = weight_block_w_ntiles;
    }

    uint32_t num_blocks_act_h_per_core =
        (p_config.per_core_out_matrix_height_ntile + act_block_h_ntiles - 1) / act_block_h_ntiles;
    uint32_t num_blocks_weight_w_per_core = p_config.per_core_out_matrix_width_ntile / weight_block_w_ntiles;

    std::map<std::string, std::string> reader_defines;

    uint32_t conv_act_c_read_bytes = conv_act_size_c * a.element_size() / (input_num_cores * per_core_num_blocks_act_w);

    bool tilize_in0 = false;

    // Select preferred NoCs for DRAM operations based on architecture
    // Must be done early to use in multicast coordinate setup
    // weights_kernel (RISCV_1) reads weights/bias from DRAM -> use preferred read NoC
    // act_kernel (RISCV_0) primarily does L1 reads and multicasts -> use preferred write NoC
    // This optimizes NoC bandwidth by separating DRAM reads from L1/multicast operations
    tt::tt_metal::NOC weights_noc = tt::tt_metal::detail::preferred_noc_for_dram_read(device->arch());
    tt::tt_metal::NOC act_noc = tt::tt_metal::detail::preferred_noc_for_dram_write(device->arch());

    log_debug(
        tt::LogOp,
        "Conv2D NoC selection: act_noc={}, weights_noc={} for arch={}",
        (uint32_t)act_noc,
        (uint32_t)weights_noc,
        (uint32_t)device->arch());

    CoreCoord act_mcast_start_core_logical(0, 0);
    CoreCoord act_mcast_end_core_logical(all_cores.bounding_box().end_coord.x, all_cores.bounding_box().end_coord.y);
    auto act_mcast_start = device->worker_core_from_logical_core(act_mcast_start_core_logical);
    auto act_mcast_end = device->worker_core_from_logical_core(act_mcast_end_core_logical);

    // Swap multicast coordinates if using NOC_1 for proper addressing
    // NOC_0 and NOC_1 have inverted coordinate systems on some architectures
    if (act_noc == tt::tt_metal::NOC::NOC_1) {
        std::swap(act_mcast_start, act_mcast_end);
        log_debug(
            tt::LogOp,
            "Conv2D: Swapped mcast coords for NOC_1: start=({},{}), end=({},{})",
            act_mcast_start.x,
            act_mcast_start.y,
            act_mcast_end.x,
            act_mcast_end.y);
    }

    TT_FATAL(act_block_h_datums % 2 == 0, "2 Indices are packed in one uint32_t word.");

    std::map<std::string, std::string> writer_defines;
    std::map<std::string, std::string> writer_mcast_sender_defines;
    std::map<std::string, std::string> compute_defines;

    const SkipMcast skip_mcast = conv_skip_mcast(parallelization_config, a.memory_config().memory_layout());
    const bool skip_activation_mcast = skip_mcast.skip_activation_mcast;
    const bool skip_weights_mcast = skip_mcast.skip_weights_mcast;
    if (skip_activation_mcast) {
        reader_defines["SKIP_MCAST"] = "1";
    }
    if (skip_weights_mcast) {
        // NOTE (behavior-preserving): the legacy width-sharded factory populated this
        // map but never applied it to any kernel — skip_weights_mcast never reaches
        // the weights kernel. Preserved verbatim (not "fixed"); flagged in the port
        // report for the op owner.
        writer_mcast_sender_defines["SKIP_MCAST"] = "1";
    }
    (void)writer_mcast_sender_defines;

    bool pack_relu = fused_activation.has_value() && fused_activation.value().op_type == unary::UnaryOpType::RELU;
    if (fused_activation.has_value() && !pack_relu) {
        compute_defines.merge(ttnn::operations::unary::utils::get_defines(
            fused_activation.value().op_type, fused_activation.value().params, "ACTIVATION", "i"));
    }

    ttnn::operations::compute_throttle_utils::throttle_mm_perf(
        device->arch(), output_cores.num_cores(), compute_defines, ttnn::get_throttle_level(compute_kernel_config));

    for (auto elem : compute_defines) {
        log_debug(tt::LogOp, "compute_defines: {} = {}", elem.first, elem.second);
    }

    const uint32_t output_image_width = sliding_window_config.get_output_shape()[2];
    Conv2dConfig conv_config = Conv2dConfig{
        .weights_dtype = b.dtype(),
        .config_tensors_in_dram = config_tensors_in_dram,
        .shard_layout = a.memory_config().memory_layout(),
        .output_layout = (untilize_out ? Layout::ROW_MAJOR : Layout::TILE),
        .enable_act_double_buffer = enable_act_double_buffer,
        .enable_weights_double_buffer = enable_weights_double_buffer};

    // ---- Op-owned conv_reader_indices tensor ----
    // The host-populated index table that backs the READER_INDICES borrowed-memory
    // DFB. Allocated + uploaded here (L1-sharded in the L1 config path) and parked
    // in ProgramArtifacts::op_owned_tensors so the adapter keeps it alive (stable
    // address) for the cached Program's life.
    ttnn::operations::sliding_window::ParallelConfig parallel_config{
        .grid = a.shard_spec().value().grid,
        .shard_scheme = a.memory_config().memory_layout(),
        .shard_orientation = a.shard_spec().value().orientation};
    std::vector<uint32_t> op_trace_metadata =
        ttnn::operations::sliding_window::generate_op_trace_metadata(sliding_window_config);
    std::vector<sliding_window::ShardBoundary> shard_boundaries =
        ttnn::operations::sliding_window::generate_shard_boundaries(sliding_window_config);
    std::vector<std::vector<uint16_t>> conv_sharded_input_top_left_indices =
        ttnn::operations::sliding_window::generate_sliding_window_op_config(
            op_trace_metadata, shard_boundaries, stride_w, true, act_block_h_datums, 0);
    Tensor conv_reader_indices_tensor = ttnn::operations::sliding_window::construct_on_host_config_tensor(
        conv_sharded_input_top_left_indices, parallel_config, config_tensors_in_dram);
    conv_reader_indices_tensor = ttnn::operations::sliding_window::move_config_tensor_to_device(
        conv_reader_indices_tensor, parallel_config, false, a.device(), config_tensors_in_dram);
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
        p_config,
        b.padded_shape(),
        {filter_h, filter_w},
        {sliding_window_config.input_hw.first, sliding_window_config.input_hw.second},
        {dilation_h, dilation_w},
        conv_config,
        a.dtype(),
        output.dtype(),
        a.memory_config().shard_spec().value().shape,
        output_image_width,
        has_bias,
        false,
        skip_activation_mcast,
        input_channels_padded,
        reader_indices_actual_page_size);

    const bool partials_cb_uses_output = get_cb_info_by_name(cb_info, Conv2dCb::MATMUL_PARTIALS).is_globally_allocated;

    // ---- ProgramSpec (immutable) ----
    m2::ProgramSpec spec;
    spec.name = "conv2d_width_sharded";

    // One DataflowBufferSpec per non-zero-page CBInfo. Globally-allocated CBs become
    // borrowed-memory DFBs (borrowed_from the backing TensorParameter); the others are
    // Program-local L1 DFBs. CB index assignment from emit_cb_descriptors is irrelevant
    // here — Metal 2.0 derives indices from bindings — but the per-CB entry_size /
    // num_entries / data_format are taken verbatim from CBInfo.
    for (const auto& cb : cb_info) {
        if (cb.num_pages == 0) {
            continue;  // matches emit_cb_descriptors / allocate_cbs skip
        }
        m2::DataflowBufferSpec dfb{
            .unique_id = dfb_name_for(cb.name),
            .entry_size = cb.page_size,
            .num_entries = cb.num_pages,
            .data_format_metadata = cb.data_format};
        if (cb.is_globally_allocated) {
            switch (cb.name) {
                case Conv2dCb::ACT_SHARDED: dfb.borrowed_from = TP_ACT_SHARDED; break;
                case Conv2dCb::OUT:
                case Conv2dCb::MATMUL_PARTIALS: dfb.borrowed_from = TP_OUT; break;
                case Conv2dCb::READER_INDICES: dfb.borrowed_from = TP_READER_INDICES; break;
                default:
                    TT_THROW("Unexpected globally-allocated CB {} in width-sharded spec", static_cast<int>(cb.name));
            }
        }
        spec.dataflow_buffers.push_back(std::move(dfb));
    }

    const bool has_act_second_reader = get_cb_info_by_name(cb_info, Conv2dCb::ACT_SECOND_READER).num_pages != 0;

    // ---- Semaphores (program-local mcast) ----
    spec.semaphores = {
        m2::SemaphoreSpec{.unique_id = SEM_SENDER, .target_nodes = all_cores},
        m2::SemaphoreSpec{.unique_id = SEM_RECEIVER, .target_nodes = all_cores},
    };

    // ---- Activation reader kernel (RISCV_0) ----
    // Named CTAs replace the positional CT args; semaphore-id CTAs become sem::
    // bindings; CB-id CTAs become dfb:: bindings. The two per-core mcast lookup
    // tables (variable length) move to runtime varargs after the three named RTAs.
    m2::KernelSpec act_kernel{
        .unique_id = K_ACT,
        .source =
            std::filesystem::path{
                "ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/activation_reader_width_sharded.cpp"},
        .compile_time_args =
            {{"stride_w", stride_w},
             {"dilation_h", dilation_h},
             {"dilation_w", dilation_w},
             {"conv_act_size_w", conv_act_size_w},
             {"conv_act_c_read_bytes", conv_act_c_read_bytes},
             {"weight_size_h", filter_h},
             {"weight_size_w", filter_w},
             {"act_block_h_datums", act_block_h_datums},
             {"act_block_num_tiles", act_block_num_tiles},
             {"num_input_cores", (uint32_t)input_num_cores},
             {"act_num_blocks_h", num_blocks_act_h_per_core},
             {"act_num_blocks_w", per_core_num_blocks_act_w},
             {"mcast_noc_x_start", (uint32_t)act_mcast_start.x},
             {"mcast_noc_y_start", (uint32_t)act_mcast_start.y},
             {"mcast_noc_x_end", (uint32_t)act_mcast_end.x},
             {"mcast_noc_y_end", (uint32_t)act_mcast_end.y},
             {"act_mcast_sender_size_bytes", (uint32_t)(act_block_num_tiles * tt::tile_size(tilized_act_df))},
             {"num_output_cores", (uint32_t)output_num_cores},
             {"num_reader_cores", (uint32_t)all_reader_cores.size()}},
        .runtime_arg_schema = {.runtime_arg_names = {"this_core_x", "this_core_y", "num_cores_x"}},
        .hw_config = m2::DataMovementHardwareConfig{.role = m2::DataMovementRoleHint::READER},
    };
    // mcast lookup tables: 2 * full_core_grid.x|.y entries appended as varargs.
    auto full_core_grid = device->compute_with_storage_grid_size();
    act_kernel.advanced_options.num_runtime_varargs = full_core_grid.x + full_core_grid.y;
    {
        m2::KernelSpec::CompilerOptions::Defines d;
        for (const auto& [k, v] : reader_defines) {
            d.insert({k, v});
        }
        act_kernel.compiler_options.defines = std::move(d);
    }
    // Act reader DFB bindings: produces ACT_ROW_MAJOR (from L1 reads), participates
    // in the ACT/ACT_TILIZED mcast loop, and consumes READER_INDICES / ACT_SHARDED /
    // L1_ARRAY as borrowed/scratch address sources (self-loop — no real FIFO peer).
    act_kernel.dfb_bindings = {
        m2::DFBBinding{
            .dfb_spec_name = DFB_ACT_ROW_MAJOR,
            .accessor_name = "act_row_major",
            .endpoint_type = m2::DFBEndpointType::PRODUCER},
        m2::DFBBinding{
            .dfb_spec_name = DFB_ACT, .accessor_name = "act", .endpoint_type = m2::DFBEndpointType::PRODUCER},
        m2::DFBBinding{
            .dfb_spec_name = DFB_ACT_TILIZED,
            .accessor_name = "act_tilized",
            .endpoint_type = m2::DFBEndpointType::CONSUMER},
        // ACT_SHARDED / READER_INDICES / L1_ARRAY: base-pointer address sources only
        // (no FIFO peer) -> self-loop to satisfy the producer+consumer invariant.
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
        m2::DFBBinding{
            .dfb_spec_name = DFB_L1_ARRAY, .accessor_name = "l1_array", .endpoint_type = m2::DFBEndpointType::PRODUCER},
        m2::DFBBinding{
            .dfb_spec_name = DFB_L1_ARRAY, .accessor_name = "l1_array", .endpoint_type = m2::DFBEndpointType::CONSUMER},
    };
    act_kernel.semaphore_bindings = {
        m2::SemaphoreBinding{.semaphore_spec_name = SEM_SENDER, .accessor_name = "act_mcast_sender"},
        m2::SemaphoreBinding{.semaphore_spec_name = SEM_RECEIVER, .accessor_name = "act_mcast_receiver"},
    };
    act_kernel.tensor_bindings = {
        m2::TensorBinding{.tensor_parameter_name = TP_ACT_SHARDED, .accessor_name = "act_sharded"},
        m2::TensorBinding{.tensor_parameter_name = TP_READER_INDICES, .accessor_name = "reader_indices"},
    };

    // ---- Weights/bias reader kernel (RISCV_1) ----
    m2::KernelSpec weights_kernel{
        .unique_id = K_WEIGHTS,
        .source =
            std::filesystem::path{
                "ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/weights_reader_width_sharded.cpp"},
        .compile_time_args =
            {{"core_in_channels_ntiles", act_block_w_ntiles / (filter_h * filter_w)},
             {"window_size_hw", filter_h * filter_w},
             {"weight_block_width_ntiles", weight_block_w_ntiles},
             {"weight_block_num_tiles", weight_block_num_tiles},
             {"weight_matrix_width_ntiles", weight_matrix_width_ntiles},
             {"weight_next_channel_stride_h", (weight_matrix_width_ntiles * input_channels_padded) / 32},
             {"weight_next_block_this_core_stride_h", weight_matrix_width_ntiles * weight_block_in_channels_ntiles},
             {"weight_next_block_other_core_stride_h",
              weight_matrix_width_ntiles * weight_block_in_channels_ntiles * per_core_num_blocks_act_w},
             {"remote_weight_height_blocks", (uint32_t)input_num_cores},
             {"local_weight_height_blocks", per_core_num_blocks_act_w},
             {"act_num_blocks_h", num_blocks_act_h_per_core},
             {"fuse_bias", (uint32_t)has_bias}},
        .runtime_arg_schema = {.runtime_arg_names = {"init_weight_start_tile_id", "is_active"}},
        .hw_config = m2::DataMovementHardwareConfig{.role = m2::DataMovementRoleHint::WRITER},
    };
    weights_kernel.dfb_bindings = {
        m2::DFBBinding{
            .dfb_spec_name = DFB_WEIGHTS, .accessor_name = "weights", .endpoint_type = m2::DFBEndpointType::PRODUCER},
    };
    weights_kernel.tensor_bindings = {
        m2::TensorBinding{.tensor_parameter_name = TP_WEIGHTS, .accessor_name = "weights"},
    };
    {
        m2::KernelSpec::CompilerOptions::Defines d;
        for (const auto& [k, v] : writer_defines) {
            d.insert({k, v});
        }
        if (has_bias) {
            d.insert({"FUSE_BIAS", "1"});
            weights_kernel.dfb_bindings.push_back(m2::DFBBinding{
                .dfb_spec_name = DFB_BIAS, .accessor_name = "bias", .endpoint_type = m2::DFBEndpointType::PRODUCER});
            weights_kernel.tensor_bindings.push_back(
                m2::TensorBinding{.tensor_parameter_name = TP_BIAS, .accessor_name = "bias"});
        }
        weights_kernel.compiler_options.defines = std::move(d);
    }

    // ---- Compute kernel (forked: conv_bmm_tilize_m2.cpp) ----
    m2::KernelSpec compute_kernel{
        .unique_id = K_COMPUTE,
        .source = std::filesystem::path{"ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/conv_bmm_tilize_m2.cpp"},
        .compile_time_args =
            {{"in0_block_w", act_block_w_ntiles},
             {"in0_num_subblocks", act_num_subblocks},
             {"in0_block_num_tiles", act_block_num_tiles},
             {"in0_subblock_num_tiles", act_subblock_num_tiles},
             {"reader_num_h_subblocks", act_subblock_h_ntiles * act_num_subblocks},
             {"in1_num_subblocks", weight_num_subblocks},
             {"in1_block_num_tiles", weight_block_num_tiles},
             {"in1_block_w", weight_block_w_ntiles},
             {"in0_num_blocks_h", num_blocks_act_h_per_core},
             {"in0_num_blocks_w", num_blocks_act_w},
             {"in1_num_blocks_w", num_blocks_weight_w_per_core},
             {"out_subblock_h", out_subblock_h_ntiles},
             {"out_subblock_w", out_subblock_w_ntiles},
             {"out_subblock_num_tiles", out_subblock_num_tiles},
             {"height_sharded", (uint32_t)tilize_in0},
             {"untilize_out", (uint32_t)untilize_out},
             {"bias_ntiles_w", bias_ntiles},
             {"partials_cb_uses_output", (uint32_t)partials_cb_uses_output},
             {"in0_nblocks_w_tilize", (uint32_t)input_num_cores},
             {"check_skip_compute", 0u},
             {"pack_relu", (uint32_t)pack_relu},
             {"packer_untilize", (uint32_t)(weight_block_w_ntiles <= 8)},
             {"packer_l1_acc", (uint32_t)packer_l1_acc},
             {"fuse_bias", (uint32_t)has_bias},
             {"split_reader", 0u},
             {"activation_reuse", 0u},
             {"image_width_in_tiles", 0u},
             {"window_reuse_offset", 0u},
             {"tilized_cb_row_offset", 0u},
             {"tilized_cb_second_reader_offset", 0u},
             {"split_reader_cb_shared", 0u}},
        .hw_config =
            m2::ComputeHardwareConfig{
                .math_fidelity = math_fidelity,
                .fp32_dest_acc_en = fp32_dest_acc_en,
            },
    };
    compute_kernel.dfb_bindings = {
        m2::DFBBinding{
            .dfb_spec_name = DFB_ACT, .accessor_name = "act", .endpoint_type = m2::DFBEndpointType::CONSUMER},
        m2::DFBBinding{
            .dfb_spec_name = DFB_ACT_ROW_MAJOR,
            .accessor_name = "act_row_major",
            .endpoint_type = m2::DFBEndpointType::CONSUMER},
        m2::DFBBinding{
            .dfb_spec_name = DFB_ACT_TILIZED,
            .accessor_name = "act_tilized",
            .endpoint_type = m2::DFBEndpointType::PRODUCER},
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
        // directly into it; there is no separate writer/FIFO consumer, so self-loop
        // it on compute to satisfy the producer+consumer invariant.
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
            // Width-sharded never sets this, but keep the gate honest if get_cb_info ever changes.
            d.insert({"SECOND_READER_PRESENT", "1"});
            compute_kernel.dfb_bindings.push_back(m2::DFBBinding{
                .dfb_spec_name = dfb_name_for(Conv2dCb::ACT_SECOND_READER),
                .accessor_name = "act_second_reader",
                .endpoint_type = m2::DFBEndpointType::CONSUMER});
        }
        compute_kernel.compiler_options.defines = std::move(d);
    }

    spec.kernels = {act_kernel, weights_kernel, compute_kernel};

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

    // ---- WorkUnit ----
    // All three kernels share local DFBs (ACT/ACT_ROW_MAJOR/ACT_TILIZED/WEIGHTS/
    // MATMUL_PARTIALS/OUT/BIAS), so they must share a single WorkUnitSpec covering
    // every node that hosts any of those DFBs. Legacy placed the act reader on the
    // bounding box of all_cores and the weights/compute kernels on all_cores; for
    // the shared-DFB invariant the Metal 2.0 spec places all three on one node set
    // (the bounding box of all_cores). Inactive (noop) cores still host the kernels,
    // which early-return via this_core_id / is_active. Both readers get runtime args
    // for every node in this set (weights with is_active = 0 on inactive cores).
    const CoreRangeSet work_nodes{all_reader_cores};
    spec.work_units = std::vector<m2::WorkUnitSpec>{
        m2::WorkUnitSpec{.name = "conv2d_ws", .kernels = {K_ACT, K_WEIGHTS, K_COMPUTE}, .target_nodes = work_nodes},
    };

    // ---- ProgramRunArgs (mutable) ----
    m2::ProgramRunArgs run;
    m2::KernelRunArgs act_run{.kernel = K_ACT};
    m2::KernelRunArgs weights_run{.kernel = K_WEIGHTS};
    m2::KernelRunArgs compute_run{.kernel = K_COMPUTE};

    std::vector<uint32_t> act_mcast_noc_x;
    std::vector<uint32_t> act_mcast_noc_y;
    act_mcast_noc_x.reserve(full_core_grid.x);
    for (uint32_t core_index = 0; core_index < full_core_grid.x; core_index++) {
        act_mcast_noc_x.push_back(device->worker_core_from_logical_core(CoreCoord(core_index, 0)).x);
    }
    act_mcast_noc_y.reserve(full_core_grid.y);
    for (uint32_t core_index = 0; core_index < full_core_grid.y; core_index++) {
        act_mcast_noc_y.push_back(device->worker_core_from_logical_core(CoreCoord(0, core_index)).y);
    }

    auto total_num_active_cores = std::max(input_num_cores, output_num_cores);
    auto total_num_cores = all_reader_cores.size();
    for (uint32_t core_index = 0; core_index < total_num_cores; core_index++) {
        uint32_t core_x = core_index % full_core_grid.x;
        uint32_t core_y = core_index / full_core_grid.x;
        CoreCoord core(core_x, core_y);

        act_run.runtime_arg_values.push_back(
            {core, {{"this_core_x", core_x}, {"this_core_y", core_y}, {"num_cores_x", (uint32_t)full_core_grid.x}}});
        // Mcast X/Y lookup tables as runtime varargs (X table then Y table).
        m2::AdvancedKernelRunArgs::Varargs varargs;
        varargs.reserve(act_mcast_noc_x.size() + act_mcast_noc_y.size());
        varargs.insert(varargs.end(), act_mcast_noc_x.begin(), act_mcast_noc_x.end());
        varargs.insert(varargs.end(), act_mcast_noc_y.begin(), act_mcast_noc_y.end());
        act_run.advanced_options.runtime_varargs.insert({core, std::move(varargs)});

        // Weights kernel now shares the WorkUnit node set, so every node needs
        // weights RTAs. Active cores (core_index < total_num_active_cores) carry the
        // real per-core start tile id; inactive (noop) cores get a benign start id of
        // 0 and is_active = 0 (the kernel's `if (is_active)` guard skips all reads).
        const bool core_is_active = core_index < total_num_active_cores;
        weights_run.runtime_arg_values.push_back(
            {core,
             {{"init_weight_start_tile_id",
               core_is_active ? static_cast<uint32_t>(core_index * weight_block_w_ntiles) : 0u},
              {"is_active", static_cast<uint32_t>(core_index < output_num_cores)}}});
    }
    (void)compute_run;  // compute kernel has no per-core RTAs

    run.kernel_run_args = {act_run, weights_run, compute_run};
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
