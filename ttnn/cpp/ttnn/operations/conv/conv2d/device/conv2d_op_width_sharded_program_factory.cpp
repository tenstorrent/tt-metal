// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <utility>

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
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/kernel_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/dataflow_buffer_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/semaphore_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/tensor_parameter.hpp>
#include <tt-metalium/experimental/metal2_host_api/node_coord.hpp>
#include <tt-metalium/mesh_command_queue.hpp>
#include <tt-metalium/tensor/mesh_tensor.hpp>
#include "ttnn/operations/compute_throttle_utils.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

namespace ttnn::prim {

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

// ---- Metal 2.0 resource names (ProgramSpec scope) ----
// DFB accessor names surface kernel-side as dfb::<name> tokens; the ported width-sharded kernels
// reference these exact spellings.
const tt::tt_metal::experimental::DFBSpecName conv2d_width_act_dfb{"act"};  // mcast result (act reader -> compute)
const tt::tt_metal::experimental::DFBSpecName conv2d_width_act_row_major_dfb{
    "act_row_major"};  // act reader -> compute (tilize input)
const tt::tt_metal::experimental::DFBSpecName conv2d_width_act_tilized_dfb{
    "act_tilized"};  // compute (tilize) -> act reader (mcast src)
const tt::tt_metal::experimental::DFBSpecName conv2d_width_weights_dfb{"weights"};  // weights reader -> compute
const tt::tt_metal::experimental::DFBSpecName conv2d_width_bias_dfb{"bias"};  // weights reader -> compute (optional)
const tt::tt_metal::experimental::DFBSpecName conv2d_width_matmul_partials_dfb{
    "matmul_partials"};  // compute self-loop (borrows OUTPUT when aliased)
const tt::tt_metal::experimental::DFBSpecName conv2d_width_out_dfb{
    "out"};  // compute packer -> OUTPUT (degenerate consumer)
const tt::tt_metal::experimental::DFBSpecName conv2d_width_reader_indices_dfb{
    "reader_indices"};  // borrowed indices (act reader address source)

const tt::tt_metal::experimental::TensorParamName conv2d_width_input_tensor{"input"};
const tt::tt_metal::experimental::TensorParamName conv2d_width_output_tensor{"output"};
const tt::tt_metal::experimental::TensorParamName conv2d_width_weights_tensor{"weights"};
const tt::tt_metal::experimental::TensorParamName conv2d_width_bias_tensor{"bias"};
const tt::tt_metal::experimental::TensorParamName conv2d_width_reader_indices_tensor{"reader_indices"};

const tt::tt_metal::experimental::SemaphoreSpecName conv2d_width_act_mcast_sender_semaphore{"act_mcast_sender"};
const tt::tt_metal::experimental::SemaphoreSpecName conv2d_width_act_mcast_receiver_semaphore{"act_mcast_receiver"};

const tt::tt_metal::experimental::KernelSpecName conv2d_width_act_kernel{"act_reader"};
const tt::tt_metal::experimental::KernelSpecName conv2d_width_weights_kernel{"weights_reader"};
const tt::tt_metal::experimental::KernelSpecName conv2d_width_compute_kernel{"compute"};

}  // namespace

ttnn::device_operation::ProgramArtifacts Conv2dWidthShardedProgramFactory::create_program_artifacts(
    const Conv2dParams& operation_attributes, const Conv2dInputs& tensor_args, Tensor& output_tensor) {
    const auto& a = tensor_args.a;
    const auto& b = tensor_args.b;
    const auto& bias = tensor_args.bias;
    const auto& ashape = ttnn::Shape(operation_attributes.input_tensor_shape);
    const auto& sliding_window_config = operation_attributes.sliding_window_config;

    ttnn::operations::sliding_window::ParallelConfig parallel_config{
        .grid = a.shard_spec().value().grid,
        .shard_scheme = a.memory_config().memory_layout(),
        .shard_orientation = a.shard_spec().value().orientation};

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

    auto packer_l1_acc = compute_kernel_config.packer_l1_acc;

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

    uint32_t input_size_w = conv_act_size_w + pad_w;
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
        a.storage_type() == ttnn::StorageType::DEVICE && b.storage_type() == ttnn::StorageType::DEVICE,
        "Operands to large matmul need to be on device!");
    TT_FATAL(a.device() == b.device(), "Operands to conv need to be on the same device!");
    TT_FATAL(
        a.buffer() != nullptr && b.buffer() != nullptr, "Operands to conv need to be allocated in buffers on device!");
    if (has_bias) {
        TT_FATAL(bias.value().storage_type() == ttnn::StorageType::DEVICE, "Bias should be on device");
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

    uint32_t conv_act_c_read_bytes = conv_act_size_c * a.element_size() / (input_num_cores * per_core_num_blocks_act_w);

    bool tilize_in0 = false;

    // Select preferred NoCs for DRAM operations based on architecture.
    tt::tt_metal::NOC weights_noc = tt::tt_metal::detail::preferred_noc_for_dram_read(device->arch());
    tt::tt_metal::NOC act_noc = tt::tt_metal::detail::preferred_noc_for_dram_write(device->arch());

    CoreCoord act_mcast_start_core_logical(0, 0);
    CoreCoord act_mcast_end_core_logical(all_cores.bounding_box().end_coord.x, all_cores.bounding_box().end_coord.y);
    auto act_mcast_start = device->worker_core_from_logical_core(act_mcast_start_core_logical);
    auto act_mcast_end = device->worker_core_from_logical_core(act_mcast_end_core_logical);

    // Swap multicast coordinates if using NOC_1 for proper addressing
    if (act_noc == tt::tt_metal::NOC::NOC_1) {
        std::swap(act_mcast_start, act_mcast_end);
    }

    TT_FATAL(act_block_h_datums % 2 == 0, "2 Indices are packed in one uint32_t word.");

    const ttnn::operations::conv::SkipMcast skip_mcast =
        ttnn::operations::conv::conv_skip_mcast(parallelization_config, a.memory_config().memory_layout());
    const bool skip_activation_mcast = skip_mcast.skip_activation_mcast;

    bool pack_relu =
        fused_activation.has_value() && fused_activation.value().op_type == ttnn::operations::unary::UnaryOpType::RELU;
    std::map<std::string, std::string> compute_defines;
    if (fused_activation.has_value() && !pack_relu) {
        // Pass the output dtype explicitly so no-parameter unary ops can generate their typed defines.
        compute_defines.merge(ttnn::operations::unary::utils::get_defines(
            fused_activation.value().op_type, fused_activation.value().params, "ACTIVATION", "i", output.dtype()));
    }
    ttnn::operations::compute_throttle_utils::throttle_mm_perf(
        device->arch(), output_cores.num_cores(), compute_defines, ttnn::get_throttle_level(compute_kernel_config));

    const uint32_t output_image_width = sliding_window_config.get_output_shape()[2];
    Conv2dConfig conv_config = Conv2dConfig{
        .weights_dtype = b.dtype(),
        .config_tensors_in_dram = config_tensors_in_dram,
        .shard_layout = a.memory_config().memory_layout(),
        .output_layout = (untilize_out ? Layout::ROW_MAJOR : Layout::TILE),
        .enable_act_double_buffer = enable_act_double_buffer,
        .enable_weights_double_buffer = enable_weights_double_buffer};

    // ---- Allocate the op-owned conv_reader_indices tensor ----
    // This intermediate config tensor must outlive the cached program; move it with the established
    // sliding-window helper so multi-device distribution and the selected DRAM/L1-small layout remain
    // identical to the legacy factory, then transfer its existing MeshTensor into ProgramArtifacts.
    std::vector<uint32_t> op_trace_metadata =
        ttnn::operations::sliding_window::generate_op_trace_metadata(sliding_window_config);
    std::vector<sliding_window::ShardBoundary> shard_boundaries =
        ttnn::operations::sliding_window::generate_shard_boundaries(sliding_window_config);
    std::vector<std::vector<uint16_t>> conv_sharded_input_top_left_indices =
        ttnn::operations::sliding_window::generate_sliding_window_op_config(
            op_trace_metadata, shard_boundaries, stride_w, true, act_block_h_datums, 0);
    Tensor host_config_tensor = ttnn::operations::sliding_window::construct_on_host_config_tensor(
        conv_sharded_input_top_left_indices, parallel_config, config_tensors_in_dram);
    Tensor conv_reader_indices_tensor = ttnn::operations::sliding_window::move_config_tensor_to_device(
        host_config_tensor, parallel_config, false, a.device(), config_tensors_in_dram);
    tt::tt_metal::Buffer* conv_reader_indices_buffer = conv_reader_indices_tensor.buffer();
    const uint32_t reader_indices_actual_page_size = conv_reader_indices_buffer->page_size();
    tt::tt_metal::MeshTensor reader_indices_mesh_tensor =
        conv_reader_indices_tensor.device_storage().release_mesh_tensor();

    // ---- Query CB sizing/format/backing via the shared conv2d helper ----
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
    const bool overlap_act_cb = get_cb_info_by_name(cb_info, Conv2dCb::ACT).overlapped_by_cb.has_value();
    const auto& act_storage_dfb = overlap_act_cb ? conv2d_width_act_tilized_dfb : conv2d_width_act_dfb;

    // Convenience accessor for CB sizing.
    auto cb = [&](Conv2dCb name) -> const CBInfo& { return get_cb_info_by_name(cb_info, name); };

    // ============================================================================
    //  Build the ProgramSpec
    // ============================================================================
    tt::tt_metal::experimental::ProgramSpec spec;
    spec.name = "conv2d_width_sharded";

    // ---- Tensor parameters ----
    spec.tensor_parameters.push_back(tt::tt_metal::experimental::TensorParameter{
        .unique_id = conv2d_width_input_tensor, .spec = a.mesh_tensor().tensor_spec()});
    spec.tensor_parameters.push_back(tt::tt_metal::experimental::TensorParameter{
        .unique_id = conv2d_width_output_tensor, .spec = output.mesh_tensor().tensor_spec()});
    spec.tensor_parameters.push_back(tt::tt_metal::experimental::TensorParameter{
        .unique_id = conv2d_width_weights_tensor, .spec = b.mesh_tensor().tensor_spec()});
    if (has_bias) {
        spec.tensor_parameters.push_back(tt::tt_metal::experimental::TensorParameter{
            .unique_id = conv2d_width_bias_tensor, .spec = bias.value().mesh_tensor().tensor_spec()});
    }
    spec.tensor_parameters.push_back(tt::tt_metal::experimental::TensorParameter{
        .unique_id = conv2d_width_reader_indices_tensor, .spec = reader_indices_mesh_tensor.tensor_spec()});

    // ---- Semaphores (act mcast sender/receiver) ----
    spec.semaphores.push_back(tt::tt_metal::experimental::SemaphoreSpec{
        .unique_id = conv2d_width_act_mcast_sender_semaphore, .target_nodes = all_reader_cores});
    spec.semaphores.push_back(tt::tt_metal::experimental::SemaphoreSpec{
        .unique_id = conv2d_width_act_mcast_receiver_semaphore, .target_nodes = all_reader_cores});

    // ---- Dataflow buffers ----
    // Sizes/formats/backing come straight from get_cb_info() (entry_size = page_size,
    // num_entries = num_pages).  Borrowed DFBs alias their backing tensor (ACT_SHARDED->INPUT,
    // OUT/MATMUL_PARTIALS->OUTPUT, READER_INDICES->indices tensor when L1-resident).
    auto make_dfb = [&](const tt::tt_metal::experimental::DFBSpecName& id, Conv2dCb name) {
        const CBInfo& info = cb(name);
        TT_FATAL(info.num_pages > 0, "Conv2D DFB '{}' must have at least one page", id);
        return tt::tt_metal::experimental::DataflowBufferSpec{
            .unique_id = id,
            .entry_size = info.page_size,
            .num_entries = info.num_pages,
            .data_format_metadata = info.data_format,
        };
    };

    // ACT (mcast result): real FIFO act reader -> compute.  The legacy skip-mcast path mirrors this
    // CB index onto ACT_TILIZED; in that case both accessor names bind the one existing DFB below.
    if (!overlap_act_cb) {
        auto dfb = make_dfb(conv2d_width_act_dfb, Conv2dCb::ACT);
        spec.dataflow_buffers.push_back(std::move(dfb));
    }
    // ACT_ROW_MAJOR_BFLOAT16: act reader -> compute (tilize input).
    {
        auto dfb = make_dfb(conv2d_width_act_row_major_dfb, Conv2dCb::ACT_ROW_MAJOR_BFLOAT16);
        spec.dataflow_buffers.push_back(std::move(dfb));
    }
    // ACT_TILIZED: compute (tilize) -> act reader (mcast source).
    {
        auto dfb = make_dfb(conv2d_width_act_tilized_dfb, Conv2dCb::ACT_TILIZED);
        dfb.advanced_options.allow_instance_multi_binding = overlap_act_cb;
        spec.dataflow_buffers.push_back(std::move(dfb));
    }
    // WEIGHTS: weights reader -> compute.
    spec.dataflow_buffers.push_back(make_dfb(conv2d_width_weights_dfb, Conv2dCb::WEIGHTS));
    // BIAS: weights reader -> compute (optional).
    if (has_bias) {
        spec.dataflow_buffers.push_back(make_dfb(conv2d_width_bias_dfb, Conv2dCb::BIAS));
    }

    // MATMUL_PARTIALS: compute self-loop accumulator.  Borrowed-from OUTPUT when
    // partials_cb_uses_output (in-place accumulate into the output buffer); the self-loop on the
    // single compute kernel keeps it SPSC-clean: compute explicitly owns both endpoint roles
    // because there is no independent producer or consumer kernel for this accumulator.
    {
        auto dfb = make_dfb(conv2d_width_matmul_partials_dfb, Conv2dCb::MATMUL_PARTIALS);
        if (partials_cb_uses_output) {
            dfb.borrowed_from = conv2d_width_output_tensor;
            dfb.borrowed_memory_offset = cb(Conv2dCb::MATMUL_PARTIALS).address_offset;
        }
        spec.dataflow_buffers.push_back(std::move(dfb));
    }

    // OUT: compute packer -> OUTPUT shard (borrowed).  Producer-only fake CB; bind a degenerate
    // consumer on the compute kernel itself (width-sharded has no DM output kernel) via a self-loop
    // to satisfy the spec completeness check; the final result remains resident in the borrowed
    // output shard and no additional data movement is required.
    {
        auto dfb = make_dfb(conv2d_width_out_dfb, Conv2dCb::OUT);
        dfb.borrowed_from = conv2d_width_output_tensor;
        spec.dataflow_buffers.push_back(std::move(dfb));
    }


    // READER_INDICES: borrowed indices tensor (L1-resident path) or fresh L1 (DRAM-config path, where
    // the reader fills it from DRAM via TensorAccessor).  Address-source/fake-fill; self-loop on the
    // act reader.
    {
        auto dfb = make_dfb(conv2d_width_reader_indices_dfb, Conv2dCb::READER_INDICES);
        if (cb(Conv2dCb::READER_INDICES).is_globally_allocated) {
            dfb.borrowed_from = conv2d_width_reader_indices_tensor;
        }
        spec.dataflow_buffers.push_back(std::move(dfb));
    }

    uint32_t emitted_l1_dfb_size = 0;
    for (const auto& dfb : spec.dataflow_buffers) {
        if (!dfb.borrowed_from.has_value()) {
            emitted_l1_dfb_size += dfb.entry_size * dfb.num_entries;
        }
    }
    const auto predicted_l1_usage =
        predicted_conv2d_l1_usage(operation_attributes, tensor_args, reader_indices_actual_page_size);
    TT_FATAL(
        emitted_l1_dfb_size == predicted_l1_usage.CB_allocation_size,
        "Predicted Conv2D L1 DFB size {} does not match emitted size {}",
        predicted_l1_usage.CB_allocation_size,
        emitted_l1_dfb_size);

    // ---- Compute kernel ----
    // Self-loop bindings: MATMUL_PARTIALS (real accumulator) and OUT (degenerate consumer).
    std::vector<tt::tt_metal::experimental::DFBBinding> compute_dfb_bindings = {
        tt::tt_metal::experimental::DFBBinding{
            .dfb_spec_name = act_storage_dfb,
            .accessor_name = "act",
            .endpoint_type = tt::tt_metal::experimental::DFBEndpointType::CONSUMER},
        tt::tt_metal::experimental::DFBBinding{
            .dfb_spec_name = conv2d_width_act_row_major_dfb,
            .accessor_name = "act_row_major",
            .endpoint_type = tt::tt_metal::experimental::DFBEndpointType::CONSUMER,
            .accessor_aliases = {"act_second_reader"}},
        tt::tt_metal::experimental::DFBBinding{
            .dfb_spec_name = conv2d_width_act_tilized_dfb,
            .accessor_name = "act_tilized",
            .endpoint_type = tt::tt_metal::experimental::DFBEndpointType::PRODUCER},
        tt::tt_metal::experimental::DFBBinding{
            .dfb_spec_name = conv2d_width_weights_dfb,
            .accessor_name = "weights",
            .endpoint_type = tt::tt_metal::experimental::DFBEndpointType::CONSUMER},
        tt::tt_metal::experimental::DFBBinding{
            .dfb_spec_name = conv2d_width_matmul_partials_dfb,
            .accessor_name = "matmul_partials",
            .endpoint_type = tt::tt_metal::experimental::DFBEndpointType::PRODUCER},
        tt::tt_metal::experimental::DFBBinding{
            .dfb_spec_name = conv2d_width_matmul_partials_dfb,
            .accessor_name = "matmul_partials",
            .endpoint_type = tt::tt_metal::experimental::DFBEndpointType::CONSUMER},
        tt::tt_metal::experimental::DFBBinding{
            .dfb_spec_name = conv2d_width_out_dfb,
            .accessor_name = "out",
            .endpoint_type = tt::tt_metal::experimental::DFBEndpointType::PRODUCER,
            .accessor_aliases = has_bias ? std::vector<std::string>{} : std::vector<std::string>{"bias"}},
        tt::tt_metal::experimental::DFBBinding{
            .dfb_spec_name = conv2d_width_out_dfb,
            .accessor_name = "out",
            .endpoint_type = tt::tt_metal::experimental::DFBEndpointType::CONSUMER},
    };
    if (has_bias) {
        compute_dfb_bindings.push_back(tt::tt_metal::experimental::DFBBinding{
            .dfb_spec_name = conv2d_width_bias_dfb,
            .accessor_name = "bias",
            .endpoint_type = tt::tt_metal::experimental::DFBEndpointType::CONSUMER});
    }

    // Preserve the legacy ProgramDescriptor lowering: only fidelity and FP32 destination mode were
    // forwarded, while approximation and destination synchronization retained hardware defaults.
    auto compute_hw_config = make_legacy_conv2d_compute_hardware_config(compute_kernel_config);
    if (compute_kernel_config.fp32_dest_acc_en) {
        auto& unpack_modes = tt::tt_metal::experimental::unpack_modes(compute_hw_config);
        auto select_source_unpack = [&](const tt::tt_metal::experimental::DFBSpecName& dfb_name) {
            const auto dfb = std::ranges::find(
                spec.dataflow_buffers, dfb_name, &decltype(spec.dataflow_buffers)::value_type::unique_id);
            TT_FATAL(
                dfb != spec.dataflow_buffers.end(),
                "Missing Conv2D DFB '{}' while selecting its unpack mode",
                dfb_name);
            if (dfb->data_format_metadata == tt::DataFormat::Float32) {
                unpack_modes.emplace(dfb_name, tt::tt_metal::UnpackMode::UnpackToSrc);
            }
        };
        select_source_unpack(act_storage_dfb);
        select_source_unpack(conv2d_width_act_row_major_dfb);
        select_source_unpack(conv2d_width_act_tilized_dfb);
        select_source_unpack(conv2d_width_weights_dfb);
        select_source_unpack(conv2d_width_matmul_partials_dfb);
        select_source_unpack(conv2d_width_out_dfb);
        if (has_bias) {
            select_source_unpack(conv2d_width_bias_dfb);
        }
    }

    tt::tt_metal::experimental::KernelSpec compute_kernel{
        .unique_id = conv2d_width_compute_kernel,
        .source = std::filesystem::path("ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/conv_bmm_tilize.cpp"),
        .compiler_options =
            {.defines = tt::tt_metal::experimental::KernelSpec::CompilerOptions::Defines(compute_defines),
             .opt_level = tt::tt_metal::KernelBuildOptLevel::O3},
        .dfb_bindings = std::move(compute_dfb_bindings),
        .compile_time_args =
            {
                {"in0_block_w", act_block_w_ntiles},
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
                {"in0_nblocks_w_tilize", input_num_cores},
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
                {"split_reader_cb_shared", 0u},
            },
        .runtime_arg_schema = {.runtime_arg_names = {"skip_compute"}},
        .hw_config = std::move(compute_hw_config),
    };

    // ---- Activation reader kernel ----
    // DFB bindings: produces ACT_ROW_MAJOR + ACT (mcast), consumes ACT_TILIZED (mcast source);
    // self-loops the borrowed ACT_SHARDED (input address source) and READER_INDICES.
    auto act_hw = tt::tt_metal::experimental::DataMovementGen1Config{
        .processor = tt::tt_metal::DataMovementProcessor::RISCV_0, .noc = act_noc};
    tt::tt_metal::experimental::KernelSpec act_kernel{
        .unique_id = conv2d_width_act_kernel,
        .source = std::filesystem::path("ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/"
                                        "activation_reader_width_sharded.cpp"),
        .dfb_bindings =
            {
                tt::tt_metal::experimental::DFBBinding{
                    .dfb_spec_name = conv2d_width_act_row_major_dfb,
                    .accessor_name = "act_row_major",
                    .endpoint_type = tt::tt_metal::experimental::DFBEndpointType::PRODUCER},
                tt::tt_metal::experimental::DFBBinding{
                    .dfb_spec_name = act_storage_dfb,
                    .accessor_name = "act",
                    .endpoint_type = tt::tt_metal::experimental::DFBEndpointType::PRODUCER},
                tt::tt_metal::experimental::DFBBinding{
                    .dfb_spec_name = conv2d_width_act_tilized_dfb,
                    .accessor_name = "act_tilized",
                    .endpoint_type = tt::tt_metal::experimental::DFBEndpointType::CONSUMER},
                tt::tt_metal::experimental::DFBBinding{
                    .dfb_spec_name = conv2d_width_reader_indices_dfb,
                    .accessor_name = "reader_indices",
                    .endpoint_type = tt::tt_metal::experimental::DFBEndpointType::PRODUCER},
                tt::tt_metal::experimental::DFBBinding{
                    .dfb_spec_name = conv2d_width_reader_indices_dfb,
                    .accessor_name = "reader_indices",
                    .endpoint_type = tt::tt_metal::experimental::DFBEndpointType::CONSUMER},
            },
        .semaphore_bindings =
            {
                tt::tt_metal::experimental::SemaphoreBinding{
                    .semaphore_spec_name = conv2d_width_act_mcast_sender_semaphore,
                    .accessor_name = "act_mcast_sender"},
                tt::tt_metal::experimental::SemaphoreBinding{
                    .semaphore_spec_name = conv2d_width_act_mcast_receiver_semaphore,
                    .accessor_name = "act_mcast_receiver"},
            },
        .compile_time_args =
            {
                {"stride_w", (uint32_t)stride_w},
                {"dilation_h", (uint32_t)dilation_h},
                {"dilation_w", (uint32_t)dilation_w},
                {"input_size_w", (uint32_t)input_size_w},
                {"conv_act_c_read_bytes", (uint32_t)conv_act_c_read_bytes},
                {"weight_size_h", (uint32_t)filter_h},
                {"weight_size_w", (uint32_t)filter_w},
                {"act_block_h_datums", (uint32_t)act_block_h_datums},
                {"act_block_num_tiles", (uint32_t)act_block_num_tiles},
                {"num_input_cores", (uint32_t)input_num_cores},
                {"act_num_blocks_h", (uint32_t)num_blocks_act_h_per_core},
                {"act_num_blocks_w", (uint32_t)per_core_num_blocks_act_w},
                {"act_mcast_start_x", (uint32_t)act_mcast_start.x},
                {"act_mcast_start_y", (uint32_t)act_mcast_start.y},
                {"act_mcast_end_x", (uint32_t)act_mcast_end.x},
                {"act_mcast_end_y", (uint32_t)act_mcast_end.y},
                {"act_mcast_sender_size_bytes", (uint32_t)act_block_num_tiles * tt::tile_size(tilized_act_df)},
                {"num_output_cores", (uint32_t)output_num_cores},
                {"num_reader_cores", (uint32_t)all_reader_cores.size()},
                {"config_page_size", conv_reader_indices_buffer->page_size()},
                {"config_tensor_in_dram", (uint32_t)config_tensors_in_dram},
                {"skip_mcast", (uint32_t)skip_activation_mcast},
            },
        .runtime_arg_schema =
            {
                .runtime_arg_names = {"this_core_x", "this_core_y", "num_cores_x"},
            },
        .hw_config = std::move(act_hw),
    };
    // Both constexpr specializations share one named interface. The resident path emits no NOC read.
    act_kernel.tensor_bindings.push_back(tt::tt_metal::experimental::TensorBinding{
        .tensor_parameter_name = conv2d_width_reader_indices_tensor, .accessor_name = "reader_indices"});
    act_kernel.tensor_bindings.push_back(tt::tt_metal::experimental::TensorBinding{
        .tensor_parameter_name = conv2d_width_input_tensor, .accessor_name = "input"});
    // X/Y mcast NoC lookup tables are passed as per-node runtime varargs (variable count: full grid).
    auto full_core_grid = device->compute_with_storage_grid_size();
    act_kernel.advanced_options.num_runtime_varargs = full_core_grid.x + full_core_grid.y;

    // ---- Weights reader kernel ----
    std::vector<tt::tt_metal::experimental::DFBBinding> weights_dfb_bindings = {
        tt::tt_metal::experimental::DFBBinding{
            .dfb_spec_name = conv2d_width_weights_dfb,
            .accessor_name = "weights",
            .endpoint_type = tt::tt_metal::experimental::DFBEndpointType::PRODUCER},
    };
    std::vector<tt::tt_metal::experimental::TensorBinding> weights_tensor_bindings = {
        tt::tt_metal::experimental::TensorBinding{
            .tensor_parameter_name = conv2d_width_weights_tensor, .accessor_name = "weights"},
    };
    if (has_bias) {
        weights_dfb_bindings.push_back(tt::tt_metal::experimental::DFBBinding{
            .dfb_spec_name = conv2d_width_bias_dfb,
            .accessor_name = "bias",
            .endpoint_type = tt::tt_metal::experimental::DFBEndpointType::PRODUCER});
        weights_tensor_bindings.push_back(tt::tt_metal::experimental::TensorBinding{
            .tensor_parameter_name = conv2d_width_bias_tensor, .accessor_name = "bias"});
    } else {
        weights_dfb_bindings.front().accessor_aliases.push_back("bias");
        weights_tensor_bindings.push_back(tt::tt_metal::experimental::TensorBinding{
            .tensor_parameter_name = conv2d_width_weights_tensor, .accessor_name = "bias"});
    }

    auto weights_hw = tt::tt_metal::experimental::DataMovementGen1Config{
        .processor = tt::tt_metal::DataMovementProcessor::RISCV_1, .noc = weights_noc};
    tt::tt_metal::experimental::KernelSpec weights_kernel{
        .unique_id = conv2d_width_weights_kernel,
        .source = std::filesystem::path("ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/"
                                        "weights_reader_width_sharded.cpp"),
        .dfb_bindings = std::move(weights_dfb_bindings),
        .tensor_bindings = std::move(weights_tensor_bindings),
        .compile_time_args =
            {
                {"core_in_channels_ntiles", act_block_w_ntiles / (filter_h * filter_w)},
                {"window_size_hw", filter_h * filter_w},
                {"weight_block_width_ntiles", weight_block_w_ntiles},
                {"weight_block_num_tiles", weight_block_num_tiles},
                {"weight_matrix_width_ntiles", weight_matrix_width_ntiles},
                {"weight_next_channel_stride_h", (weight_matrix_width_ntiles * input_channels_padded) / 32},
                {"weight_next_block_this_core_stride_h", weight_matrix_width_ntiles * weight_block_in_channels_ntiles},
                {"weight_next_block_other_core_stride_h",
                 weight_matrix_width_ntiles * weight_block_in_channels_ntiles * per_core_num_blocks_act_w},
                {"remote_weight_height_blocks", input_num_cores},
                {"local_weight_height_blocks", per_core_num_blocks_act_w},
                {"act_num_blocks_h", num_blocks_act_h_per_core},
                {"fuse_bias", (uint32_t)has_bias},
            },
        .runtime_arg_schema =
            {
                .runtime_arg_names = {"init_weight_start_tile_id", "is_active"},
            },
        .hw_config = std::move(weights_hw),
    };

    spec.kernels.push_back(std::move(act_kernel));
    spec.kernels.push_back(std::move(weights_kernel));
    spec.kernels.push_back(std::move(compute_kernel));

    if (!overlap_act_cb) {
        add_fictional_dfb_endpoints(spec, conv2d_width_act_dfb);
        add_fictional_dfb_endpoints(spec, conv2d_width_act_tilized_dfb);
    }
    add_fictional_dfb_endpoints(spec, conv2d_width_act_row_major_dfb);

    // ---- Work units ----
    // Preserve the legacy activation-reader bounding rectangle: its multicast targets every node in
    // that rectangle, so padding nodes must retain the same ACT storage even though they return
    // immediately. This restores the legacy footprint; it does not introduce an additional DFB.
    spec.work_units.push_back(tt::tt_metal::experimental::WorkUnitSpec{
        .name = "conv2d_width_sharded",
        .kernels = {conv2d_width_act_kernel, conv2d_width_weights_kernel, conv2d_width_compute_kernel},
        .target_nodes = all_cores,
    });
    const CoreRangeSet padding_reader_cores = CoreRangeSet(all_reader_cores).subtract(all_cores);
    if (!padding_reader_cores.empty()) {
        spec.work_units.push_back(tt::tt_metal::experimental::WorkUnitSpec{
            .name = "conv2d_width_sharded_padding_receivers",
            .kernels = {conv2d_width_act_kernel},
            .target_nodes = padding_reader_cores,
        });
    }

    // ============================================================================
    //  Build the ProgramRunArgs
    // ============================================================================
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

    tt::tt_metal::experimental::ProgramRunArgs run_args;
    tt::tt_metal::experimental::KernelRunArgs act_run_args{.kernel = conv2d_width_act_kernel};
    tt::tt_metal::experimental::KernelRunArgs weights_run_args{.kernel = conv2d_width_weights_kernel};
    tt::tt_metal::experimental::KernelRunArgs compute_run_args{.kernel = conv2d_width_compute_kernel};

    for (uint32_t core_index = 0; core_index < total_num_cores; core_index++) {
        uint32_t core_x = core_index % full_core_grid.x;
        uint32_t core_y = core_index / full_core_grid.x;
        CoreCoord core(core_x, core_y);

        tt::tt_metal::experimental::KernelRunArgs::RuntimeArgValues& act_rtas = act_run_args.runtime_arg_values;
        tt::tt_metal::experimental::AddRuntimeArgsForNode(
            act_rtas,
            core,
            {
                {"this_core_x", core_x},
                {"this_core_y", core_y},
                {"num_cores_x", full_core_grid.x},
            });
        // X/Y mcast lookup tables as per-node varargs.
        tt::tt_metal::experimental::AdvancedKernelRunArgs::Varargs varargs;
        varargs.reserve(act_mcast_noc_x.size() + act_mcast_noc_y.size());
        varargs.insert(varargs.end(), act_mcast_noc_x.begin(), act_mcast_noc_x.end());
        varargs.insert(varargs.end(), act_mcast_noc_y.begin(), act_mcast_noc_y.end());
        act_run_args.advanced_options.runtime_varargs.insert({core, std::move(varargs)});

        if (core_index < total_num_active_cores) {
            tt::tt_metal::experimental::KernelRunArgs::RuntimeArgValues& weights_rtas =
                weights_run_args.runtime_arg_values;
            tt::tt_metal::experimental::AddRuntimeArgsForNode(
                weights_rtas,
                core,
                {
                    {"init_weight_start_tile_id", core_index * weight_block_w_ntiles},
                    {"is_active", static_cast<uint32_t>(core_index < output_num_cores)},
                });
            tt::tt_metal::experimental::AddRuntimeArgsForNode(
                compute_run_args.runtime_arg_values, core, {{"skip_compute", 0u}});
        }
    }

    run_args.kernel_run_args.push_back(std::move(act_run_args));
    run_args.kernel_run_args.push_back(std::move(weights_run_args));
    run_args.kernel_run_args.push_back(std::move(compute_run_args));

    // ---- Op-owned tensors ----
    // Move the sole-owner indices MeshTensor in first so the TensorArgument below references the
    // parked element (the adapter matches by pointer identity; a vector move keeps the address).
    std::vector<tt::tt_metal::MeshTensor> op_owned_tensors;
    op_owned_tensors.reserve(1);
    op_owned_tensors.push_back(std::move(reader_indices_mesh_tensor));
    const tt::tt_metal::MeshTensor& reader_indices_owned = op_owned_tensors[0];

    // ---- Tensor args ----
    run_args.tensor_args.emplace(conv2d_width_input_tensor, std::cref(a.mesh_tensor()));
    run_args.tensor_args.emplace(conv2d_width_output_tensor, std::cref(output.mesh_tensor()));
    run_args.tensor_args.emplace(conv2d_width_weights_tensor, std::cref(b.mesh_tensor()));
    if (has_bias) {
        run_args.tensor_args.emplace(conv2d_width_bias_tensor, std::cref(bias.value().mesh_tensor()));
    }
    run_args.tensor_args.emplace(conv2d_width_reader_indices_tensor, std::cref(reader_indices_owned));

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_args),
        .op_owned_tensors = std::move(op_owned_tensors),
    };
}

}  // namespace ttnn::prim
