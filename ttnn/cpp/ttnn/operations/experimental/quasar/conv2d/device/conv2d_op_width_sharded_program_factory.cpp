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
#include "ttnn/operations/experimental/quasar/conv2d/device/conv2d_op_width_sharded_program_factory.hpp"
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
#include "ttnn/operations/compute_throttle_utils.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

namespace ttnn::prim::qsr {

using namespace tt::tt_metal;
namespace m2 = tt::tt_metal::experimental;
namespace unary = ttnn::operations::unary;
using ttnn::operations::conv::conv_skip_mcast;
using ttnn::operations::conv::SkipMcast;
using ttnn::prim::CBInfo;
using ttnn::prim::Conv2dCb;
using ttnn::prim::get_cb_info;
using ttnn::prim::get_cb_info_by_name;

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
namespace CMAKE_UNIQUE_NAMESPACE {

// ---- Metal 2.0 resource names (ProgramSpec scope) ----
// DFB accessor names surface kernel-side as dfb::<name> tokens; the ported width-sharded kernels
// reference these exact spellings.
const m2::DFBSpecName DFB_ACT{"act"};                          // mcast result (act reader -> compute)
const m2::DFBSpecName DFB_ACT_RM{"act_row_major"};             // act reader -> compute (tilize input)
const m2::DFBSpecName DFB_ACT_TILIZED{"act_tilized"};          // compute (tilize) -> act reader (mcast src)
const m2::DFBSpecName DFB_WEIGHTS{"weights"};                  // weights reader -> compute
const m2::DFBSpecName DFB_BIAS{"bias"};                        // weights reader -> compute (optional)
const m2::DFBSpecName DFB_MATMUL_PARTIALS{"matmul_partials"};  // compute self-loop (borrows OUTPUT when aliased)
const m2::DFBSpecName DFB_OUT{"out"};                          // compute packer -> OUTPUT (degenerate consumer)
const m2::DFBSpecName DFB_ACT_SHARDED{"act_sharded"};          // borrowed INPUT (act reader address source)
const m2::DFBSpecName DFB_READER_INDICES{"reader_indices"};    // borrowed indices (act reader address source)

const m2::TensorParamName TP_INPUT{"input"};
const m2::TensorParamName TP_OUTPUT{"output"};
const m2::TensorParamName TP_WEIGHTS{"weights"};
const m2::TensorParamName TP_BIAS{"bias"};
const m2::TensorParamName TP_READER_INDICES{"reader_indices"};

const m2::SemaphoreSpecName SEM_ACT_MCAST_SENDER{"act_mcast_sender"};
const m2::SemaphoreSpecName SEM_ACT_MCAST_RECEIVER{"act_mcast_receiver"};

const m2::KernelSpecName KERNEL_ACT{"act_reader"};
const m2::KernelSpecName KERNEL_WEIGHTS{"weights_reader"};
const m2::KernelSpecName KERNEL_COMPUTE{"compute"};

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

ttnn::device_operation::ProgramArtifacts Conv2dWidthShardedProgramFactory::create_program_artifacts(
    const Conv2dParams& operation_attributes, const Conv2dInputs& tensor_args, Tensor& output_tensor) {
    using namespace CMAKE_UNIQUE_NAMESPACE;  // resolve the file-local ids/helpers below
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

    const SkipMcast skip_mcast = conv_skip_mcast(parallelization_config, a.memory_config().memory_layout());
    const bool skip_activation_mcast = skip_mcast.skip_activation_mcast;

    bool pack_relu = fused_activation.has_value() && fused_activation.value().op_type == unary::UnaryOpType::RELU;
    std::map<std::string, std::string> compute_defines;
    if (fused_activation.has_value() && !pack_relu) {
        compute_defines.merge(ttnn::operations::unary::utils::get_defines(
            fused_activation.value().op_type, fused_activation.value().params, "ACTIVATION", "i"));
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
    // This intermediate config tensor must outlive the cached program; it is parked on
    // ProgramArtifacts::op_owned_tensors (the Metal 2.0 replacement for the legacy
    // WorkloadDescriptor::buffers parking) so the adapter keeps it alive in the program cache at a
    // stable address.  op_owned_tensors requires a SOLE-OWNER MeshTensor (#44565), so rather than
    // hand back the shared-storage Tensor that move_config_tensor_to_device() returns, we replicate
    // its layout decision and write the host config straight to device via enqueue_write_tensor(),
    // which returns a sole-owner MeshTensor.
    std::vector<uint32_t> op_trace_metadata =
        ttnn::operations::sliding_window::generate_op_trace_metadata(sliding_window_config);
    std::vector<sliding_window::ShardBoundary> shard_boundaries =
        ttnn::operations::sliding_window::generate_shard_boundaries(sliding_window_config);
    std::vector<std::vector<uint16_t>> conv_sharded_input_top_left_indices =
        ttnn::operations::sliding_window::generate_sliding_window_op_config(
            op_trace_metadata, shard_boundaries, stride_w, true, act_block_h_datums, 0);
    Tensor host_config_tensor = ttnn::operations::sliding_window::construct_on_host_config_tensor(
        conv_sharded_input_top_left_indices, parallel_config, config_tensors_in_dram);

    // Same MemoryConfig that move_config_tensor_to_device() selects for the non-block-sharded path.
    tt::tt_metal::MemoryConfig reader_indices_mem_config = [&]() {
        if (config_tensors_in_dram) {
            return tt::tt_metal::MemoryConfig{TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM};
        }
        std::array<uint32_t, 2> shard_shape{1, (uint32_t)host_config_tensor.logical_shape()[-1]};
        tt::tt_metal::ShardSpec shard_spec(parallel_config.grid, shard_shape, ShardOrientation::ROW_MAJOR);
        return tt::tt_metal::MemoryConfig{
            TensorMemoryLayout::HEIGHT_SHARDED, tt::tt_metal::BufferType::L1_SMALL, shard_spec};
    }();

    auto& cq = a.device()->mesh_command_queue();
    tt::tt_metal::MeshTensor reader_indices_mesh_tensor = tt::tt_metal::enqueue_write_tensor(
        cq, host_config_tensor.host_tensor(), *a.device(), reader_indices_mem_config);
    tt::tt_metal::Buffer* conv_reader_indices_buffer = reader_indices_mesh_tensor.mesh_buffer().get_reference_buffer();
    const uint32_t reader_indices_actual_page_size = conv_reader_indices_buffer->page_size();

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

    // Convenience accessor for CB sizing.
    auto cb = [&](Conv2dCb name) -> const CBInfo& { return get_cb_info_by_name(cb_info, name); };

    // ============================================================================
    //  Build the ProgramSpec
    // ============================================================================
    m2::ProgramSpec spec;
    spec.name = "conv2d_width_sharded";

    const CoreRangeSet all_reader_cores_set(all_reader_cores);

    // ---- Tensor parameters ----
    spec.tensor_parameters.push_back(m2::TensorParameter{.unique_id = TP_INPUT, .spec = a.mesh_tensor().tensor_spec()});
    spec.tensor_parameters.push_back(
        m2::TensorParameter{.unique_id = TP_OUTPUT, .spec = output.mesh_tensor().tensor_spec()});
    spec.tensor_parameters.push_back(
        m2::TensorParameter{.unique_id = TP_WEIGHTS, .spec = b.mesh_tensor().tensor_spec()});
    if (has_bias) {
        spec.tensor_parameters.push_back(
            m2::TensorParameter{.unique_id = TP_BIAS, .spec = bias.value().mesh_tensor().tensor_spec()});
    }
    spec.tensor_parameters.push_back(
        m2::TensorParameter{.unique_id = TP_READER_INDICES, .spec = reader_indices_mesh_tensor.tensor_spec()});

    // ---- Semaphores (act mcast sender/receiver) ----
    spec.semaphores.push_back(m2::SemaphoreSpec{.unique_id = SEM_ACT_MCAST_SENDER, .target_nodes = all_cores});
    spec.semaphores.push_back(m2::SemaphoreSpec{.unique_id = SEM_ACT_MCAST_RECEIVER, .target_nodes = all_cores});

    // ---- Dataflow buffers ----
    // Sizes/formats/backing come straight from get_cb_info() (entry_size = page_size,
    // num_entries = num_pages).  Borrowed DFBs alias their backing tensor (ACT_SHARDED->INPUT,
    // OUT/MATMUL_PARTIALS->OUTPUT, READER_INDICES->indices tensor when L1-resident).
    auto make_dfb = [&](const m2::DFBSpecName& id, Conv2dCb name) {
        const CBInfo& info = cb(name);
        return m2::DataflowBufferSpec{
            .unique_id = id,
            .entry_size = info.page_size,
            .num_entries = info.num_pages,
            .data_format_metadata = info.data_format,
        };
    };

    // ACT (mcast result): real FIFO act reader -> compute.
    spec.dataflow_buffers.push_back(make_dfb(DFB_ACT, Conv2dCb::ACT));
    // ACT_ROW_MAJOR_BFLOAT16: act reader -> compute (tilize input).
    spec.dataflow_buffers.push_back(make_dfb(DFB_ACT_RM, Conv2dCb::ACT_ROW_MAJOR_BFLOAT16));
    // ACT_TILIZED: compute (tilize) -> act reader (mcast source).
    spec.dataflow_buffers.push_back(make_dfb(DFB_ACT_TILIZED, Conv2dCb::ACT_TILIZED));
    // WEIGHTS: weights reader -> compute.
    spec.dataflow_buffers.push_back(make_dfb(DFB_WEIGHTS, Conv2dCb::WEIGHTS));
    // BIAS: weights reader -> compute (optional).
    if (has_bias) {
        spec.dataflow_buffers.push_back(make_dfb(DFB_BIAS, Conv2dCb::BIAS));
    }

    // MATMUL_PARTIALS: compute self-loop accumulator.  Borrowed-from OUTPUT when
    // partials_cb_uses_output (in-place accumulate into the output buffer); the self-loop on the
    // single compute kernel keeps it SPSC-clean.  See CB_TAXONOMY_ANALYSIS.md resolution #2.
    {
        auto dfb = make_dfb(DFB_MATMUL_PARTIALS, Conv2dCb::MATMUL_PARTIALS);
        if (partials_cb_uses_output) {
            dfb.borrowed_from = TP_OUTPUT;
        }
        spec.dataflow_buffers.push_back(std::move(dfb));
    }

    // OUT: compute packer -> OUTPUT shard (borrowed).  Producer-only fake CB; bind a degenerate
    // consumer on the compute kernel itself (width-sharded has no DM output kernel) via a self-loop
    // to satisfy the spec completeness check.  See CB_TAXONOMY_ANALYSIS.md resolution #1.
    {
        auto dfb = make_dfb(DFB_OUT, Conv2dCb::OUT);
        dfb.borrowed_from = TP_OUTPUT;
        spec.dataflow_buffers.push_back(std::move(dfb));
    }

    // ACT_SHARDED: borrowed INPUT; the act reader uses it purely as an address source.  Self-loop on
    // the act reader to satisfy the producer-and-consumer completeness check.
    {
        auto dfb = make_dfb(DFB_ACT_SHARDED, Conv2dCb::ACT_SHARDED);
        dfb.borrowed_from = TP_INPUT;
        spec.dataflow_buffers.push_back(std::move(dfb));
    }

    // READER_INDICES: borrowed indices tensor (L1-resident path) or fresh L1 (DRAM-config path, where
    // the reader fills it from DRAM via TensorAccessor).  Address-source/fake-fill; self-loop on the
    // act reader.
    {
        auto dfb = make_dfb(DFB_READER_INDICES, Conv2dCb::READER_INDICES);
        if (cb(Conv2dCb::READER_INDICES).is_globally_allocated) {
            dfb.borrowed_from = TP_READER_INDICES;
        }
        spec.dataflow_buffers.push_back(std::move(dfb));
    }

    // ---- Compute kernel ----
    // Self-loop bindings: MATMUL_PARTIALS (real accumulator) and OUT (degenerate consumer).
    std::vector<m2::DFBBinding> compute_dfb_bindings = {
        m2::DFBBinding{
            .dfb_spec_name = DFB_ACT, .accessor_name = "act", .endpoint_type = m2::DFBEndpointType::CONSUMER},
        m2::DFBBinding{
            .dfb_spec_name = DFB_ACT_RM,
            .accessor_name = "act_row_major",
            .endpoint_type = m2::DFBEndpointType::CONSUMER},
        m2::DFBBinding{
            .dfb_spec_name = DFB_ACT_TILIZED,
            .accessor_name = "act_tilized",
            .endpoint_type = m2::DFBEndpointType::PRODUCER},
        m2::DFBBinding{
            .dfb_spec_name = DFB_WEIGHTS, .accessor_name = "weights", .endpoint_type = m2::DFBEndpointType::CONSUMER},
        m2::DFBBinding{
            .dfb_spec_name = DFB_MATMUL_PARTIALS,
            .accessor_name = "matmul_partials",
            .endpoint_type = m2::DFBEndpointType::PRODUCER},
        m2::DFBBinding{
            .dfb_spec_name = DFB_MATMUL_PARTIALS,
            .accessor_name = "matmul_partials",
            .endpoint_type = m2::DFBEndpointType::CONSUMER},
        m2::DFBBinding{
            .dfb_spec_name = DFB_OUT, .accessor_name = "out", .endpoint_type = m2::DFBEndpointType::PRODUCER},
        m2::DFBBinding{
            .dfb_spec_name = DFB_OUT, .accessor_name = "out", .endpoint_type = m2::DFBEndpointType::CONSUMER},
    };
    if (has_bias) {
        compute_dfb_bindings.push_back(m2::DFBBinding{
            .dfb_spec_name = DFB_BIAS, .accessor_name = "bias", .endpoint_type = m2::DFBEndpointType::CONSUMER});
    }

    m2::KernelSpec compute_kernel{
        .unique_id = KERNEL_COMPUTE,
        .source = std::filesystem::path(
            "ttnn/cpp/ttnn/operations/experimental/quasar/conv2d/device/kernels/conv_bmm_tilize_metal2.cpp"),
        .compiler_options = {.defines = m2::KernelSpec::CompilerOptions::Defines(compute_defines)},
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
                {"partials_cb_uses_output", (uint32_t)partials_cb_uses_output},
                {"in0_nblocks_w_tilize", input_num_cores},
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
        .hw_config = ttnn::to_compute_hardware_config(device->arch(), compute_kernel_config),
    };

    // ---- Activation reader kernel ----
    // DFB bindings: produces ACT_ROW_MAJOR + ACT (mcast), consumes ACT_TILIZED (mcast source);
    // self-loops the borrowed ACT_SHARDED (input address source) and READER_INDICES.
    m2::DataMovementHardwareConfig act_hw;
    if (device->arch() == tt::ARCH::QUASAR) {
        // QSR: this width-sharded activation reader fills the ACT_ROW_MAJOR/ACT DFB via per-window "stick"
        // sub-tile NOC reads; that pattern stalls the DFB implicit-sync credit accounting (reader pinned at
        // NRBW). Opt out so explicit reserve/push credits stay authoritative (mirrors tilize/transpose HC-sharded).
        act_hw = m2::DataMovementGen2Config{.disable_dfb_implicit_sync_for_all = true};
    } else {
        act_hw = m2::DataMovementGen1Config{.processor = tt::tt_metal::DataMovementProcessor::RISCV_0, .noc = act_noc};
    }
    m2::KernelSpec act_kernel{
        .unique_id = KERNEL_ACT,
        .source = std::filesystem::path("ttnn/cpp/ttnn/operations/experimental/quasar/conv2d/device/kernels/"
                                        "activation_reader_width_sharded_metal2.cpp"),
        .dfb_bindings =
            {
                m2::DFBBinding{
                    .dfb_spec_name = DFB_ACT_RM,
                    .accessor_name = "act_row_major",
                    .endpoint_type = m2::DFBEndpointType::PRODUCER},
                m2::DFBBinding{
                    .dfb_spec_name = DFB_ACT, .accessor_name = "act", .endpoint_type = m2::DFBEndpointType::PRODUCER},
                m2::DFBBinding{
                    .dfb_spec_name = DFB_ACT_TILIZED,
                    .accessor_name = "act_tilized",
                    .endpoint_type = m2::DFBEndpointType::CONSUMER},
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
            },
        .semaphore_bindings =
            {
                m2::SemaphoreBinding{.semaphore_spec_name = SEM_ACT_MCAST_SENDER, .accessor_name = "act_mcast_sender"},
                m2::SemaphoreBinding{
                    .semaphore_spec_name = SEM_ACT_MCAST_RECEIVER, .accessor_name = "act_mcast_receiver"},
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
            },
        .runtime_arg_schema =
            {
                .runtime_arg_names = {"this_core_x", "this_core_y", "num_cores_x"},
            },
        .hw_config = std::move(act_hw),
    };
    if (skip_activation_mcast) {
        act_kernel.compiler_options.defines.insert({"SKIP_MCAST", "1"});
    }
    if (config_tensors_in_dram) {
        // DRAM-config path: the reader DMAs its indices slice in via tensor::reader_indices (the DFB
        // is fresh L1, not borrowed).  Bind the indices tensor and pass its page size as a CTA.
        act_kernel.compiler_options.defines.insert({"CONFIG_TENSOR_IN_DRAM", "1"});
        act_kernel.compile_time_args.insert({"config_page_size", conv_reader_indices_buffer->page_size()});
        act_kernel.tensor_bindings.push_back(
            m2::TensorBinding{.tensor_parameter_name = TP_READER_INDICES, .accessor_name = "reader_indices"});
    }
    // X/Y mcast NoC lookup tables are passed as per-node runtime varargs (variable count: full grid).
    auto full_core_grid = device->compute_with_storage_grid_size();
    act_kernel.advanced_options.num_runtime_varargs = full_core_grid.x + full_core_grid.y;

    // ---- Weights reader kernel ----
    std::vector<m2::DFBBinding> weights_dfb_bindings = {
        m2::DFBBinding{
            .dfb_spec_name = DFB_WEIGHTS, .accessor_name = "weights", .endpoint_type = m2::DFBEndpointType::PRODUCER},
    };
    std::vector<m2::TensorBinding> weights_tensor_bindings = {
        m2::TensorBinding{.tensor_parameter_name = TP_WEIGHTS, .accessor_name = "weights"},
    };
    if (has_bias) {
        weights_dfb_bindings.push_back(m2::DFBBinding{
            .dfb_spec_name = DFB_BIAS, .accessor_name = "bias", .endpoint_type = m2::DFBEndpointType::PRODUCER});
        weights_tensor_bindings.push_back(m2::TensorBinding{.tensor_parameter_name = TP_BIAS, .accessor_name = "bias"});
    }

    m2::DataMovementHardwareConfig weights_hw;
    if (device->arch() == tt::ARCH::QUASAR) {
        weights_hw = m2::DataMovementGen2Config{};
    } else {
        weights_hw =
            m2::DataMovementGen1Config{.processor = tt::tt_metal::DataMovementProcessor::RISCV_1, .noc = weights_noc};
    }
    m2::KernelSpec weights_kernel{
        .unique_id = KERNEL_WEIGHTS,
        .source = std::filesystem::path("ttnn/cpp/ttnn/operations/experimental/quasar/conv2d/device/kernels/"
                                        "weights_reader_width_sharded_metal2.cpp"),
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
            },
        .runtime_arg_schema =
            {
                .runtime_arg_names = {"init_weight_start_tile_id", "is_active"},
            },
        .hw_config = std::move(weights_hw),
    };

    // FUSE_BIAS preprocessor define gates the conditionally-bound dfb::bias / tensor::bias references
    // in the weights and compute kernels.
    if (has_bias) {
        weights_kernel.compiler_options.defines.insert({"FUSE_BIAS", "1"});
        compute_kernel.compiler_options.defines.insert({"FUSE_BIAS", "1"});
    }
    // Width-sharded always binds dfb::act_row_major on compute (mcast tilize-input path), so the shared
    // compute kernel's in0_pretilize_cb_id = dfb::act_row_major reference is always valid here.
    compute_kernel.compiler_options.defines.insert({"HAS_ACT_ROW_MAJOR", "1"});

    spec.kernels.push_back(std::move(act_kernel));
    spec.kernels.push_back(std::move(weights_kernel));
    spec.kernels.push_back(std::move(compute_kernel));

    // ---- Work units ----
    // Compute + readers run on all_cores.  (Width-sharded uses one homogeneous topology; the weights
    // reader is gated per-core by the is_active RTA rather than by node placement, mirroring legacy.)
    spec.work_units.push_back(m2::WorkUnitSpec{
        .name = "wu",
        .kernels = {KERNEL_ACT, KERNEL_WEIGHTS, KERNEL_COMPUTE},
        .target_nodes = all_reader_cores_set,
    });

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

    m2::ProgramRunArgs run_args;
    m2::KernelRunArgs act_run_args{.kernel = KERNEL_ACT};
    m2::KernelRunArgs weights_run_args{.kernel = KERNEL_WEIGHTS};

    for (uint32_t core_index = 0; core_index < total_num_cores; core_index++) {
        uint32_t core_x = core_index % full_core_grid.x;
        uint32_t core_y = core_index / full_core_grid.x;
        CoreCoord core(core_x, core_y);

        m2::KernelRunArgs::RuntimeArgValues& act_rtas = act_run_args.runtime_arg_values;
        m2::AddRuntimeArgsForNode(
            act_rtas,
            core,
            {
                {"this_core_x", core_x},
                {"this_core_y", core_y},
                {"num_cores_x", full_core_grid.x},
            });
        // X/Y mcast lookup tables as per-node varargs.
        m2::AdvancedKernelRunArgs::Varargs varargs;
        varargs.reserve(act_mcast_noc_x.size() + act_mcast_noc_y.size());
        varargs.insert(varargs.end(), act_mcast_noc_x.begin(), act_mcast_noc_x.end());
        varargs.insert(varargs.end(), act_mcast_noc_y.begin(), act_mcast_noc_y.end());
        act_run_args.advanced_options.runtime_varargs.insert({core, std::move(varargs)});

        if (core_index < total_num_active_cores) {
            m2::KernelRunArgs::RuntimeArgValues& weights_rtas = weights_run_args.runtime_arg_values;
            m2::AddRuntimeArgsForNode(
                weights_rtas,
                core,
                {
                    {"init_weight_start_tile_id", core_index * weight_block_w_ntiles},
                    {"is_active", (uint32_t)(core_index < output_num_cores)},
                });
        }
    }

    run_args.kernel_run_args.push_back(std::move(act_run_args));
    run_args.kernel_run_args.push_back(std::move(weights_run_args));
    // Compute kernel has no RTAs.
    run_args.kernel_run_args.push_back(m2::KernelRunArgs{.kernel = KERNEL_COMPUTE});

    // ---- Op-owned tensors ----
    // Move the sole-owner indices MeshTensor in first so the TensorArgument below references the
    // parked element (the adapter matches by pointer identity; a vector move keeps the address).
    std::vector<tt::tt_metal::MeshTensor> op_owned_tensors;
    op_owned_tensors.reserve(1);
    op_owned_tensors.push_back(std::move(reader_indices_mesh_tensor));
    const tt::tt_metal::MeshTensor& reader_indices_owned = op_owned_tensors[0];

    // ---- Tensor args ----
    run_args.tensor_args.emplace(TP_INPUT, std::cref(a.mesh_tensor()));
    run_args.tensor_args.emplace(TP_OUTPUT, std::cref(output.mesh_tensor()));
    run_args.tensor_args.emplace(TP_WEIGHTS, std::cref(b.mesh_tensor()));
    if (has_bias) {
        run_args.tensor_args.emplace(TP_BIAS, std::cref(bias.value().mesh_tensor()));
    }
    run_args.tensor_args.emplace(TP_READER_INDICES, std::cref(reader_indices_owned));

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_args),
        .op_owned_tensors = std::move(op_owned_tensors),
    };
}

}  // namespace ttnn::prim::qsr
