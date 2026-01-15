// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <string>
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
#include "ttnn/operations/compute_throttle_utils.hpp"

namespace ttnn::operations::conv::conv2d::program {

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

Conv2dWidthShardedProgramFactory::cached_program_t Conv2dWidthShardedProgramFactory::create(
    const Conv2dParams& operation_attributes, const Conv2dInputs& tensor_args, Tensor& output_tensor) {
    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();
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

    std::string compute_kernel_path = "ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/conv_bmm_tilize.cpp";
    std::string activation_kernel_path =
        "ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/activation_reader_width_sharded.cpp";
    std::string weights_kernel_path =
        "ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/weights_reader_width_sharded.cpp";

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

    uint32_t act_mcast_sender_semaphore = tt::tt_metal::CreateSemaphore(program, all_cores, 0);    // 0==INVALID
    uint32_t act_mcast_receiver_semaphore = tt::tt_metal::CreateSemaphore(program, all_cores, 0);  // 0==INVALID.

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
        writer_mcast_sender_defines["SKIP_MCAST"] = "1";
    }

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
        input_channels_padded);

    std::vector<std::vector<uint16_t>> conv_sharded_input_top_left_indices =
        ttnn::operations::sliding_window::generate_sliding_window_op_config(
            op_trace_metadata, shard_boundaries, stride_w, true, act_block_h_datums, 0);

    // create sharded ttnn config tensors
    Tensor conv_reader_indices_tensor = ttnn::operations::sliding_window::construct_on_host_config_tensor(
        conv_sharded_input_top_left_indices, parallel_config, config_tensors_in_dram);
    conv_reader_indices_tensor = ttnn::operations::sliding_window::move_config_tensor_to_device(
        conv_reader_indices_tensor, parallel_config, false, a.device(), config_tensors_in_dram);

    log_trace(tt::LogOp, "Conv2D Config Tensor : {}", conv_reader_indices_tensor);
    const tt::tt_metal::DeviceStorage& conv_reader_indices_storage = conv_reader_indices_tensor.device_storage();

    if (config_tensors_in_dram) {
        // The actual CB reader size is difficult to calculate in calculate_L1_size. So instead keep the CB size as the
        // maximum possible size.
        TT_FATAL(
            access_cb_info_by_name(cb_info, Conv2dCb::READER_INDICES).page_size >=
                conv_reader_indices_storage.get_buffer()->page_size(),
            "CB page size {} should be greater than the config tensor page size {}",
            access_cb_info_by_name(cb_info, Conv2dCb::READER_INDICES).page_size,
            conv_reader_indices_storage.get_buffer()->page_size());
    } else {
        access_cb_info_by_name(cb_info, Conv2dCb::READER_INDICES).page_size =
            conv_reader_indices_storage.get_buffer()->page_size();
    }

    // call function to allocate circular buffers
    allocate_cbs(cb_info, program, all_reader_cores, a, output, conv_reader_indices_tensor);
    const tt::tt_metal::CBHandle cb_sharded_act = get_cb_info_by_name(cb_info, Conv2dCb::ACT_SHARDED).handle;
    const tt::tt_metal::CBHandle cb_output = get_cb_info_by_name(cb_info, Conv2dCb::OUT).handle;
    const bool partials_cb_uses_output = get_cb_info_by_name(cb_info, Conv2dCb::MATMUL_PARTIALS).is_globally_allocated;
    const tt::tt_metal::CBHandle cb_partials = get_cb_info_by_name(cb_info, Conv2dCb::MATMUL_PARTIALS).handle;

    std::vector<uint32_t> compute_kernel_args = {
        act_block_w_ntiles,                         // in0_block_w
        act_num_subblocks,                          // in0_num_sublocks
        act_block_num_tiles,                        // in0_block_num_tiles,
        act_subblock_num_tiles,                     // in0_sublock_num_tiles
        act_subblock_h_ntiles * act_num_subblocks,  // reader_num_h_subblocks

        weight_num_subblocks,    // in1_num_sublocks
        weight_block_num_tiles,  // in1_block_num_tiles,
        weight_block_w_ntiles,   // in1_block_w

        num_blocks_act_h_per_core,     // in0_num_blocks_h
        num_blocks_act_w,              // in0_num_blocks_w,
        num_blocks_weight_w_per_core,  // in1_num_blocks_w

        out_subblock_h_ntiles,   // out_sublock_h
        out_subblock_w_ntiles,   // out_sublock_w
        out_subblock_num_tiles,  // out_sublock_num_tiles

        tilize_in0,    // tilize_in0
        untilize_out,  // untilize_out

        bias_ntiles,
        get_cb_info_by_name(cb_info, Conv2dCb::BIAS).index,
        get_cb_info_by_name(cb_info, Conv2dCb::ACT).index,
        get_cb_info_by_name(cb_info, Conv2dCb::WEIGHTS).index,
        get_cb_info_by_name(cb_info, Conv2dCb::ACT_ROW_MAJOR_BFLOAT16).index,
        get_cb_info_by_name(cb_info, Conv2dCb::ACT_SECOND_READER).index,
        get_cb_info_by_name(cb_info, Conv2dCb::MATMUL_PARTIALS).index,
        get_cb_info_by_name(cb_info, Conv2dCb::ACT_TILIZED).index,
        get_cb_info_by_name(cb_info, Conv2dCb::OUT).index,
        0,
        partials_cb_uses_output,
        input_num_cores,  // in0_nblocks_w_tilize. Repeat tilize after all cores have done one round of MCAST.
        false,            // check_skip_compute; not used in width sharded
        pack_relu,
        weight_block_w_ntiles <= 8,  // packer_untilize
        packer_l1_acc,
        has_bias,
        false,  // enable_split_reader (not used in width sharded)
        false,  // enable_activation_reuse (not used in width sharded)
        0,
        0,
        0,
        0,                              // activation reuse related arguments
        static_cast<uint32_t>(false)};  // split_reader_cb_shared (not used in width sharded)

    std::vector<uint32_t> activation_kernel_compile_args = {
        (uint32_t)stride_w,
        (uint32_t)dilation_h,
        (uint32_t)dilation_w,
        (uint32_t)input_size_w,
        (uint32_t)conv_act_c_read_bytes,
        (uint32_t)filter_h,  // Input filter window height
        (uint32_t)filter_w,  // Input filter window width
        (uint32_t)act_block_h_datums,
        (uint32_t)act_block_num_tiles,
        (uint32_t)input_num_cores,
        (uint32_t)num_blocks_act_h_per_core,
        (uint32_t)per_core_num_blocks_act_w,
        (uint32_t)act_mcast_sender_semaphore,
        (uint32_t)act_mcast_receiver_semaphore,
        (uint32_t)act_mcast_start.x,
        (uint32_t)act_mcast_start.y,
        (uint32_t)act_mcast_end.x,
        (uint32_t)act_mcast_end.y,
        (uint32_t)act_block_num_tiles * tt::tile_size(tilized_act_df),
        (uint32_t)output_num_cores,
        (uint32_t)all_reader_cores.size(),
        get_cb_info_by_name(cb_info, Conv2dCb::ACT).index,
        get_cb_info_by_name(cb_info, Conv2dCb::ACT_SHARDED).index,
        get_cb_info_by_name(cb_info, Conv2dCb::READER_INDICES).index,
        get_cb_info_by_name(cb_info, Conv2dCb::L1_ARRAY).index,
        get_cb_info_by_name(cb_info, Conv2dCb::ACT_ROW_MAJOR_BFLOAT16).index,
        get_cb_info_by_name(cb_info, Conv2dCb::ACT_TILIZED).index};

    std::vector<uint32_t> weights_kernel_compile_args = {
        get_cb_info_by_name(cb_info, Conv2dCb::WEIGHTS).index,          // cb_id_weight
        act_block_w_ntiles / (filter_h * filter_w),                     // core_in_channels_ntiles
        filter_h * filter_w,                                            // window_size_hw
        weight_block_w_ntiles,                                          // weight_block_width_ntiles
        weight_block_num_tiles,                                         // weight_block_num_tiles
        weight_matrix_width_ntiles,                                     // weight_matrix_width_ntiles
        (weight_matrix_width_ntiles * input_channels_padded) / 32,      // weight_next_channel_stride_h
        weight_matrix_width_ntiles * weight_block_in_channels_ntiles,   // weight_next_block_this_core_stride_h
        weight_matrix_width_ntiles * weight_block_in_channels_ntiles *  // weight_next_block_other_core_stride_h
            per_core_num_blocks_act_w,
        input_num_cores,            // other_core_weight_height_blocks
        per_core_num_blocks_act_w,  // this_core_weight_height_blocks
        num_blocks_act_h_per_core,
        get_cb_info_by_name(cb_info, Conv2dCb::BIAS).index,
        (uint32_t)has_bias};

    if (config_tensors_in_dram) {
        reader_defines["CONFIG_TENSOR_IN_DRAM"] = "1";
        activation_kernel_compile_args.push_back(conv_reader_indices_storage.get_buffer()->address());
        activation_kernel_compile_args.push_back(conv_reader_indices_storage.get_buffer()->page_size());
        tt::tt_metal::TensorAccessorArgs(conv_reader_indices_storage.get_buffer())
            .append_to(activation_kernel_compile_args);
    }

    for (uint32_t index = 0; index < activation_kernel_compile_args.size(); index++) {
        log_debug(tt::LogOp, "activation_kernel_compile_args[{}] = {}", index, activation_kernel_compile_args[index]);
    }

    tt::tt_metal::TensorAccessorArgs(b.buffer()).append_to(weights_kernel_compile_args);
    tt::tt_metal::TensorAccessorArgs(bias ? bias->buffer() : nullptr).append_to(weights_kernel_compile_args);

    auto act_kernel_id = CreateKernel(
        program,
        activation_kernel_path,
        all_reader_cores,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = act_noc,
            .compile_args = activation_kernel_compile_args,
            .defines = reader_defines});

    auto weights_kernel_id = CreateKernel(
        program,
        weights_kernel_path,
        all_cores,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
            .noc = weights_noc,
            .compile_args = weights_kernel_compile_args,
            .defines = writer_defines});

    CreateKernel(
        program,
        compute_kernel_path,
        all_cores,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .compile_args = compute_kernel_args,
            .defines = compute_defines});

    auto full_core_grid = device->compute_with_storage_grid_size();
    std::vector<uint32_t> act_mcast_noc_y;
    std::vector<uint32_t> act_mcast_noc_x;

    act_mcast_noc_x.reserve(full_core_grid.x);
    for (uint32_t core_index = 0; core_index < full_core_grid.x; core_index++) {
        act_mcast_noc_x.push_back(device->worker_core_from_logical_core(CoreCoord(core_index, 0)).x);
    }

    act_mcast_noc_y.reserve(full_core_grid.y);
    for (uint32_t core_index = 0; core_index < full_core_grid.y; core_index++) {
        act_mcast_noc_y.push_back(device->worker_core_from_logical_core(CoreCoord(0, core_index)).y);
    }

    uint32_t bias_base_address = 0;
    if (bias) {
        bias_base_address = bias.value().buffer()->address();
    }
    auto total_num_active_cores = std::max(input_num_cores, output_num_cores);
    auto total_num_cores = all_reader_cores.size();
    for (uint32_t core_index = 0; core_index < total_num_cores; core_index++) {
        uint32_t core_x = core_index % full_core_grid.x;
        uint32_t core_y = core_index / full_core_grid.x;
        std::vector<uint32_t> rt_args = {
            core_x,
            core_y,
            full_core_grid.x,  // num_cores_x
        };

        // Mcast X Lookup table
        rt_args.insert(rt_args.end(), act_mcast_noc_x.begin(), act_mcast_noc_x.end());

        // Mcast Y Lookup Table
        rt_args.insert(rt_args.end(), act_mcast_noc_y.begin(), act_mcast_noc_y.end());

        SetRuntimeArgs(program, act_kernel_id, CoreCoord(core_x, core_y), rt_args);

        // Weights kernel is not placed on inactive cores.
        if (core_index < total_num_active_cores) {
            SetRuntimeArgs(
                program,
                weights_kernel_id,
                CoreCoord(core_x, core_y),
                {core_index * weight_block_w_ntiles,
                 b.buffer()->address(),
                 bias_base_address,
                 (uint32_t)(core_index < output_num_cores)});
        }
    }

    post_conv2d_op_memory_checks(program, operation_attributes, tensor_args, output_tensor);
    return cached_program_t{
        std::move(program),
        shared_variables_t{
            .cb_sharded_act = cb_sharded_act,
            .cb_output = cb_output,
            .cb_partials = cb_partials,
            .partials_cb_uses_output = partials_cb_uses_output,
            .has_bias = has_bias,
            .full_core_grid = full_core_grid,
            .weights_kernel_id = weights_kernel_id,
            .total_num_active_cores = total_num_active_cores,
            .conv_reader_indices_storage = conv_reader_indices_storage,
        }};
}

void Conv2dWidthShardedProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const Conv2dParams& /*operation_attributes*/,
    const Conv2dInputs& tensor_args,
    Tensor& output_tensor) {
    auto* src_buffer_a = tensor_args.a.buffer();
    auto* src_buffer_b = tensor_args.b.buffer();
    auto& program = cached_program.program;
    const auto& shared_variables = cached_program.shared_variables;

    auto& weights_kernel_runtime_args = GetRuntimeArgs(program, shared_variables.weights_kernel_id);
    for (uint32_t core_index = 0; core_index < shared_variables.total_num_active_cores; core_index++) {
        uint32_t core_x = core_index % shared_variables.full_core_grid.x;
        uint32_t core_y = core_index / shared_variables.full_core_grid.x;

        auto& this_core_weights_kernel_runtime_args = weights_kernel_runtime_args[core_x][core_y];
        this_core_weights_kernel_runtime_args[1] = src_buffer_b->address();
        if (shared_variables.has_bias) {
            this_core_weights_kernel_runtime_args[2] = tensor_args.bias.value().buffer()->address();
        }
    }

    UpdateDynamicCircularBufferAddress(program, shared_variables.cb_sharded_act, *src_buffer_a);
    UpdateDynamicCircularBufferAddress(program, shared_variables.cb_output, *output_tensor.buffer());
    if (shared_variables.partials_cb_uses_output) {
        UpdateDynamicCircularBufferAddress(program, shared_variables.cb_partials, *output_tensor.buffer());
    }
}

}  // namespace ttnn::operations::conv::conv2d::program
