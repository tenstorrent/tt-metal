// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "conv2d_op.hpp"

#include <cstdint>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>
#include <ttnn/operations/cb_utils.hpp>
#include <ttnn/operations/core/compute_kernel/compute_kernel_config.hpp>
#include <ttnn/operations/conv/conv2d/conv2d_utils.hpp>
#include <ttnn/operations/conv/conv2d/conv2d_op_program_factory_common.hpp>
#include <ttnn/operations/sliding_window/sliding_window.hpp>
#include <ttnn/tensor/shape/shape.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

// Pool-related includes for reusing pool kernels
#include <ttnn/operations/pool/pool_utils.hpp>

// Unary operations for activation functions
#include <ttnn/operations/eltwise/unary/common/unary_op_utils.hpp>

namespace ttnn::operations::conv {
namespace conv2d {

tt::tt_metal::operation::ProgramWithCallbacks multi_core_conv2d_depthwise(
    tt::tt_metal::Program& program,
    const Tensor& a,
    const Tensor& b,
    const ttnn::Shape& ashape,
    const std::optional<const Tensor>& bias,
    const sliding_window::SlidingWindowConfig& sliding_window_config,
    const sliding_window::ParallelConfig& parallel_config,
    const std::vector<uint32_t>& op_trace_metadata,
    const std::vector<sliding_window::ShardBoundary>& shard_boundaries,
    uint32_t output_channels,
    uint32_t groups,
    bool untilize_out,
    bool has_bias,
    const std::optional<unary::UnaryWithParam>& fused_activation,
    const Conv2dParallelizationConfig& parallelization_config,
    const Conv2dBlockConfig& block_config,
    bool is_col_major,
    Tensor& output,
    DeviceComputeKernelConfig compute_kernel_config,
    bool enable_act_double_buffer,
    bool enable_weights_double_buffer,
    bool full_inner_dim,
    bool enable_activation_reuse,
    bool config_tensors_in_dram,
    std::optional<bool> force_split_reader) {
    log_info(tt::LogOp, "output_channels: {}, groups: {}", output_channels, groups);
    TT_FATAL(
        groups == ashape[3] && groups == output_channels,
        "Depthwise factory requires groups == input_channels == output_channels");

    log_debug(tt::LogOp, "Creating 2D depthwise convolution program using pool-based approach");
    log_debug(
        tt::LogOp,
        "Input shape: [{}, {}, {}, {}], groups: {}, output_channels: {}",
        ashape[0],
        ashape[1],
        ashape[2],
        ashape[3],
        groups,
        output_channels);

    // Extract parameters from sliding window config for pool kernel
    const uint32_t kernel_h = sliding_window_config.window_hw.first;
    const uint32_t kernel_w = sliding_window_config.window_hw.second;
    const uint32_t stride_h = sliding_window_config.stride_hw.first;
    const uint32_t stride_w = sliding_window_config.stride_hw.second;
    const uint32_t pad_t = sliding_window_config.padding[0];
    const uint32_t pad_b = sliding_window_config.padding[1];
    const uint32_t pad_l = sliding_window_config.padding[2];
    const uint32_t pad_r = sliding_window_config.padding[3];
    const uint32_t num_shards_c = parallelization_config.num_cores_c_in;

    log_debug(
        tt::LogOp,
        "Kernel: {}x{}, Stride: {}x{}, Padding: t={} b={} l={} r={}",
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_t,
        pad_b,
        pad_l,
        pad_r);

    // Get output shape from sliding window config
    auto output_shape = sliding_window_config.get_output_shape();
    const uint32_t out_h = output_shape[1];
    const uint32_t out_w = output_shape[2];

    log_debug(
        tt::LogOp, "Sliding window output shape: [{}, {}, {}], groups: {}", ashape[0], out_h, out_w, output_channels);
    log_debug(
        tt::LogOp,
        "Actual output tensor shape: [{}, {}, {}, {}]",
        output.logical_shape()[0],
        output.logical_shape()[1],
        output.logical_shape()[2],
        output.logical_shape()[3]);

    // Suppress unused variable warnings for parameters that will be used in kernel integration
    (void)stride_h;
    (void)stride_w;
    (void)pad_t;
    (void)pad_b;
    (void)pad_l;
    (void)pad_r;
    (void)out_h;
    (void)out_w;

    // For depthwise convolution, we need to work with the flattened tensor layout
    // The output tensor comes in as [1, 1, batch*out_h*out_w, out_channels] = [1, 1, 64, 4]
    // We need to set up pool parameters to handle this correctly
    ttnn::operations::pool::FactoryParameters params = ttnn::operations::pool::get_factory_parameters(
        parallelization_config.num_cores_c_out,  // num_shards_c - simple sharding
        a.dtype(),                               // input dtype
        output.dtype(),                          // output dtype
        kernel_h,
        kernel_w,
        output_channels,                                 // Use output channels (4) for pool calculations
        ttnn::operations::pool::Pool2DType::AVG_POOL2D,  // Use AVG_POOL2D as base for depthwise
        false,                                           // return_indices
        output.layout()                                  // output layout
    );
    params.split_reader = false;  // Depthwise conv uses single reader for simplicity

    // Calculate effective tiles the same way as pool factory
    uint32_t effective_tiles = (kernel_h * kernel_w * params.in_ntiles_c * 32 + 1023) / 1024;
    log_debug(tt::LogOp, "Effective tiles for depthwise conv: {}", effective_tiles);

    // Create circular buffers for depthwise convolution (same as pool factory pattern)
    uint32_t next_cb_index = 0;

    // Input CB - using same logic as pool factory with effective_tiles
    const uint32_t in_cb_id = next_cb_index++;
    const uint32_t in_cb_pagesize = tt::tile_size(params.data_format);
    const uint32_t in_cb_npages = params.multi_buffering_factor * std::min(effective_tiles, 3u);
    tt::tt_metal::create_cb(in_cb_id, program, parallel_config.grid, in_cb_pagesize, in_cb_npages, params.data_format);
    log_debug(tt::LogOp, "CB {} (input) :: PS = {}, NP = {}", in_cb_id, in_cb_pagesize, in_cb_npages);

    // mul_cb - stores results of element-wise multiplication (using effective_tiles logic)
    const uint32_t mul_cb_id = next_cb_index++;
    const uint32_t mul_cb_pagesize = tt::tile_size(params.data_format);
    const uint32_t mul_cb_npages = std::min(effective_tiles, 3u) * params.multi_buffering_factor;
    tt::tt_metal::create_cb(
        mul_cb_id, program, parallel_config.grid, mul_cb_pagesize, mul_cb_npages, params.data_format);
    log_debug(tt::LogOp, "CB {} (mul_cb) :: PS = {}, NP = {}", mul_cb_id, mul_cb_pagesize, mul_cb_npages);

    // weight_cb - stores weight tensors (match pool factory sizing)
    const uint32_t weight_cb_id = next_cb_index++;
    const uint32_t weight_cb_pagesize = tt::tile_size(params.data_format);
    const uint32_t weight_cb_npages = std::min(params.in_ntiles_c, 8u);  // Max 8 tiles per width (because of dest)
    tt::tt_metal::create_cb(
        weight_cb_id, program, parallel_config.grid, weight_cb_pagesize, weight_cb_npages, params.data_format);
    log_debug(tt::LogOp, "CB {} (weight_cb) :: PS = {}, NP = {}", weight_cb_id, weight_cb_pagesize, weight_cb_npages);

    // Output CB - match pool factory sizing based on output layout (same as pool factory)
    const bool is_output_tiled = output.layout() == Layout::TILE;
    uint32_t out_cb_pagesize;
    uint32_t out_cb_npages;

    if (is_output_tiled) {
        out_cb_pagesize = tt::tile_size(params.output_data_format);
        out_cb_npages =
            output.shard_spec().value().shape[0] * output.shard_spec().value().shape[1] / tt::constants::TILE_HW;
    } else {
        log_info(tt::LogOp, "Shard spec: {}", output.shard_spec().value());
        log_info(tt::LogOp, "params.out_ntiles_c: {}", params.out_ntiles_c);
        // Match pool factory logic for non-tiled output
        out_cb_pagesize = std::min(
                              static_cast<uint32_t>(tt::constants::TILE_WIDTH),
                              tt::round_up(output.shard_spec().value().shape[1], tt::constants::TILE_WIDTH)) *
                          params.nbytes;
        out_cb_npages = output.shard_spec().value().shape[0] * params.out_ntiles_c;
    }

    // CRITICAL: Use same CB creation pattern as pool factory to avoid CB ID mismatches
    log_info(
        tt::LogOp, "CB {} (output) :: PS = {}, NP = {} [BOUND TO OUTPUT BUFFER]", 0, out_cb_pagesize, out_cb_npages);
    const auto [out_cb_id, out_cb] = tt::tt_metal::create_cb(
        next_cb_index++,
        program,
        parallel_config.grid,
        out_cb_pagesize,
        out_cb_npages,
        params.output_data_format,
        output.buffer());

    // Scalar CB - stores scalar values for pool operations (match pool factory sizing)
    const uint32_t in_scalar_cb_id_0 = next_cb_index++;
    const uint32_t in_scalar_cb_pagesize = tt::tile_size(params.data_format);
    const uint32_t in_scalar_cb_npages = params.multi_buffering_factor;
    TT_FATAL(in_scalar_cb_npages <= 2, "Kernel logic relies on scalar cb page number being <= 2");
    tt::tt_metal::create_cb(
        in_scalar_cb_id_0,
        program,
        parallel_config.grid,
        in_scalar_cb_pagesize,
        in_scalar_cb_npages,
        params.data_format);
    log_debug(
        tt::LogOp,
        "CB {} (in_scalar_cb_0) :: PS = {}, NP = {}",
        in_scalar_cb_id_0,
        in_scalar_cb_pagesize,
        in_scalar_cb_npages);

    // Clear value CB - stores "clear value" (-inf for maxpool, 0 for avgpool)
    const uint32_t clear_value_cb_id = next_cb_index++;
    tt::tt_metal::create_cb(
        clear_value_cb_id, program, parallel_config.grid, tt::tile_size(params.data_format), 1, params.data_format);
    log_debug(
        tt::LogOp,
        "CB {} (clear_value_cb) :: PS = {}, NP = {}",
        clear_value_cb_id,
        tt::tile_size(params.data_format),
        1);

    // Reader indices will be created after out_nhw_per_core is calculated

    // Now create the actual pool kernels by referencing their paths directly
    log_debug(tt::LogOp, "Setting up pool kernels for depthwise convolution");

    // Calculate out_nhw_per_core exactly like pool factory does
    // Use the output tensor's shard specification rather than simple division
    const uint32_t out_nhw_per_core = output.shard_spec()->shape[0];

    // Generate proper top_left_indices using sliding window infrastructure
    // This ensures correct multi-core memory access patterns like pool operations
    log_debug(
        tt::LogOp,
        "Generating proper sliding window indices: cores={}, output_per_core={}",
        parallel_config.grid.num_cores(),
        out_nhw_per_core);

    // Use the sliding window infrastructure to generate proper reader indices
    // This accounts for stride patterns, shard boundaries, and memory access patterns
    std::vector<std::vector<uint16_t>> top_left_indices = sliding_window::generate_sliding_window_op_config(
        op_trace_metadata, shard_boundaries, stride_w
        // true,  // is_conv = true (for depthwise convolution)
        // 0,     // reader0_datums = 0 (use defaults)
        // 0,     // reader1_datums = 0 (use defaults)
        // true   // pad_cores = true
    );

    // for (const auto& core_indices : top_left_indices) {
    //     log_info(tt::LogOp, "Core indices size: {}", core_indices.size());
    //     for (size_t i = 0; i < std::min(core_indices.size(), size_t(10)); ++i) {
    //         log_info(tt::LogOp, "  Index[{}] = {}", i, core_indices[i]);
    //     }
    // }

    log_debug(
        tt::LogOp,
        "Generated proper sliding window indices: cores={}, indices_per_core={}",
        top_left_indices.size(),
        top_left_indices.empty() ? 0 : top_left_indices[0].size());
    if (!top_left_indices.empty() && !top_left_indices[0].empty()) {
        log_debug(
            tt::LogOp,
            "Core 0 indices: size={}, first_few=[{}, {}, {}...]",
            top_left_indices[0].size(),
            !top_left_indices[0].empty() ? top_left_indices[0][0] : 0,
            top_left_indices[0].size() > 1 ? top_left_indices[0][1] : 0,
            top_left_indices[0].size() > 2 ? top_left_indices[0][2] : 0);
    }

    log_debug(tt::LogOp, "About to call construct_on_host_config_tensor with {} cores", top_left_indices.size());
    Tensor reader_indices = sliding_window::construct_on_host_config_tensor(top_left_indices, parallel_config);
    log_debug(tt::LogOp, "construct_on_host_config_tensor completed successfully");
    const auto& reader_shape = reader_indices.logical_shape();
    log_debug(tt::LogOp, "reader_indices shape: rank={}, shape_size={}", reader_shape.rank(), reader_shape.size());
    for (uint32_t i = 0; i < reader_shape.rank(); ++i) {
        log_debug(tt::LogOp, "  shape[{}] = {}", i, reader_shape[i]);
    }

    bool is_block_sharded = (a.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED);
    log_debug(tt::LogOp, "About to call move_config_tensor_to_device with is_block_sharded={}", is_block_sharded);
    Tensor reader_indices_on_device =
        sliding_window::move_config_tensor_to_device(reader_indices, parallel_config, is_block_sharded, a.device());
    log_debug(tt::LogOp, "move_config_tensor_to_device completed successfully");

    const auto& moved_shape = reader_indices_on_device.logical_shape();
    log_debug(
        tt::LogOp, "Reader indices tensor on device: rank={}, shape_size={}", moved_shape.rank(), moved_shape.size());
    for (uint32_t i = 0; i < moved_shape.rank(); ++i) {
        log_debug(tt::LogOp, "  moved_shape[{}] = {}", i, moved_shape[i]);
    }

    // Create reader indices CB using the same pattern as pool factory
    const tt::tt_metal::DeviceStorage& reader_indices_storage = reader_indices_on_device.device_storage();
    const uint32_t in_reader_indices_cb_id = next_cb_index++;
    const uint32_t reader_indices_size = top_left_indices[0].size();
    const uint32_t in_reader_indices_cb_pagesize = tt::round_up(reader_indices_size * sizeof(uint16_t), 4);
    constexpr uint32_t in_reader_indices_cb_npages = 1;

    tt::tt_metal::create_cb(
        in_reader_indices_cb_id,
        program,
        parallel_config.grid,
        in_reader_indices_cb_pagesize,
        in_reader_indices_cb_npages,
        tt::DataFormat::UInt16,
        reader_indices_storage.get_buffer());

    log_debug(
        tt::LogOp,
        "CB {} (reader_indices_cb) :: PS = {}, NP = {}, reader_size={}",
        in_reader_indices_cb_id,
        in_reader_indices_cb_pagesize,
        in_reader_indices_cb_npages,
        reader_indices_size);

    const uint32_t in_w = ashape[2];

    // Calculate parameters dynamically like pool factory does
    const auto& input_shape = a.padded_shape();
    const uint32_t shard_width = a.shard_spec()->shape[1];
    const uint32_t in_c = ashape[3];  // input channels

    // Calculate in_c_per_shard_ceil the same way as pool factory
    const uint32_t in_c_per_shard_ceil = in_c % shard_width != 0 && num_shards_c > 1
                                             ? (in_c - (in_c % shard_width)) / (num_shards_c - 1)
                                             : in_c / num_shards_c;

    // Calculate in_nbytes_c dynamically based on shard distribution
    const uint32_t in_nbytes_c = in_c_per_shard_ceil * params.nbytes;  // row of input (channels)

    // Calculate shard_width_bytes dynamically
    const uint32_t shard_width_bytes = input_shape[3] / num_shards_c * params.nbytes;

    // Add the same validation check as pool factory
    TT_FATAL(
        input_shape[3] % num_shards_c == 0,
        "Input channels {} should be divisible by number of shards {}",
        input_shape[3],
        num_shards_c);

    // Calculate in_nbytes_leftover the same way as pool factory
    const uint32_t in_nbytes_leftover =
        params.is_wide_reduction &&
                (input_shape[3] / num_shards_c) % (params.MAX_TILES_PER_REDUCTION * tt::constants::TILE_WIDTH) != 0
            ? tt::round_up(
                  (input_shape[3] / num_shards_c) % (params.MAX_TILES_PER_REDUCTION * tt::constants::TILE_WIDTH),
                  tt::constants::TILE_WIDTH) *
                  params.nbytes
            : tt::round_up(input_shape[3] / num_shards_c, tt::constants::TILE_WIDTH) * params.nbytes;

    // Calculate in_cb_sz and in_nblocks_c dynamically like pool factory does
    uint32_t in_cb_sz = 0;
    uint32_t in_nblocks_c = 1;

    // For depthwise convolution, we don't return indices, so use the else branch logic from pool factory
    if (params.is_wide_reduction) {
        in_cb_sz = params.MAX_TILES_PER_REDUCTION * tt::constants::TILE_WIDTH * params.num_tilized_rows;
        in_nblocks_c = std::ceil((float)params.in_ntiles_c / params.MAX_TILES_PER_REDUCTION);
    } else {
        in_cb_sz = params.in_ntiles_c * tt::constants::TILE_WIDTH * params.num_tilized_rows;
    }
    const bool is_output_block_format = is_block_float(output.dtype());

    // Add debug logging similar to pool factory
    log_debug(tt::LogOp, "Calculated parameters like pool factory:");
    log_debug(tt::LogOp, "  in_c_per_shard_ceil: {}", in_c_per_shard_ceil);
    log_debug(tt::LogOp, "  in_nbytes_c: {}", in_nbytes_c);
    log_debug(tt::LogOp, "  shard_width_bytes: {}", shard_width_bytes);
    log_debug(tt::LogOp, "  in_nbytes_leftover: {}", in_nbytes_leftover);
    log_debug(tt::LogOp, "  in_cb_sz: {}", in_cb_sz);
    log_debug(tt::LogOp, "  in_nblocks_c: {}", in_nblocks_c);
    log_debug(tt::LogOp, "  is_wide_reduction: {}", params.is_wide_reduction);
    log_debug(tt::LogOp, "  MAX_TILES_PER_REDUCTION: {}", params.MAX_TILES_PER_REDUCTION);
    log_debug(tt::LogOp, "  split_reader: {}", params.split_reader);
    log_debug(tt::LogOp, "  multi_buffering_factor: {}", params.multi_buffering_factor);
    log_debug(tt::LogOp, "  max_rows_for_reduction: {}", params.max_rows_for_reduction);
    log_debug(tt::LogOp, "  in_ntiles_c: {}", params.in_ntiles_c);

    // Calculate proper BF16 scalar values for depthwise convolution
    // Use AVG_POOL2D with divisor_override=1 to get SUM instead of AVERAGE
    // This gives us summation of element-wise multiplications without division by kernel size
    const uint32_t bf16_scalar = ttnn::operations::pool::get_bf16_pool_scalar(
        ttnn::operations::pool::Pool2DType::AVG_POOL2D, kernel_h, kernel_w, 1);  // divisor_override=1
    const uint32_t bf16_init_value =
        ttnn::operations::pool::get_bf16_pool_init_value(ttnn::operations::pool::Pool2DType::AVG_POOL2D);

    // Calculate proper padding and dilation values
    const uint32_t dilation_h = 1;  // Depthwise convolution typically uses dilation=1
    const uint32_t dilation_w = 1;  // Depthwise convolution typically uses dilation=1
    const uint32_t pad_h = pad_t + pad_b;
    const uint32_t pad_w = pad_l + pad_r;
    const uint32_t eff_kernel_h = ((kernel_h - 1) * dilation_h) + 1;
    const uint32_t eff_kernel_w = ((kernel_w - 1) * dilation_w) + 1;

    // Calculate one_scalar_per_core like pool factory does
    // Use divisor_override=1 to get SUM behavior with AVG_POOL2D
    const bool one_scalar_per_core = ttnn::operations::pool::is_pool_op_one_scalar_per_core(
        ttnn::operations::pool::Pool2DType::AVG_POOL2D, false, 0, 0, true, pad_h, pad_w, 1);

    // Calculate zero_pages the same way as pool factory
    const bool zero_pages = is_output_tiled && is_output_block_format;

    // Conditionally allocate pre_tilize_cb like pool factory - only needed for TILED output
    uint32_t pre_tilize_cb_id = 32;  // default invalid CB ID

    if (is_output_tiled) {
        pre_tilize_cb_id = next_cb_index++;
        const uint32_t pre_tilize_cb_pagesize = tt::constants::TILE_WIDTH * params.nbytes;
        const uint32_t pre_tilize_cb_npages = tt::constants::TILE_HEIGHT * params.in_ntiles_c;
        tt::tt_metal::create_cb(
            pre_tilize_cb_id,
            program,
            parallel_config.grid,
            pre_tilize_cb_pagesize,
            pre_tilize_cb_npages,
            params.data_format);
        log_debug(
            tt::LogOp,
            "CB {} (pre_tilize_cb) :: PS = {}, NP = {}",
            pre_tilize_cb_id,
            pre_tilize_cb_pagesize,
            pre_tilize_cb_npages);
    }

    // Raw input CB - match pool factory sizing pattern
    const uint32_t raw_in_cb_npages = a.shard_spec().value().shape[0];
    const uint32_t raw_in_cb_pagesize = in_nbytes_c;
    const auto [raw_in_cb_id, raw_in_cb] = tt::tt_metal::create_cb(
        next_cb_index++,
        program,
        parallel_config.grid,
        raw_in_cb_pagesize,
        raw_in_cb_npages,
        params.data_format,
        a.buffer());
    log_debug(
        tt::LogOp,
        "CB {} (raw_in_cb) :: PS = {}, NP = {} [BOUND TO INPUT BUFFER]",
        raw_in_cb_id,
        raw_in_cb_pagesize,
        raw_in_cb_npages);

    // For depthwise convolution, we don't need the complex CBs that pool factory creates
    // We'll point unused CB arguments to existing valid CBs to avoid kernel hangs

    log_debug(tt::LogOp, "Setting up pool kernel arguments: out_nhw_per_core={}, in_w={}", out_nhw_per_core, in_w);

    // Set up reader arguments following exact pool factory pattern (47 args)
    // First, create base arguments for reader0
    std::vector<uint32_t> reader0_ct_args = {
        out_nhw_per_core,               // 0
        kernel_h,                       // 1
        kernel_w,                       // 2
        pad_w,                          // 3 - pad_w (FIXED: use total width padding)
        in_nbytes_leftover,             // 4
        in_w,                           // 5
        in_c_per_shard_ceil,            // 6
        params.split_reader,            // 7 - split_reader (FIXED: use calculated value)
        0,                              // 8 - split reader id for reader0
        bf16_scalar,                    // 9 - bf16_scalar (FIXED: use calculated value)
        bf16_init_value,                // 10 - bf16_init_value (FIXED: use calculated value)
        in_nblocks_c,                   // 11
        in_cb_sz,                       // 12
        params.max_rows_for_reduction,  // 13 - max_rows_for_reduction (FIXED: use calculated value)
        0,                              // 14 - ceil_pad_w
        in_cb_id,                       // 15 - in_cb_id_0
        in_cb_id,                       // 16 - in_cb_id_1 (point to in_cb_id for depthwise - no split reader)
        raw_in_cb_id,                   // 17 - raw_in_cb_id (FIXED: now uses actual CB)
        in_reader_indices_cb_id,        // 18 - in_reader_indices_cb_id
        in_scalar_cb_id_0,              // 19 - in_scalar_cb_id_0
        in_scalar_cb_id_0,              // 20 - in_scalar_cb_id_1 (point to scalar_cb_0 for depthwise)
        clear_value_cb_id,              // 21 - in_idx_cb_id (point to clear_value_cb for depthwise)
        mul_cb_id,                      // 22 - pack_tmp_cb_id (point to mul_cb for depthwise)
        weight_cb_id,                   // 23 - pack_idx_tmp_cb_id (point to weight_cb for depthwise)
        out_cb_id,                      // 24 - right_inc_cb_id (point to out_cb for depthwise)
        out_cb_id,                      // 25 - down_left_wrap_inc_cb_id (point to out_cb for depthwise)
        out_cb_id,                      // 26 - up_left_wrap_inc_cb_id (point to out_cb for depthwise)
        clear_value_cb_id,              // 27 - clear_value_cb_id
        static_cast<uint32_t>(ttnn::operations::pool::Pool2DType::AVG_POOL2D),  // 28 - pool_type
        one_scalar_per_core,            // 29 - one_scalar_per_core (FIXED: use calculated value)
        clear_value_cb_id,              // 30 - config_cb_id (point to clear_value_cb for depthwise)
        in_nbytes_c,                    // 31
        shard_width_bytes,              // 32
        params.multi_buffering_factor,  // 33 - multi_buffering_factor (FIXED: use calculated value)
        stride_w,                       // 34
        dilation_h,                     // 35 - dilation_h (FIXED: use actual value)
        dilation_w,                     // 36 - dilation_w (FIXED: use actual value)
        false,                          // 37 - return_indices
        pad_t,                          // 38
        pad_l,                          // 39
        0,                              // 40 - right_inc
        0,                              // 41 - down_left_wrap_inc
        0,                              // 42 - up_left_wrap_inc
        zero_pages,                     // 43 - zero_pages (FIXED: use calculated value)
        out_cb_id,                      // 44
        out_cb_id,                      // 45 - out_idx_cb_id (point to out_cb for depthwise)
        weight_cb_id                    // 46 - weight_cb_id (for L1 storage)
    };

    // Add tensor accessor args for weight buffer (similar to sharded factory)
    // This allows the kernel to access weight data from DRAM via TensorAccessor
    tt::tt_metal::TensorAccessorArgs(b.buffer()).append_to(reader0_ct_args);

    // Create reader1 arguments by copying reader0 and updating reader_id
    std::vector<uint32_t> reader1_ct_args = reader0_ct_args;
    reader1_ct_args[8] = 1;  // split reader id for reader1

    // Set up compute arguments following exact pool factory pattern (33 args)
    std::vector<uint32_t> compute_ct_args = {
        params.in_ntiles_c,             // 0 - in_ntiles_c (FIXED: use calculated value)
        kernel_h * kernel_w,            // 1
        params.split_reader,            // 2 - split_reader (FIXED: use calculated value)
        out_nhw_per_core,               // 3
        in_c_per_shard_ceil,            // 4
        in_nblocks_c,                   // 5
        params.max_rows_for_reduction,  // 6 - max_rows_for_reduction (FIXED: use calculated value)
        in_cb_id,                       // 7 - in_cb_id_0
        in_cb_id,                       // 8 - in_cb_id_1 (point to in_cb_id for depthwise)
        in_scalar_cb_id_0,              // 9 - in_scalar_cb_id_0
        in_scalar_cb_id_0,              // 10 - in_scalar_cb_id_1 (point to scalar_cb_0 for depthwise)
        clear_value_cb_id,              // 11 - in_idx_cb_id (point to clear_value_cb for depthwise)
        mul_cb_id,                      // 12 - pack_tmp_cb_id (point to mul_cb for depthwise)
        weight_cb_id,                   // 13 - pack_idx_tmp_cb_id (point to weight_cb for depthwise)
        out_cb_id,                      // 14 - right_inc_cb_id (point to out_cb for depthwise)
        out_cb_id,                      // 15 - down_left_wrap_inc_cb_id (point to out_cb for depthwise)
        out_cb_id,                      // 16 - up_left_wrap_inc_cb_id (point to out_cb for depthwise)
        out_cb_id,                      // 17
        out_cb_id,                      // 18 - out_idx_cb_id (point to out_cb for depthwise)
        one_scalar_per_core,            // 19 - one_scalar_per_core
        pre_tilize_cb_id,               // 20 - pre_tilize_cb_id (FIXED: use actual pre_tilize_cb)
        is_output_tiled,                // 21 - is_output_tiled (FIXED: use calculated value)
        is_output_block_format,         // 22
        false,                          // 23 - return_indices
        stride_h,                       // 24
        stride_w,                       // 25
        ashape[1] + pad_h,              // 26 - in_h_padded
        ashape[2] + pad_w,              // 27 - in_w_padded
        eff_kernel_h,                   // 28 - eff_kernel_h
        eff_kernel_w,                   // 29 - eff_kernel_w
        pad_l,                          // 30
        weight_cb_id,                   // 31
        mul_cb_id                       // 32
    };

    // Create reader kernels using pool reader kernel path (same pattern as pool factory)
    std::string reader_kernel_path = "ttnn/cpp/ttnn/operations/pool/generic/device/kernels/dataflow/reader_pool_2d.cpp";

    auto reader0_config = tt::tt_metal::DataMovementConfig{
        .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
        .noc = tt::tt_metal::NOC::RISCV_0_default,
        .compile_args = reader0_ct_args};
    auto reader0_kernel = CreateKernel(program, reader_kernel_path, parallel_config.grid, reader0_config);

    uint64_t weight_buffer_addr = b.buffer()->address();

    // TensorAccessor expects base address as single 32-bit value (like conv2d kernels)
    std::vector<uint32_t> reader_args = {
        static_cast<uint32_t>(weight_buffer_addr),
        1  // weight buffer base address for TensorAccessor
    };
    SetRuntimeArgs(program, reader0_kernel, CoreCoord{0, 0}, reader_args);

    auto reader1_config = tt::tt_metal::DataMovementConfig{
        .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
        .noc = tt::tt_metal::NOC::RISCV_1_default,
        .compile_args = reader1_ct_args};
    // CRITICAL: Only create reader1_kernel when split_reader is true (same as pool factory)
    auto reader1_kernel =
        params.split_reader ? CreateKernel(program, reader_kernel_path, parallel_config.grid, reader1_config) : 0;

    // Create compute kernel using pool compute kernel path
    std::string compute_kernel_path =
        "ttnn/cpp/ttnn/operations/pool/generic/device/kernels/compute/compute_pool_2d.cpp";

    // Get the defines needed for pool kernel compilation (same as pool factory)
    std::map<std::string, std::string> compute_defines = ttnn::operations::pool::get_defines(
        ttnn::operations::pool::Pool2DType::AVG_POOL2D  // Use AVG_POOL2D for depthwise
    );

    // Let the kernel compilation system handle MATH_FIDELITY and DST_ACCUM_MODE automatically
    // based on the ComputeConfig settings. Only set TILE_C_DIM manually.
    compute_defines["TILE_C_DIM"] = std::to_string(tt::constants::TILE_WIDTH);

    // Merge activation defines into compute defines so they get passed to the kernel
    if (fused_activation.has_value()) {
        compute_defines.merge(ttnn::operations::unary::utils::get_defines(
            fused_activation.value().op_type, fused_activation.value().params, "ACTIVATION", "i", a.dtype()));
    }

    auto compute_kernel = CreateKernel(
        program,
        compute_kernel_path,
        parallel_config.grid,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = get_math_fidelity(compute_kernel_config),
            .fp32_dest_acc_en = get_fp32_dest_acc_en(compute_kernel_config),
            .math_approx_mode = false,
            .compile_args = compute_ct_args,
            .defines = compute_defines});

    log_debug(tt::LogOp, "Depthwise conv2d factory: Pool kernels created successfully");
    log_debug(tt::LogOp, "Using reader: {}", reader_kernel_path);
    log_debug(tt::LogOp, "Using compute: {}", compute_kernel_path);
    log_debug(
        tt::LogOp,
        "Grid size: {}x{} = {} cores",
        parallel_config.grid.bounding_box().grid_size().x,
        parallel_config.grid.bounding_box().grid_size().y,
        parallel_config.grid.num_cores());
    log_debug(
        tt::LogOp,
        "out_nhw_per_core={}, expected_sticks={}",
        out_nhw_per_core,
        out_nhw_per_core * in_c_per_shard_ceil / in_nblocks_c);

    // Suppress unused variable warnings for kernel handles (they are stored in the program)
    (void)reader0_kernel;
    (void)reader1_kernel;
    (void)compute_kernel;

    // Create runtime arguments callback to properly set up buffer addresses for pool kernels
    auto override_runtime_arguments_callback =
        [reader0_kernel,
         reader1_kernel,
         compute_kernel,
         in_cb_id,
         out_cb_id,
         raw_in_cb_id,
         weight_cb_id,
         mul_cb_id,
         in_reader_indices_cb_id,
         reader_indices_on_device,
         raw_in_cb](
            const void* operation,
            tt::tt_metal::Program& program,
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<Tensor>& output_tensors) {
            log_debug(tt::LogOp, "Setting up runtime arguments for depthwise conv2d pool kernels");

            const auto& input_tensor = input_tensors.at(0);
            // const auto& weight_tensor = input_tensors.at(1);

            // NOTE: output_tensor not needed since output CB is bound at creation time

            auto src_buffer = input_tensor.buffer();
            // auto weight_buffer = weight_tensor.buffer();

            // Update circular buffer addresses for raw input (critical for pool reader)
            if (input_tensor.is_sharded()) {
                UpdateDynamicCircularBufferAddress(program, raw_in_cb_id, *src_buffer);
            }
            // NOTE: Output CB is already bound to buffer at creation time, no need to update here

            // Update reader indices circular buffer address
            auto reader_indices_buffer = reader_indices_on_device.buffer();
            UpdateDynamicCircularBufferAddress(program, in_reader_indices_cb_id, *reader_indices_buffer);

        };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

}  // namespace conv2d
}  // namespace ttnn::operations::conv
