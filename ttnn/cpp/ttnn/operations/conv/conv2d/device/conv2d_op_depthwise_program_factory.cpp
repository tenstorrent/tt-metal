// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "conv2d_op_depthwise_program_factory.hpp"

#include <cstdint>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>
#include <hostdevcommon/common_values.hpp>
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

namespace ttnn::operations::conv::conv2d::program {

// Forward declare the internal implementation function
static tt::tt_metal::operation::ProgramWithCallbacks multi_core_conv2d_depthwise_impl(
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
    const std::optional<ttnn::operations::unary::UnaryWithParam>& fused_activation,
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
    std::optional<bool> force_split_reader);

}  // namespace ttnn::operations::conv::conv2d::program

namespace ttnn::operations::conv::conv2d::program {

static tt::tt_metal::operation::ProgramWithCallbacks multi_core_conv2d_depthwise_impl(
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
    TT_FATAL(
        groups == ashape[3] && groups == output_channels,
        "Depthwise factory requires groups == input_channels == output_channels");

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

    // Get output shape from sliding window config
    auto output_shape = sliding_window_config.get_output_shape();
    const uint32_t out_h = output_shape[1];
    const uint32_t out_w = output_shape[2];

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

    // Create circular buffers for depthwise convolution (same as pool factory pattern)
    uint32_t next_cb_index = 0;

    // Input CB - using same logic as pool factory with effective_tiles
    const uint32_t in_cb_id = next_cb_index++;
    const uint32_t in_cb_pagesize = tt::tile_size(params.data_format);
    const uint32_t in_cb_npages = params.multi_buffering_factor * std::min(effective_tiles, 3u);
    tt::tt_metal::create_cb(in_cb_id, program, parallel_config.grid, in_cb_pagesize, in_cb_npages, params.data_format);

    // mul_cb - stores results of element-wise multiplication (using effective_tiles logic)
    const uint32_t mul_cb_id = next_cb_index++;
    const uint32_t mul_cb_pagesize = tt::tile_size(params.data_format);
    const uint32_t mul_cb_npages = std::min(effective_tiles, 3u) * params.multi_buffering_factor;
    tt::tt_metal::create_cb(
        mul_cb_id, program, parallel_config.grid, mul_cb_pagesize, mul_cb_npages, params.data_format);

    // weight_cb - stores weight tensors for depthwise conv
    // Weight layout uses face-based format: rows_per_stick = ceil(channels_per_core / 16)
    // Total rows = kernel_h * kernel_w * rows_per_stick, then padded to tile boundary
    const uint32_t weight_cb_id = next_cb_index++;
    // Use weight tensor's data format (may differ from input, e.g., bfp8 weights with bf16 input)
    const tt::DataFormat weight_data_format = datatype_to_dataformat_converter(b.dtype());
    const uint32_t weight_cb_pagesize = tt::tile_size(weight_data_format);
    // Calculate weight tiles matching prepare_conv2d_weights layout
    const uint32_t weight_rows_per_stick =
        (params.in_ntiles_c * tt::constants::TILE_WIDTH + tt::constants::FACE_WIDTH - 1) / tt::constants::FACE_WIDTH;
    const uint32_t weight_total_rows = kernel_h * kernel_w * weight_rows_per_stick;
    const uint32_t weight_padded_rows =
        ((weight_total_rows + tt::constants::TILE_HEIGHT - 1) / tt::constants::TILE_HEIGHT) *
        tt::constants::TILE_HEIGHT;
    const uint32_t weight_ntiles_height = weight_padded_rows / tt::constants::TILE_HEIGHT;
    const uint32_t weight_ntiles_width = params.in_ntiles_c;
    const uint32_t weight_cb_npages = weight_ntiles_height * weight_ntiles_width;
    tt::tt_metal::create_cb(
        weight_cb_id, program, parallel_config.grid, weight_cb_pagesize, weight_cb_npages, weight_data_format);

    // Output CB - match pool factory sizing based on output layout (same as pool factory)
    const bool is_output_tiled = output.layout() == Layout::TILE;
    uint32_t out_cb_pagesize;
    uint32_t out_cb_npages;

    if (is_output_tiled) {
        out_cb_pagesize = tt::tile_size(params.output_data_format);
        out_cb_npages =
            output.shard_spec().value().shape[0] * output.shard_spec().value().shape[1] / tt::constants::TILE_HW;
    } else {
        // Match pool factory logic for non-tiled output
        out_cb_pagesize = std::min(
                              static_cast<uint32_t>(tt::constants::TILE_WIDTH),
                              tt::round_up(output.shard_spec().value().shape[1], tt::constants::TILE_WIDTH)) *
                          params.nbytes;
        out_cb_npages = output.shard_spec().value().shape[0] * params.out_ntiles_c;
    }

    // CRITICAL: Use same CB creation pattern as pool factory to avoid CB ID mismatches
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

    // Clear value CB - stores "clear value" (-inf for maxpool, 0 for avgpool)
    const uint32_t clear_value_cb_id = next_cb_index++;
    tt::tt_metal::create_cb(
        clear_value_cb_id, program, parallel_config.grid, tt::tile_size(params.data_format), 1, params.data_format);

    // Bias CB - only allocated if has_bias is true
    uint32_t bias_cb_id = 32;  // Invalid CB ID by default
    if (has_bias && bias.has_value()) {
        bias_cb_id = next_cb_index++;
        // CRITICAL: Bias tensor is converted to weight dtype in prepare_conv_bias_internal
        // So we must use the WEIGHT dtype here, not the original bias dtype
        const tt::DataFormat bias_data_format = datatype_to_dataformat_converter(b.dtype());
        const uint32_t bias_cb_pagesize = tt::tile_size(bias_data_format);
        // One tile row of bias per output channel tile
        const uint32_t bias_cb_npages = params.in_ntiles_c;  // out_channels = in_channels for depthwise
        tt::tt_metal::create_cb(
            bias_cb_id, program, parallel_config.grid, bias_cb_pagesize, bias_cb_npages, bias_data_format);
    }

    // Bias temp CB - intermediate buffer for tilized data before bias addition
    // This is needed to avoid CB ordering issues when there are multiple tile rows
    // Without this, the second tilization would read stale data from out_cb's front
    // CRITICAL: Use params.data_format (bfloat16) NOT output_data_format (bfloat8_b)
    // because add_tiles_bcast_rows cannot unpack block float formats like bfloat8_b
    uint32_t bias_temp_cb_id = 32;  // Invalid CB ID by default
    if (has_bias && bias.has_value() && is_output_tiled) {
        bias_temp_cb_id = next_cb_index++;
        const uint32_t bias_temp_cb_pagesize = tt::tile_size(params.data_format);
        const uint32_t bias_temp_cb_npages = params.in_ntiles_c;  // Same as output tiles per row
        tt::tt_metal::create_cb(
            bias_temp_cb_id,
            program,
            parallel_config.grid,
            bias_temp_cb_pagesize,
            bias_temp_cb_npages,
            params.data_format);
    }

    // Reader indices will be created after out_nhw_per_core is calculated

    // Now create the actual pool kernels by referencing their paths directly
    // Calculate out_nhw_per_core exactly like pool factory does
    // Use the output tensor's shard specification rather than simple division
    const uint32_t out_nhw_per_core = output.shard_spec()->shape[0];

    // Generate proper top_left_indices using sliding window infrastructure
    // This ensures correct multi-core memory access patterns like pool operations
    // Use the sliding window infrastructure to generate proper reader indices
    // This accounts for stride patterns, shard boundaries, and memory access patterns
    std::vector<std::vector<uint16_t>> top_left_indices = sliding_window::generate_sliding_window_op_config(
        op_trace_metadata, shard_boundaries, stride_w
        // true,  // is_conv = true (for depthwise convolution)
        // 0,     // reader0_datums = 0 (use defaults)
        // 0,     // reader1_datums = 0 (use defaults)
        // true   // pad_cores = true
    );

    Tensor reader_indices = sliding_window::construct_on_host_config_tensor(top_left_indices, parallel_config);

    bool is_block_sharded = (a.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED);
    Tensor reader_indices_on_device =
        sliding_window::move_config_tensor_to_device(reader_indices, parallel_config, is_block_sharded, a.device());

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

    // For depthwise convolution, we don't need the complex CBs that pool factory creates
    // We'll point unused CB arguments to existing valid CBs to avoid kernel hangs

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
        weight_cb_id,                   // 46 - weight_cb_id (for L1 storage)
        num_shards_c                    // 47 - num_shards_c (for 2D tile iteration in weight reading)
    };

    // Add tensor accessor args for weight buffer (similar to sharded factory)
    // This allows the kernel to access weight data from DRAM via TensorAccessor
    tt::tt_metal::TensorAccessorArgs(b.buffer()).append_to(reader0_ct_args);

    // Add bias compile-time args (after weight TensorAccessor args)
    // These will be at positions 48+ after weight accessor args are appended
    reader0_ct_args.push_back(has_bias);            // has_bias flag
    reader0_ct_args.push_back(bias_cb_id);          // bias_cb_id
    reader0_ct_args.push_back(params.in_ntiles_c);  // bias_ntiles (same as out channel tiles for depthwise)

    // ALWAYS add TensorAccessorArgs for bias buffer to maintain consistent compile-time arg positions
    // When has_bias=false, we use a placeholder (dram_interleaved) that won't be accessed at runtime
    if (has_bias && bias.has_value()) {
        tt::tt_metal::TensorAccessorArgs(bias.value().buffer()).append_to(reader0_ct_args);
    } else {
        // Add placeholder TensorAccessorArgs when no bias - kernel needs consistent arg positions
        tt::tt_metal::TensorAccessorArgs::create_dram_interleaved().append_to(reader0_ct_args);
    }

    // Create reader1 arguments by copying reader0 and updating reader_id
    std::vector<uint32_t> reader1_ct_args = reader0_ct_args;
    reader1_ct_args[8] = 1;  // split reader id for reader1

    // Set up compute arguments following exact pool factory pattern (33 args)
    std::vector<uint32_t> compute_ct_args = {
        params.in_ntiles_c,             // 0 - in_ntiles_c (FIXED: use calculated value)
        kernel_h * kernel_w,            // 1
        params.split_reader,            // 2 - split_reader (FIXED: use calculated value)
        0,                              // 3 - max_out_sticks_per_core (0 = use runtime args like pool2d)
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
        mul_cb_id,                      // 32
        has_bias,                       // 33 - has_bias flag for bias addition
        bias_cb_id,                     // 34 - bias_cb_id
        params.in_ntiles_c,             // 35 - bias_ntiles (same as out channel tiles for depthwise)
        bias_temp_cb_id                 // 36 - bias_temp_cb_id (intermediate buffer for tilization before bias)
    };

    // Create reader kernels using pool reader kernel path (same pattern as pool factory)
    std::string reader_kernel_path = "ttnn/cpp/ttnn/operations/pool/generic/device/kernels/dataflow/reader_pool_2d.cpp";

    // Add IS_DEPTHWISE=1 define for depthwise reader kernels
    std::map<std::string, std::string> reader_defines = {{"IS_DEPTHWISE", "1"}};

    auto reader0_config = tt::tt_metal::DataMovementConfig{
        .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
        .noc = tt::tt_metal::NOC::RISCV_0_default,
        .compile_args = reader0_ct_args,
        .defines = reader_defines};
    auto reader0_kernel = CreateKernel(program, reader_kernel_path, parallel_config.grid, reader0_config);

    // ============================================================
    // Weight Multicast Setup
    // ============================================================
    // Create semaphores for weight multicast synchronization
    uint32_t weights_mcast_sender_semaphore_id = tt::tt_metal::CreateSemaphore(program, parallel_config.grid, INVALID);
    uint32_t weights_mcast_receiver_semaphore_id =
        tt::tt_metal::CreateSemaphore(program, parallel_config.grid, INVALID);

    uint64_t weight_buffer_addr = b.buffer()->address();
    auto* device = a.device();

    // Get all cores from the grid and identify sender/receivers
    auto all_cores = corerange_to_cores(parallel_config.grid, std::nullopt, true);
    uint32_t num_cores = all_cores.size();

    // Check sharding mode early for weight distribution
    const bool is_width_sharded = (a.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED);

    // Grid dimensions for block sharding
    const uint32_t num_cores_c = parallelization_config.num_cores_c_in;
    const uint32_t num_cores_r = num_cores_c > 0 ? num_cores / num_cores_c : num_cores;

    // Get bias buffer address if bias exists
    uint64_t bias_buffer_addr = has_bias && bias.has_value() ? bias.value().buffer()->address() : 0;

    // Helper lambda to create sender runtime args
    // Sender args: 0=weight_addr, 1=is_sender, 2-5=mcast coords, 6-7=num_dests, 8-9=semaphores, 10=start_tile_id,
    // 11=bias_addr
    auto make_sender_args = [&](uint32_t start_x,
                                uint32_t start_y,
                                uint32_t end_x,
                                uint32_t end_y,
                                uint32_t num_dests,
                                uint32_t start_tile_id) {
        return std::vector<uint32_t>{
            static_cast<uint32_t>(weight_buffer_addr),
            1,  // is_sender = true
            start_x,
            start_y,
            end_x,
            end_y,
            num_dests,
            num_dests,  // num_dests == num_mcast_cores
            weights_mcast_sender_semaphore_id,
            weights_mcast_receiver_semaphore_id,
            start_tile_id,
            static_cast<uint32_t>(bias_buffer_addr)};  // 11 - bias buffer address for bias reading
    };

    // Helper lambda to create receiver runtime args
    // Receiver args: padded to match sender layout so bias reading works uniformly
    // 0=weight_addr, 1=is_sender, 2-3=sender coords, 4-5=semaphores, 6-9=padding, 10=start_tile_id, 11=bias_addr
    auto make_receiver_args = [&](uint32_t sender_x, uint32_t sender_y, uint32_t start_tile_id) {
        return std::vector<uint32_t>{
            static_cast<uint32_t>(weight_buffer_addr),
            0,  // is_sender = false
            sender_x,
            sender_y,
            weights_mcast_sender_semaphore_id,
            weights_mcast_receiver_semaphore_id,
            0,
            0,
            0,
            0,                                         // padding for args 6-9 to align with sender layout
            start_tile_id,                             // 10 - used for bias reading (same channel offset as sender)
            static_cast<uint32_t>(bias_buffer_addr)};  // 11 - bias buffer address for bias reading
    };

    if (num_cores > 0) {
        if (is_width_sharded) {
            // WIDTH_SHARDED: Each core reads its own weight shard - no multicast
            // Pass start_col_tile_id = i * in_ntiles_c (column offset in tile coordinates)
            for (uint32_t i = 0; i < num_cores; i++) {
                auto args = make_sender_args(0, 0, 0, 0, 0, i * params.in_ntiles_c);  // num_dests=0 skips multicast
                SetRuntimeArgs(program, reader0_kernel, all_cores[i], args);
            }
        } else if (is_block_sharded) {
            // BLOCK_SHARDED: Row 0 cores read from DRAM and multicast to their column
            for (uint32_t i = 0; i < num_cores; i++) {
                CoreCoord core = all_cores[i];
                uint32_t col_idx = i % num_cores_c;
                uint32_t row_idx = i / num_cores_c;
                uint32_t col_start_tile_id = col_idx * params.in_ntiles_c;

                if (row_idx == 0) {
                    // Sender: multicast to entire column
                    // Pass start_col_tile_id = col_idx * in_ntiles_c (column offset in tile coordinates)
                    CoreCoord col_start_phys = device->worker_core_from_logical_core({col_idx, 0});
                    CoreCoord col_end_phys = device->worker_core_from_logical_core({col_idx, num_cores_r - 1});
                    auto args = make_sender_args(
                        col_start_phys.x,
                        col_start_phys.y,
                        col_end_phys.x,
                        col_end_phys.y,
                        num_cores_r - 1,
                        col_start_tile_id);
                    SetRuntimeArgs(program, reader0_kernel, core, args);
                } else {
                    // Receiver: wait for sender in row 0
                    // Pass same start_tile_id as sender (same column = same channels)
                    CoreCoord sender_phys = device->worker_core_from_logical_core({col_idx, 0});
                    SetRuntimeArgs(
                        program,
                        reader0_kernel,
                        core,
                        make_receiver_args(sender_phys.x, sender_phys.y, col_start_tile_id));
                }
            }
        } else {
            // HEIGHT_SHARDED: First core reads and multicasts to all others
            // All cores process same channels, so start_tile_id = 0 for all
            CoreCoord sender_phys = device->worker_core_from_logical_core(all_cores[0]);
            auto grid_start = parallel_config.grid.bounding_box().start_coord;
            auto grid_end = parallel_config.grid.bounding_box().end_coord;
            CoreCoord start_phys = device->worker_core_from_logical_core(grid_start);
            CoreCoord end_phys = device->worker_core_from_logical_core(grid_end);

            // Sender
            auto sender_args = make_sender_args(start_phys.x, start_phys.y, end_phys.x, end_phys.y, num_cores - 1, 0);
            SetRuntimeArgs(program, reader0_kernel, all_cores[0], sender_args);

            // Receivers - all use start_tile_id=0 (same channels as sender)
            for (uint32_t i = 1; i < num_cores; i++) {
                SetRuntimeArgs(
                    program, reader0_kernel, all_cores[i], make_receiver_args(sender_phys.x, sender_phys.y, 0));
            }
        }
    }

    auto reader1_config = tt::tt_metal::DataMovementConfig{
        .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
        .noc = tt::tt_metal::NOC::RISCV_1_default,
        .compile_args = reader1_ct_args,
        .defines = reader_defines};
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
    compute_defines["IS_DEPTHWISE"] = "1";  // Enable depthwise convolution path

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

    // Set runtime args for compute kernel (same pattern as pool2d factory)
    // This passes out_nhw_this_core to each core since compile arg 3 is set to 0
    const uint32_t max_out_nhw_per_core = out_nhw_per_core;
    const uint32_t total_out_nhw = ashape[0] * out_h * out_w;
    const uint32_t rectangular_x = is_block_sharded ? parallel_config.grid.ranges()[0].end_coord.x + 1
                                                    : parallel_config.grid.bounding_box().grid_size().x;

    for (uint32_t core_i = 0; core_i < num_cores; core_i++) {
        const uint32_t core_x_i = core_i % rectangular_x;
        const uint32_t core_y_i = core_i / rectangular_x;
        const CoreRange core(CoreCoord(core_x_i, core_y_i), CoreCoord(core_x_i, core_y_i));

        uint32_t total_out_nhw_processed;
        if (is_block_sharded) {
            total_out_nhw_processed = core_y_i * max_out_nhw_per_core;
        } else if (is_width_sharded) {
            total_out_nhw_processed = 0;
        } else {
            total_out_nhw_processed = core_i * max_out_nhw_per_core;
        }

        uint32_t remaining_out_nhw =
            total_out_nhw_processed < total_out_nhw ? total_out_nhw - total_out_nhw_processed : 0;
        uint32_t out_nhw_this_core = std::min(max_out_nhw_per_core, remaining_out_nhw);
        std::vector<uint32_t> compute_rt_args = {out_nhw_this_core};

        SetRuntimeArgs(program, compute_kernel, core, compute_rt_args);
    }

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
         raw_in_cb,
         has_bias,
         all_cores](
            const void* operation,
            tt::tt_metal::Program& program,
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<Tensor>& output_tensors) {
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

            // Update bias buffer address in runtime args if bias is present
            if (has_bias && !optional_input_tensors.empty() && optional_input_tensors.at(0).has_value()) {
                const auto& bias_tensor = optional_input_tensors.at(0).value();
                uint32_t bias_buffer_addr = static_cast<uint32_t>(bias_tensor.buffer()->address());

                // Update runtime arg 11 (bias_buffer_addr) for all cores
                for (const auto& core : all_cores) {
                    auto& runtime_args = GetRuntimeArgs(program, reader0_kernel, core);
                    runtime_args[11] = bias_buffer_addr;
                }
            }
        };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

// New infrastructure wrapper methods
Conv2dDepthwiseProgramFactory::cached_program_t Conv2dDepthwiseProgramFactory::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output_tensor) {
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

    // Call the implementation function
    auto program_with_callbacks = multi_core_conv2d_depthwise_impl(
        program,
        a,
        b,
        ashape,
        bias,
        sliding_window_config,
        parallel_config,
        op_trace_metadata,
        shard_boundaries,
        output_channels,
        groups,
        untilize_out,
        has_bias,
        fused_activation,
        parallelization_config,
        block_config,
        transpose_mcast,
        output,
        compute_kernel_config,
        enable_act_double_buffer,
        enable_weights_double_buffer,
        full_inner_dim,
        enable_activation_reuse,
        config_tensors_in_dram,
        force_split_reader);

    // Create shared variables (simplified for now - can be expanded based on actual needs)
    shared_variables_t shared_vars;
    shared_vars.has_bias = has_bias;

    return cached_program_t{std::move(program_with_callbacks.program), std::move(shared_vars)};
}

void Conv2dDepthwiseProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output_tensor) {
    // For now, the depthwise implementation's runtime argument handling
    // is embedded in the program itself via the ProgramWithCallbacks pattern.
    // In the future, this can be refactored to directly update runtime arguments
    // similar to how Conv2dShardedProgramFactory does it.

    // TODO: Refactor to follow the pattern of directly updating runtime arguments
    // like Conv2dShardedProgramFactory does
    (void)cached_program;
    (void)operation_attributes;
    (void)tensor_args;
    (void)output_tensor;
}

}  // namespace ttnn::operations::conv::conv2d::program
